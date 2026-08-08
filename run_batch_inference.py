#!/usr/bin/env python3
"""Batch inference over the 299-case test set for MSEC ablation.

Replicates steps 4-7 of src/submission/submission-synthesis/inference.py
exactly, but iterates over the whole test set and reuses the precomputed
breast / tumor masks in data_multislice_v3/test/mha/ instead of re-running
nnU-Net per case. The GAN forward path, per-image normalization,
de-normalization, hflip TTA and Gaussian-blur composite are unchanged.

Both the ablation weights and the baseline weights are meant to be run
through THIS script, so that the comparison is single-variable regardless
of how the historical predictions_* directories were produced.

Usage:
    python run_batch_inference.py --weights <path> --out <dir> [--limit N]
"""
import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

SUBMISSION_DIR = Path(__file__).parent / "src" / "submission" / "submission-synthesis"
sys.path.insert(0, str(SUBMISSION_DIR))
from models.networks import GlobalGenerator  # noqa: E402

TEST_ROOT = Path("/Users/ehogjig/git/kth/data_multislice_v3/test/mha")
MODEL_SIZE = 512


def build_generator(weights_path: str, device: torch.device,
                    ngf: int = 64, n_blocks: int = 12) -> nn.Module:
    """Load the 3-channel conditional Pix2PixHD generator in residual mode."""
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(
        input_nc=3, output_nc=1, ngf=ngf,
        n_downsampling=4, n_blocks=n_blocks,
        norm_layer=norm_layer, residual_mode=True,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    netG.load_state_dict(state)
    netG.to(device).eval()
    return netG


def load_slice(path: Path) -> np.ndarray:
    """Read a 2D .mha slice as float32, squeezing any singleton axis."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32).squeeze()


def synthesize(netG: nn.Module, sl: np.ndarray, breast_mask: np.ndarray,
               tumor_mask: np.ndarray, device: torch.device,
               use_tta: bool = True, blend: str = "blur") -> np.ndarray:
    """Run one case through normalization, generator, de-normalization, composite.

    Args:
        sl: pre-contrast slice in global z-score space.
        breast_mask: binary breast mask, same shape as sl.
        tumor_mask: binary predicted tumor mask, same shape as sl.
        use_tta: average the prediction with its horizontally flipped pass.
        blend: 'blur' for Gaussian-smoothed composite, 'hard' for binary mask.

    Returns:
        Synthetic post-contrast slice in global z-score space.
    """
    orig_h, orig_w = sl.shape

    # Per-image z-score over breast foreground, background zeroed.
    fg = sl[breast_mask > 0.5]
    if fg.size > 100:
        img_mean = float(fg.mean())
        img_std = max(float(fg.std()), 1e-8)
    else:
        img_mean, img_std = 0.0, 1.0
    sl_norm = (sl - img_mean) / img_std * breast_mask

    inp = np.stack([sl_norm, breast_mask, tumor_mask], axis=0)
    t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
    t = F.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE),
                      mode="bilinear", align_corners=False)

    with torch.no_grad():
        out = netG(t)
        if use_tta:
            out = (out + netG(t.flip(-1)).flip(-1)) / 2.0

    out = F.interpolate(out, size=(orig_h, orig_w),
                        mode="bilinear", align_corners=False)
    synthetic = out[0, 0].cpu().numpy() * img_std + img_mean

    if blend == "blur":
        soft = np.clip(gaussian_filter(breast_mask, sigma=5.0), 0, 1)
    else:
        soft = breast_mask
    return soft * synthetic + (1 - soft) * sl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Generator .pth path")
    ap.add_argument("--out", required=True, help="Output directory for predictions")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N cases (0 = all)")
    ap.add_argument("--no-tta", action="store_true", help="Disable hflip TTA")
    ap.add_argument("--blend", choices=["blur", "hard"], default="blur")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device
    device = torch.device(dev)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    netG = build_generator(args.weights, device)
    cases = sorted((TEST_ROOT / "input").glob("*.mha"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"device={dev}  weights={Path(args.weights).name}")
    print(f"cases={len(cases)}  tta={not args.no_tta}  blend={args.blend}")
    print(f"out={out_dir}")

    t0 = time.time()
    for i, f in enumerate(cases, 1):
        sl = load_slice(f)
        bm = (load_slice(TEST_ROOT / "breast_mask" / f.name) > 0.5).astype(np.float32)
        tm = (load_slice(TEST_ROOT / "predicted_tumor" / f.name) > 0.5).astype(np.float32)

        result = synthesize(netG, sl, bm, tm, device,
                            use_tta=not args.no_tta, blend=args.blend)

        ref = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(ref)
        if arr.ndim == 3:
            result = result[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(ref)
        sitk.WriteImage(out_img, str(out_dir / f.name))

        if i % 25 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"  {i}/{len(cases)}  {el:.0f}s  ({el/i:.2f}s/case)")

    print(f"Done: {len(cases)} cases in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
