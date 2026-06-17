#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – Docker inference entry point.

Pipeline:
  1. Load pre-contrast input
  2. Resize to 512×512
  3. Pix2PixHD synthesis (residual mode: output = input + Δ)
  4. Resize back to original resolution
  5. Write output preserving spatial metadata

Grand Challenge I/O contract:
  Input:  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
  Output: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(__file__))
from models.networks import GlobalGenerator

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = "pre-contrast-dce-mri-slice-breast"
OUTPUT_SLUG = "synthetic-contrast-dce-mri-slice-breast"
WEIGHTS_PATH = "/opt/app/weights/latest_net_G.pth"
BREAST_SEG_DIR = "/opt/app/weights/breast_seg/nnUNetTrainer__nnUNetResEncUNetLPlans__2d"
PAD_BASE = 16
MODEL_SIZE = 512
BREAST_LABELS = {1, 2, 5, 6, 9}  # Dataset910 labels that form breast region


def build_generator(device):
    from functools import partial
    import torch.nn as nn
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(
        input_nc=1, output_nc=1, ngf=64,
        n_downsampling=4, n_blocks=9,
        norm_layer=norm_layer, residual_mode=True
    )
    netG.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    netG.to(device).eval()
    return netG


def build_breast_segmentor(device):
    """Load Dataset910 nnUNet for breast segmentation (single-channel T1 input)."""
    if not Path(BREAST_SEG_DIR).exists():
        print(f"Breast seg model not found at {BREAST_SEG_DIR}, skipping masking.")
        return None
    os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=device,
        verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        BREAST_SEG_DIR, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    return predictor


def generate_breast_mask(predictor, sl_2d):
    """Generate breast mask from a 2D slice using Dataset910."""
    input_arr = sl_2d[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    props = {'sitk_stuff': {'spacing': (1.0, 1.0, 1.0), 'origin': (0,0,0),
              'direction': (1,0,0,0,1,0,0,0,1)}, 'spacing': [1.0, 1.0, 1.0]}
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    return np.isin(pred, list(BREAST_LABELS)).astype(np.float32)


def find_input_image() -> Path:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = list(search_dir.glob("*.mha"))
    if not candidates:
        raise FileNotFoundError(f"No .mha found in {search_dir}")
    return candidates[0]


def pad_to_multiple(arr, base=PAD_BASE):
    h, w = arr.shape[-2:]
    new_h = ((h - 1) // base + 1) * base
    new_w = ((w - 1) // base + 1) * base
    if new_h == h and new_w == w:
        return arr, (h, w)
    padded = np.zeros(arr.shape[:-2] + (new_h, new_w), dtype=arr.dtype)
    padded[..., :h, :w] = arr
    return padded, (h, w)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = build_generator(device)
    breast_predictor = build_breast_segmentor(device)

    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # --- Step 1: Generate breast mask ---
    breast_mask = None
    if breast_predictor is not None:
        breast_mask = generate_breast_mask(breast_predictor, sl)
        print(f"  Breast mask: {breast_mask.sum():.0f}/{breast_mask.size} pixels ({100*breast_mask.mean():.1f}%)")

    # --- Step 2: Resize to 512×512 ---
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        t_input = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
        t_input = torch.nn.functional.interpolate(t_input, size=(MODEL_SIZE, MODEL_SIZE), mode='bilinear', align_corners=False)
        sl_resized = t_input[0, 0].numpy()
        if breast_mask is not None:
            bm_resized = torch.nn.functional.interpolate(
                torch.from_numpy(breast_mask).unsqueeze(0).unsqueeze(0),
                size=(MODEL_SIZE, MODEL_SIZE), mode='nearest'
            )[0, 0].numpy()
        else:
            bm_resized = None
    else:
        sl_resized = sl
        bm_resized = breast_mask

    # --- Step 3: Apply breast mask to input ---
    if bm_resized is not None:
        sl_masked = sl_resized * bm_resized
    else:
        sl_masked = sl_resized

    padded, (ph, pw) = pad_to_multiple(sl_masked)

    # --- Step 4: Pix2PixHD inference ---
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = netG(t)
    result_512 = out[0, 0, :ph, :pw].cpu().numpy()

    # --- Step 5: Fill chest wall with pre-contrast values ---
    if bm_resized is not None:
        result_512 = result_512 * bm_resized + sl_resized * (1 - bm_resized)

    # --- Step 6: Resize back to original resolution ---
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        result = torch.nn.functional.interpolate(
            torch.from_numpy(result_512).unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )[0, 0].numpy().astype(np.float32)
    else:
        result = result_512.astype(np.float32)

    # Restore original ndim
    if arr.ndim == 3:
        result = result[np.newaxis, ...]

    # --- Step 7: Write output ---
    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}")


if __name__ == "__main__":
    main()
