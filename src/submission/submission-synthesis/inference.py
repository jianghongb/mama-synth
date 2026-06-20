#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – Docker inference entry point.

Two-stage pipeline:
  Stage 1: Pix2PixHD GAN → coarse synthesis (~1s)
  Stage 2 (optional): SDEdit diffusion refinement → sharper tumor boundaries (~5-8s)

If refiner weights are not found, falls back to GAN-only (backward compatible).

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
WEIGHTS_PATH = os.environ.get("MAMA_WEIGHTS_PATH", "/opt/app/weights/latest_net_G.pth")
REFINER_PATH = os.environ.get("MAMA_REFINER_PATH", "/opt/app/weights/refiner_latest.pth")
RESREFINER_PATH = os.environ.get("MAMA_RESREFINER_PATH", "/opt/app/weights/resrefiner_latest.pth")
MODEL_SIZE = 512

# SDEdit parameters
SDEDIT_STRENGTH = float(os.environ.get("MAMA_SDEDIT_STRENGTH", "0.3"))
DDIM_STEPS = int(os.environ.get("MAMA_DDIM_STEPS", "20"))


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


def build_refiner(device):
    """Load diffusion refiner if weights exist, otherwise return None."""
    if not Path(REFINER_PATH).exists():
        return None
    from models.refiner_network import RefinerUNet
    refiner = RefinerUNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4, 4))
    refiner.load_state_dict(torch.load(REFINER_PATH, map_location=device))
    refiner.to(device).eval()
    print(f"Diffusion refiner loaded from {REFINER_PATH}")
    return refiner


def build_residual_refiner(device):
    """Load residual refiner if weights exist, otherwise return None."""
    if not Path(RESREFINER_PATH).exists():
        return None
    from models.train_residual_refiner import ResidualRefiner
    resrefiner = ResidualRefiner(base_ch=64)
    resrefiner.load_state_dict(torch.load(RESREFINER_PATH, map_location=device))
    resrefiner.to(device).eval()
    print(f"Residual refiner loaded from {RESREFINER_PATH}")
    return resrefiner


def cosine_alpha_bar(T, s=0.008):
    """Precompute ᾱ schedule."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar.float()


@torch.no_grad()
def sdedit_refine(refiner, gan_output, pre, device, strength=0.3, num_steps=20, T=1000):
    """SDEdit: add noise to GAN output, then denoise with DDIM.

    Args:
        refiner: RefinerUNet model
        gan_output: (1, 1, H, W) GAN synthesis result
        pre: (1, 1, H, W) pre-contrast input
        strength: fraction of noise schedule to use (0.3 = start from t=300)
        num_steps: DDIM sampling steps
    """
    alpha_bar = cosine_alpha_bar(T).to(device)

    # Determine starting timestep
    t_start = int(T * strength)
    if t_start == 0:
        return gan_output

    # Add noise to GAN output at t_start
    ab = alpha_bar[t_start]
    noise = torch.randn_like(gan_output)
    x_t = torch.sqrt(ab) * gan_output + torch.sqrt(1 - ab) * noise

    # DDIM sampling from t_start → 0
    timesteps = torch.linspace(t_start, 0, num_steps + 1).long().to(device)

    for i in range(num_steps):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]

        # Predict noise
        t_batch = t_cur.unsqueeze(0)
        refiner_input = torch.cat([x_t, pre, gan_output], dim=1)
        eps_pred = refiner(refiner_input, t_batch)

        # DDIM deterministic update
        ab_cur = alpha_bar[t_cur]
        ab_next = alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1 - ab_cur) * eps_pred) / torch.sqrt(ab_cur)

        # Step to t_next
        if t_next > 0:
            x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_pred
        else:
            x_t = x0_pred

    return x_t


def find_input_image() -> Path:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = list(search_dir.glob("*.mha"))
    if not candidates:
        raise FileNotFoundError(f"No .mha found in {search_dir}")
    return candidates[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    netG = build_generator(device)
    resrefiner = build_residual_refiner(device)
    refiner = build_refiner(device) if resrefiner is None else None

    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # Resize to 512×512
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        t = torch.nn.functional.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode='bilinear', align_corners=False)
    pre_512 = t

    # Stage 1: Pix2PixHD
    with torch.no_grad():
        gan_out = netG(pre_512)

    # Stage 2: Refinement (if available)
    if resrefiner is not None:
        # Residual refiner: single forward pass, output = pre + Δ
        with torch.no_grad():
            result_512 = resrefiner(pre_512, gan_out)
        mode = "residual"
    elif refiner is not None:
        result_512 = sdedit_refine(refiner, gan_out, pre_512, device,
                                   strength=SDEDIT_STRENGTH, num_steps=DDIM_STEPS)
        mode = "sdedit"
    else:
        result_512 = gan_out
        mode = "gan_only"

    # Resize back
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        result_512 = torch.nn.functional.interpolate(result_512, size=(orig_h, orig_w),
                                                     mode='bilinear', align_corners=False)

    result = result_512[0, 0].cpu().numpy().astype(np.float32)

    # Restore original ndim
    if arr.ndim == 3:
        result = result[np.newaxis, ...]

    # Write output
    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}  mode={mode}")


if __name__ == "__main__":
    main()
