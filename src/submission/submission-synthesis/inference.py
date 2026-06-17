#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – Docker inference entry point.

Pipeline:
  1. Load pre-contrast input
  2. Resize to 512×512
  3. Pix2PixHD synthesis (residual mode: output = input + Δ)
  4. Resize back to original resolution
  5. Write output preserving spatial metadata

Note: No breast mask at inference time. The model was trained with breast-masked
loss, so it has already learned to only enhance the breast region. Adding inference
mask actually hurts MSE/SSIM/AUROC (see v11 ablation results).

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
MODEL_SIZE = 512


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


def find_input_image() -> Path:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = list(search_dir.glob("*.mha"))
    if not candidates:
        raise FileNotFoundError(f"No .mha found in {search_dir}")
    return candidates[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = build_generator(device)

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

    # Pix2PixHD inference
    with torch.no_grad():
        out = netG(t)

    # Resize back
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        out = torch.nn.functional.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    result = out[0, 0].cpu().numpy().astype(np.float32)

    # Restore original ndim
    if arr.ndim == 3:
        result = result[np.newaxis, ...]

    # Write output preserving spatial metadata
    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}")


if __name__ == "__main__":
    main()
