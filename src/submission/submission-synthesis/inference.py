#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – Docker inference entry point.

Grand Challenge I/O contract:
  Input:  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
  Output: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha

Each job receives exactly ONE input file; output must be named output.mha.
"""
import os
import sys
from glob import glob
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
PAD_BASE = 16


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

    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # 2D slice input from GC
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # Resize to 512x512 (model trained at this resolution)
    MODEL_SIZE = 512
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        t_input = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        t_input = torch.nn.functional.interpolate(t_input, size=(MODEL_SIZE, MODEL_SIZE), mode='bilinear', align_corners=False)
        sl_resized = t_input[0, 0].numpy()
    else:
        sl_resized = sl

    # Pad to multiple of 16 (should be no-op for 512)
    padded, (ph, pw) = pad_to_multiple(sl_resized)

    # Inference
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = netG(t)
    result_512 = out[0, 0, :ph, :pw].cpu().numpy()

    # Resize back to original resolution
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        result = torch.nn.functional.interpolate(
            torch.from_numpy(result_512).unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )[0, 0].numpy().astype(np.float32)
    else:
        result = result_512.astype(np.float32)

    # Restore original ndim
    if arr.ndim == 3:
        result = result[np.newaxis, ...]

    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(img)

    # Write to GC output path
    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}")


if __name__ == "__main__":
    main()
