#!/usr/bin/env python3
"""Inference for v23 (LocalEnhancer architecture).

Usage:
    MAMA_WEIGHTS_PATH=weights_v23/latest_net_G.pth \
    MAMA_INPUT_DIR=/path/to/test/mha \
    MAMA_OUTPUT_DIR=predictions_v23 \
    python inference_v23.py
"""
import os
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src/submission/submission-synthesis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src/submission/submission-synthesis/models"))
from networks import LocalEnhancer

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "predictions_v23"))
INPUT_SLUG = "pre-contrast-dce-mri-slice-breast"
OUTPUT_SLUG = "synthetic-contrast-dce-mri-slice-breast"
WEIGHTS_PATH = os.environ.get("MAMA_WEIGHTS_PATH", "weights_v23/latest_net_G.pth")
MODEL_SIZE = 512


def build_generator(device):
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = LocalEnhancer(
        input_nc=1, output_nc=1, ngf=32,
        n_downsample_global=4, n_blocks_global=9,
        n_local_enhancers=1, n_blocks_local=3,
        norm_layer=norm_layer,
    )
    netG.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    netG.to(device).eval()
    print(f"LocalEnhancer loaded from {WEIGHTS_PATH}")
    return netG


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = build_generator(device)

    # Find all input files
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    if search_dir.exists():
        files = sorted(search_dir.glob("*.mha"))
    else:
        files = sorted(INPUT_PATH.glob("input/*.mha"))
        if not files:
            files = sorted(INPUT_PATH.glob("*.mha"))

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(files)} files...")
    for i, f in enumerate(files):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        sl = arr.squeeze()
        orig_h, orig_w = sl.shape

        t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
        if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
            t = torch.nn.functional.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode='bilinear', align_corners=False)

        with torch.no_grad():
            out = netG(t)
        # residual mode: network outputs Δ, result = pre + Δ
        out = t + out

        if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
            out = torch.nn.functional.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        result = out[0, 0].cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            result = result[np.newaxis, ...]

        out_img = sitk.GetImageFromArray(result)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_dir / f.name))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {f.name}")

    # Also save flat (for evaluation compatibility)
    for mha in (out_dir).glob("*.mha"):
        flat_path = OUTPUT_PATH / mha.name
        if not flat_path.exists():
            os.symlink(mha, flat_path)

    print(f"Done. {len(files)} predictions in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
