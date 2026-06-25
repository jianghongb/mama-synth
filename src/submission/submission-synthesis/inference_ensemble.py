#!/usr/bin/env python3
"""MAMA-SYNTH GC inference — K-Fold Ensemble (5 models averaged).

Pipeline:
  1. Load pre-contrast input
  2. Resize to 512×512
  3. Run 5 Pix2PixHD generators, average outputs
  4. Resize back + write output

Grand Challenge I/O:
  Input:  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
  Output: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
"""
import os
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(__file__))
from models.networks import GlobalGenerator

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = "pre-contrast-dce-mri-slice-breast"
OUTPUT_SLUG = "synthetic-contrast-dce-mri-slice-breast"
WEIGHTS_DIR = Path(os.environ.get("MAMA_WEIGHTS_DIR", "/opt/app/weights"))
MODEL_SIZE = 512


def load_models(device):
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    models = []
    for i in range(5):
        w = WEIGHTS_DIR / f"fold_{i}_net_G.pth"
        if not w.exists():
            continue
        netG = GlobalGenerator(1, 1, 64, 4, 9, norm_layer, residual_mode=True)
        netG.load_state_dict(torch.load(str(w), map_location=device))
        netG.to(device).eval()
        models.append(netG)
    print(f"Loaded {len(models)} ensemble models")
    return models


def find_input_image() -> Path:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = list(search_dir.glob("*.mha"))
    if not candidates:
        raise FileNotFoundError(f"No .mha found in {search_dir}")
    return candidates[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # Resize to 512
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        t = torch.nn.functional.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode='bilinear', align_corners=False)

    # Ensemble: average all models
    preds = []
    with torch.no_grad():
        for netG in models:
            preds.append(netG(t))
    result = torch.stack(preds).mean(dim=0)

    # Resize back
    if orig_h != MODEL_SIZE or orig_w != MODEL_SIZE:
        result = torch.nn.functional.interpolate(result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    out_arr = result[0, 0].cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        out_arr = out_arr[np.newaxis, ...]

    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={out_arr.shape}  models={len(models)}")


if __name__ == "__main__":
    main()
