#!/usr/bin/env python3
"""Batch prediction for the GAN v34_seg checkpoint over the data_multislice_v3
test set (299 slices), reusing the precomputed breast/tumour masks so the result
is directly comparable to the FM / I2SB / GAN evaluations in the DSB repo.

This replicates the per-case logic of
``src/submission/submission-synthesis/inference.py`` (per-image z-score over the
breast foreground, 3-channel conditional input, hflip TTA, residual generator,
soft-breast composite) but:
  * reads breast_mask and predicted_tumor from disk instead of running nnU-Net,
  * iterates over all test .mha files,
  * writes one output .mha per case named like the input, into a flat folder
    the MAMA-SYNTH evaluator can consume.

Usage:
    MAMA_WEIGHTS_PATH=/Users/ehogjig/git/kth/mama-synth/weights_v34/latest_net_G.pth \
    bash  (or just run with python; env vars have sensible defaults below)

Env (all optional, defaults shown):
    MAMA_WEIGHTS_PATH  weights_v34/latest_net_G.pth
    TEST_ROOT          /Users/ehogjig/git/kth/data_multislice_v3/test/mha
    OUT_DIR            mama-synth/predictions_v34_seg
    MAMA_NGF=64  MAMA_N_BLOCKS=12  (must match training)
    DEVICE             mps (falls back to cpu)
"""
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

REPO = Path("/Users/ehogjig/git/kth/mama-synth")
sys.path.insert(0, str(REPO / "src/submission/submission-synthesis"))
from models.networks import GlobalGenerator  # noqa: E402

MODEL_SIZE = 512
NGF = int(os.environ.get("MAMA_NGF", "64"))
N_BLOCKS = int(os.environ.get("MAMA_N_BLOCKS", "12"))
WEIGHTS = os.environ.get("MAMA_WEIGHTS_PATH", str(REPO / "weights_v34/latest_net_G.pth"))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/Users/ehogjig/git/kth/data_multislice_v3/test/mha"))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO / "predictions_v34_seg")))
DEVICE = os.environ.get("DEVICE", "mps")


def read_slice(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32).squeeze()


def build_generator(device):
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(input_nc=3, output_nc=1, ngf=NGF,
                           n_downsampling=4, n_blocks=N_BLOCKS,
                           norm_layer=norm_layer, residual_mode=True)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    netG.load_state_dict(sd)
    return netG.to(device).eval()


def main():
    device = torch.device(DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu")
    print(f"device={device}  weights={WEIGHTS}  ngf={NGF} n_blocks={N_BLOCKS}")
    netG = build_generator(device)

    in_dir = TEST_ROOT / "input"
    breast_dir = TEST_ROOT / "breast_mask"
    tumor_dir = TEST_ROOT / "predicted_tumor"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.mha"))
    print(f"{len(files)} test cases -> {OUT_DIR}")
    for i, f in enumerate(files, 1):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        sl = arr.squeeze()
        orig_h, orig_w = sl.shape

        breast = (read_slice(breast_dir / f.name) > 0.5).astype(np.float32)
        tpath = tumor_dir / f.name
        tumor = (read_slice(tpath) > 0.5).astype(np.float32) if tpath.exists() else np.zeros_like(sl)

        fg = sl[breast > 0.5]
        if fg.size > 100:
            mu, sigma = float(fg.mean()), max(float(fg.std()), 1e-8)
        else:
            mu, sigma = 0.0, 1.0
        sl_norm = (sl - mu) / sigma * breast

        inp = np.stack([sl_norm, breast, tumor], axis=0)
        t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
        t = F.interpolate(t, (MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = (netG(t) + netG(t.flip(-1)).flip(-1)) / 2.0
        out = F.interpolate(out, (orig_h, orig_w), mode="bilinear", align_corners=False)
        synthetic = out[0, 0].cpu().numpy() * sigma + mu

        soft = np.clip(gaussian_filter(breast, sigma=5.0), 0, 1)
        result = soft * synthetic + (1 - soft) * sl

        if arr.ndim == 3:
            result = result[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(OUT_DIR / f.name))
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f.name}")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
