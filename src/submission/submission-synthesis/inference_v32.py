#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – v32 Conditional GAN Docker inference.

Pipeline:
  1. BreastDivider 2D → breast_mask
  2. T1w Tumor Seg (Dataset940) → predicted_tumor_mask
  3. Per-image z-score normalize (breast foreground)
  4. GAN(concat(pre_norm, breast_mask, tumor_mask)) → synthetic_norm
  5. De-normalize → synthetic
  6. Breast mask composite: output = breast_mask × synthetic + (1-breast_mask) × pre

Grand Challenge I/O contract:
  Input:  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
  Output: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
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

sys.path.insert(0, os.path.dirname(__file__))
from models.networks import GlobalGenerator

# === Paths ===
INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = "pre-contrast-dce-mri-slice-breast"
OUTPUT_SLUG = "synthetic-contrast-dce-mri-slice-breast"

WEIGHTS_PATH = os.environ.get("MAMA_WEIGHTS_PATH", "/opt/app/weights/latest_net_G.pth")
BREAST_SEG_PATH = os.environ.get("MAMA_BREAST_SEG_PATH", "/opt/app/weights/breast_seg")
TUMOR_SEG_PATH = os.environ.get("MAMA_TUMOR_SEG_PATH", "/opt/app/weights/tumor_seg")

MODEL_SIZE = 512


def build_generator(device):
    """Load v32 conditional GAN (input_nc=3)."""
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(
        input_nc=3, output_nc=1, ngf=64,
        n_downsampling=4, n_blocks=12,
        norm_layer=norm_layer, residual_mode=True,
    )
    netG.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    netG.to(device).eval()
    return netG


def build_breast_seg(device):
    """Load BreastDivider 2D (Dataset930) for breast segmentation."""
    os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=device,
        verbose=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        BREAST_SEG_PATH, use_folds=(0,), checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def build_tumor_seg(device):
    """Load T1w Tumor Seg (Dataset940)."""
    os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=device,
        verbose=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        TUMOR_SEG_PATH, use_folds=(0,), checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def predict_mask(predictor, slice_2d):
    """Run nnUNet prediction on a 2D slice, return binary mask."""
    props = {
        "sitk_stuff": {
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
        "spacing": [1.0, 1.0, 1.0],
    }
    input_arr = slice_2d[np.newaxis, np.newaxis, :, :]
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    return (pred > 0).astype(np.float32)


def postprocess_breast_mask(mask):
    """Clean up breast mask: remove small components."""
    from scipy import ndimage as ndi

    bm = mask.astype(np.uint8)
    labeled, n = ndi.label(bm)
    if n > 1:
        sizes = ndi.sum(bm, labeled, range(1, n + 1))
        thresh = sizes.max() * 0.1
        bm = np.zeros_like(bm, dtype=np.uint8)
        for i, s in enumerate(sizes):
            if s >= thresh:
                bm[labeled == (i + 1)] = 1
    return bm.astype(np.float32)


def find_input_image() -> Path:
    """Find the input MHA file."""
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates = list(search_dir.glob("*.mha"))
    if not candidates:
        raise FileNotFoundError(f"No .mha found in {search_dir}")
    return candidates[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    netG = build_generator(device)
    breast_seg = build_breast_seg(device)
    tumor_seg = build_tumor_seg(device)
    print("Models loaded.")

    # Read input
    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # Step 1: Breast segmentation
    breast_mask = predict_mask(breast_seg, sl)
    breast_mask = postprocess_breast_mask(breast_mask)

    # Step 2: Tumor segmentation
    tumor_mask = predict_mask(tumor_seg, sl)

    # Step 3: Per-image z-score normalize (breast foreground)
    fg = sl[breast_mask > 0.5]
    if fg.size > 100:
        mu = float(fg.mean())
        sigma = max(float(fg.std()), 1e-8)
    else:
        mu, sigma = 0.0, 1.0
    sl_norm = (sl - mu) / sigma * breast_mask

    # Step 4: Build 3-channel input and run GAN
    input_3ch = np.stack([sl_norm, breast_mask, tumor_mask], axis=0)  # (3, H, W)
    t = torch.from_numpy(input_3ch).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    t = F.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)

    with torch.no_grad():
        out_norm = netG(t)

    # Step 5: De-normalize
    result_norm = F.interpolate(out_norm, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    synthetic = result_norm[0, 0].cpu().numpy() * sigma + mu

    # Step 6: Breast mask composite
    result = breast_mask * synthetic + (1 - breast_mask) * sl

    # Write output
    if arr.ndim == 3:
        result = result[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(result.astype(np.float32))
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}")


if __name__ == "__main__":
    main()
