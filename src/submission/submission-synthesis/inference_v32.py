#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge – Final Submission Inference.

Pipeline:
  1. Dataset910 (10-class) → breast_mask + heart_mask
  2. Per-image z-score normalize (breast foreground)
  3. Pix2PixHD GAN with hflip TTA (2x inference, averaged)
  4. De-normalize back to global z-score space
  5. Composite: breast=synthetic, heart=pre+adaptive_offset, other_bg=pre

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

MODEL_SIZE = 512
NGF = int(os.environ.get("MAMA_NGF", "64"))
N_BLOCKS = int(os.environ.get("MAMA_N_BLOCKS", "12"))


def build_generator(device):
    """Load Pix2PixHD generator (1-channel input, per-image norm, residual mode)."""
    norm_layer = partial(nn.InstanceNorm2d, affine=False)
    netG = GlobalGenerator(
        input_nc=1, output_nc=1, ngf=NGF,
        n_downsampling=4, n_blocks=N_BLOCKS,
        norm_layer=norm_layer, residual_mode=True,
    )
    netG.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    netG.to(device).eval()
    return netG


def build_breast_seg(device):
    """Load Dataset910 nnUNet (10-class) for breast + heart segmentation."""
    if not Path(BREAST_SEG_PATH).exists():
        print(f"WARNING: Breast seg not found at {BREAST_SEG_PATH}, using threshold fallback")
        return None
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
        BREAST_SEG_PATH, use_folds=(0,), checkpoint_name="checkpoint_best.pth",
    )
    return predictor


def predict_segmentation(predictor, slice_2d):
    """Run nnUNet 10-class prediction, return label map.

    Labels: 0=bg, 1=tissue, 2=vessel, 3=muscle, 4=bone,
            5=lesion, 6=lymphnode, 7=heart, 8=liver, 9=implant
    """
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
    return pred


def get_masks(seg_map):
    """Extract breast mask and heart mask from 10-class segmentation."""
    breast_mask = np.isin(seg_map, [1, 2, 5, 6, 9]).astype(np.float32)
    heart_mask = (seg_map == 7).astype(np.float32)
    return breast_mask, heart_mask


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
    print(f"Generator: ngf={NGF}, n_blocks={N_BLOCKS}, residual_mode")
    print("Models loaded.")

    # Read input
    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # === Step 1: Segmentation → breast_mask + heart_mask ===
    if breast_seg is not None:
        seg_map = predict_segmentation(breast_seg, sl)
        breast_mask, heart_mask = get_masks(seg_map)
        # Dilate heart mask slightly (segmentation boundary can be tight)
        from scipy.ndimage import binary_dilation
        heart_mask = binary_dilation(heart_mask, iterations=3).astype(np.float32)
    else:
        # Fallback: threshold-based breast mask, no heart
        breast_mask = (sl > -0.3).astype(np.float32)
        heart_mask = np.zeros_like(sl)

    # === Step 2: Per-image z-score normalize (breast foreground) ===
    fg_pixels = sl[breast_mask > 0.5]
    if fg_pixels.size > 100:
        img_mean = float(fg_pixels.mean())
        img_std = float(fg_pixels.std())
        img_std = max(img_std, 1e-8)
    else:
        img_mean, img_std = 0.0, 1.0

    sl_norm = (sl - img_mean) / img_std * breast_mask  # zero background

    # === Step 3: GAN inference with hflip TTA ===
    t = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    t = F.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)

    with torch.no_grad():
        # Original forward pass
        out_1 = netG(t)
        # Horizontal flip TTA
        out_2 = netG(t.flip(-1)).flip(-1)
        # Average
        out_norm = (out_1 + out_2) / 2.0

    # === Step 4: De-normalize ===
    result_norm = F.interpolate(out_norm, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    result_norm = result_norm[0, 0].cpu().numpy()
    synthetic = result_norm * img_std + img_mean

    # === Step 5: Composite with heart offset ===
    # Heart region: adaptive offset (contrast agent pools in heart blood)
    heart_bg = heart_mask * (1 - breast_mask)  # heart outside breast only
    heart_offset = np.clip(sl * 1.0, 0, 4.0) * heart_bg

    # Final: breast=synthetic, background=pre, heart=pre+offset
    result = breast_mask * synthetic + (1 - breast_mask) * sl + heart_offset

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
