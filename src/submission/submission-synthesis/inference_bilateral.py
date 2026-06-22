#!/usr/bin/env python3
"""MAMA-SYNTH — Bilateral split inference for unilateral-trained models.

Pipeline:
  1. Load bilateral pre-contrast slice
  2. Generate breast mask (nnUNet Dataset920/910 2D)
  3. Split into L/R unilateral crops via midline detection
  4. Resize each crop to 512×512, run Pix2PixHD (+refiner)
  5. Stitch L/R predictions back into bilateral canvas
  6. Apply breast mask: output = mask × synthetic + (1-mask) × pre

Grand Challenge I/O contract:
  Input:  /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha
  Output: /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha

Environment:
  MAMA_INPUT_DIR, MAMA_OUTPUT_DIR, MAMA_WEIGHTS_PATH, MAMA_BREAST_SEG_DIR
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))

from models.networks import GlobalGenerator
from stitch_bilateral import BilateralSplitter

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = "pre-contrast-dce-mri-slice-breast"
OUTPUT_SLUG = "synthetic-contrast-dce-mri-slice-breast"
WEIGHTS_PATH = os.environ.get("MAMA_WEIGHTS_PATH", "/opt/app/weights/latest_net_G.pth")
BREAST_SEG_DIR = os.environ.get("MAMA_BREAST_SEG_DIR", "/opt/app/weights/breast_seg")
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


def build_breast_predictor(device):
    """Build nnUNet 2D breast segmentation predictor."""
    os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=device,
        verbose=False, allow_tqdm=False
    )
    # Try Dataset920 first, fallback to Dataset910
    model_dir = None
    for candidate in [
        Path(BREAST_SEG_DIR) / "fold_0",
        Path(BREAST_SEG_DIR),
    ]:
        if (candidate / "checkpoint_final.pth").exists():
            model_dir = str(candidate.parent) if candidate.name == "fold_0" else str(candidate)
            break
    if model_dir is None:
        raise FileNotFoundError(f"No breast seg model found in {BREAST_SEG_DIR}")

    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    return predictor


def predict_breast_mask(predictor, arr_2d: np.ndarray) -> np.ndarray:
    """Run 2D breast segmentation, return binary mask."""
    from scipy import ndimage as ndi
    input_arr = arr_2d.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    props = {'sitk_stuff': {'spacing': (1., 1., 1.), 'origin': (0., 0., 0.),
             'direction': (1., 0., 0., 0., 1., 0., 0., 0., 1.)}, 'spacing': [1., 1., 1.]}
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    mask = (pred > 0).astype(np.uint8)
    # Keep components >= 10% of largest
    labeled, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labeled, range(1, n + 1))
        thresh = sizes.max() * 0.1
        mask = np.zeros_like(mask, dtype=np.uint8)
        for i, s in enumerate(sizes):
            if s >= thresh:
                mask[labeled == (i + 1)] = 1
    return mask.astype(np.float32)


@torch.no_grad()
def synthesize_crop(netG, crop: np.ndarray, device) -> np.ndarray:
    """Run Pix2PixHD on a single unilateral crop resized to MODEL_SIZE."""
    t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
    t = torch.nn.functional.interpolate(t, size=(MODEL_SIZE, MODEL_SIZE),
                                        mode='bilinear', align_corners=False)
    out = netG(t)
    return out[0, 0].cpu().numpy()


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
    breast_predictor = build_breast_predictor(device)

    # Splitter matches training preprocessing
    splitter = BilateralSplitter(target_size=MODEL_SIZE, pad_ratio=0.03)

    input_file = find_input_image()
    print(f"Input: {input_file}")

    img = sitk.ReadImage(str(input_file))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    # Step 1: Breast mask
    breast_mask = predict_breast_mask(breast_predictor, sl)

    # Step 2: Split bilateral → L/R
    left, right, meta = splitter.split(sl, breast_mask)

    # Step 3: Synthesize each side
    pred_left = synthesize_crop(netG, left, device) if left is not None else None
    pred_right = synthesize_crop(netG, right, device) if right is not None else None

    # Step 4: Stitch back
    result = splitter.stitch(pred_left, pred_right, meta)

    # Step 5: Apply breast mask (preserve pre-contrast outside breast)
    result = breast_mask * result + (1 - breast_mask) * sl

    # Restore original ndim
    if arr.ndim == 3:
        result = result[np.newaxis, ...]
    result = result.astype(np.float32)

    # Write output
    out_img = sitk.GetImageFromArray(result)
    out_img.CopyInformation(img)

    out_dir = OUTPUT_PATH / "images" / OUTPUT_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output.mha"
    sitk.WriteImage(out_img, str(out_file))
    print(f"Output: {out_file}  shape={result.shape}  mode=bilateral_split")


if __name__ == "__main__":
    main()
