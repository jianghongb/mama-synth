"""
segment_breast_labreast.py — 2D breast segmentation for LA-Breast dataset.

Uses Dataset910_BreastSegNet (nnUNet 2D, 1ch T1) to generate breast masks
from LA-Breast pre-contrast TIFF slices.

Usage:
    python segment_breast_labreast.py \
        --input_dir /path/to/breast_data/d0/train \
        --output_dir /path/to/breast_masks/train \
        --model_dir /path/to/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetPlans__2d
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")

# Dataset910 labels: tissue(1), vessel(2), lesion(5), lymphnode(6), implant(9) = breast
BREAST_LABELS = {1, 2, 5, 6, 9}


def build_predictor(model_dir: str, device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device(device), verbose=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth",
    )
    logger.info(f"Model loaded on {device}")
    return predictor


def predict_mask(predictor, image_2d: np.ndarray) -> np.ndarray:
    """Run 2D nnUNet prediction, return binary breast mask."""
    arr = image_2d.astype(np.float32)
    # nnUNet 2D expects (C, 1, H, W) — single channel, single "slice"
    input_arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    props = {
        "sitk_stuff": {
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
        "spacing": [1.0, 1.0, 1.0],
    }
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)

    # Squeeze to 2D
    while pred.ndim > 2:
        pred = pred[0]

    return np.isin(pred, list(BREAST_LABELS)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="2D breast segmentation for LA-Breast")
    parser.add_argument("--input_dir", required=True, help="Directory with .tiff slices (e.g. d0/train)")
    parser.add_argument("--output_dir", required=True, help="Output directory for breast masks")
    parser.add_argument("--model_dir", required=True, help="Dataset910 nnUNet 2D model folder")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    predictor = build_predictor(args.model_dir, device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiffs = sorted(input_dir.glob("*.tiff")) + sorted(input_dir.glob("*.tif"))
    logger.info(f"Found {len(tiffs)} TIFF files")

    for tiff_path in tqdm(tiffs, desc="Segmenting"):
        out_path = output_dir / tiff_path.name
        if out_path.exists():
            continue

        img = np.array(Image.open(tiff_path)).astype(np.float32)
        mask = predict_mask(predictor, img)

        # Save as TIFF (same format)
        Image.fromarray(mask * 255).save(str(out_path))

    logger.info(f"Done. Masks saved to {output_dir}")


if __name__ == "__main__":
    main()
