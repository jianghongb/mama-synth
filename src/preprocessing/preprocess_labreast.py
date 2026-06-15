"""
Preprocess LA-Breast DCE-MRI Dataset for MAMA-SYNTH training.

Converts TIFF DCE pairs (d0=pre, d1=first post-contrast) into
z-score normalized MHA files matching MAMA-SYNTH format.

Usage:
    python preprocess_labreast.py \
        --src "/Users/ehogjig/Downloads/LA-Breast DCE-MRI Dataset/breast_data" \
        --dst /Users/ehogjig/git/kth/data_split_v2/train_labreast/mha \
        --split train
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

# MAMA-SYNTH z-score stats
MEAN = 104.86
STD = 215.86


def load_tiff(path: str) -> np.ndarray:
    """Load TIFF as float32."""
    return np.array(Image.open(path)).astype(np.float32)


def zscore(arr: np.ndarray) -> np.ndarray:
    """Apply MAMA-SYNTH z-score normalization.
    
    LA-Breast raw range is [0, ~260], MAMA-SYNTH raw range is [0, ~500+].
    We first rescale LA-Breast to match MAMA-SYNTH intensity distribution,
    then apply z-score with MAMA-SYNTH stats.
    """
    # Rescale: LA-Breast p99 ~200 → MAMA-SYNTH p99 ~500
    # Scale factor estimated from ranges
    SCALE_FACTOR = 1.5
    arr_scaled = arr * SCALE_FACTOR
    return (arr_scaled - MEAN) / STD


def make_ellipse_mask(shape, cx, cy, rx, ry):
    """Create binary ellipse mask from center and radii."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    return mask.astype(np.float32)


def save_mha(arr: np.ndarray, path: str):
    """Save 2D array as MHA with shape (1, H, W)."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    sitk.WriteImage(img, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="LA-Breast breast_data dir")
    parser.add_argument("--dst", required=True, help="Output dir (mha structure)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "input").mkdir(parents=True, exist_ok=True)
    (dst / "ground_truth").mkdir(parents=True, exist_ok=True)
    (dst / "mask").mkdir(parents=True, exist_ok=True)

    # Read metadata
    with open(src / "metadata" / f"{args.split}.csv") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} slices from {args.split} split...")

    count = 0
    for i, r in enumerate(rows):
        d0_file = r["A2_d0"]

        d0_path = src / "d0" / args.split / d0_file

        if not d0_path.exists():
            continue

        # Load pre-contrast
        pre = load_tiff(str(d0_path))

        # Ground truth = pixel-wise max across d1-d5 (peak enhancement)
        peak = np.full_like(pre, -np.inf)
        found_post = False
        for phase_idx in range(1, 6):
            col = f"A2_d{phase_idx}"
            if col not in r or not r[col]:
                continue
            di_path = src / f"d{phase_idx}" / args.split / r[col]
            if not di_path.exists():
                continue
            di = load_tiff(str(di_path))
            peak = np.maximum(peak, di)
            found_post = True

        if not found_post:
            continue

        # Z-score normalize
        pre_z = zscore(pre)
        post_z = zscore(peak)

        # Generate ellipse mask from ROI coordinates
        # Distancia = diameter (divide by 2 for radius)
        cx = float(r["Centro_x_d0"])
        cy = float(r["Centro_y_d0"])
        rx = float(r["Distancia_x_d0"]) / 2.0
        ry = float(r["Distancia_y_d0"]) / 2.0

        mask = make_ellipse_mask(pre.shape, cx, cy, max(rx, 1.0), max(ry, 1.0))

        # Output filename: LABREAST_{patient}_{ROI}_{slice_idx}
        patient_id = r["patient"].replace("Breast_Mri_", "")
        roi = r["ROI"]
        # Use d0 filename stem as slice identifier
        slice_id = Path(d0_file).stem.split("_")[-1]
        out_name = f"LABREAST_{patient_id}_{roi}_{slice_id}.mha"

        save_mha(pre_z, str(dst / "input" / out_name))
        save_mha(post_z, str(dst / "ground_truth" / out_name))
        save_mha(mask, str(dst / "mask" / out_name))
        count += 1

        if (count) % 100 == 0:
            print(f"  {count}/{len(rows)}")

    print(f"Done. Saved {count} cases to {dst}")


if __name__ == "__main__":
    main()
