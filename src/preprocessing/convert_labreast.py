"""
convert_labreast.py — Convert LA-Breast DCE-MRI Dataset to data_multislice_v2 format.

Reads paired d0 (pre-contrast) and d1-d5 (post-contrast) TIFF slices,
applies z-score normalization, and outputs MHA files compatible with
the MAMA-SYNTH training pipeline.

Each row in metadata CSV → one sample per post-contrast phase (d1-d5).
Output naming: LAB_{patient}_{ROI}_{row_idx}_p{phase}.mha

Usage:
    python convert_labreast.py \
        --data_dir "/path/to/LA-Breast DCE-MRI Dataset/breast_data" \
        --output_dir /path/to/data_multislice_v2/train \
        --global_stats src/preprocessing/training_pre_stats.json \
        --split train
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tiff(path: Path) -> np.ndarray:
    """Load TIFF and return float32 numpy array."""
    img = Image.open(path)
    return np.array(img).astype(np.float32)


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score normalise."""
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert LA-Breast to MHA format")
    parser.add_argument("--data_dir", required=True, help="Path to breast_data/")
    parser.add_argument("--output_dir", required=True, help="Output dir (e.g. data_multislice_v2/train)")
    parser.add_argument("--global_stats", required=True, help="JSON with mean/std for z-score")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.global_stats) as f:
        stats = json.load(f)
    norm_mean, norm_std = float(stats["mean"]), float(stats["std"])

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mha_input_dir = output_dir / "mha" / "input"
    mha_gt_dir = output_dir / "mha" / "ground_truth"
    mha_mask_dir = output_dir / "mha" / "mask"
    for d in (mha_input_dir, mha_gt_dir, mha_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Read metadata
    meta_csv = data_dir / "metadata" / f"{args.split}.csv"
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Split: {args.split}, {len(rows)} rows in metadata")

    # Phase columns in CSV (TIFF filenames)
    phase_cols = ["A2_d0", "A2_d1", "A2_d2", "A2_d3", "A2_d4", "A2_d5"]
    total = 0

    for idx, row in enumerate(rows):
        patient = row["patient"].replace("Breast_", "")  # Mri_101
        roi = row["ROI"]  # R1, R2, etc.

        # Load pre-contrast (d0)
        d0_fname = row["A2_d0"]
        d0_path = data_dir / "d0" / args.split / d0_fname
        if not d0_path.exists():
            continue

        pre_arr = load_tiff(d0_path)

        # For each post-contrast phase
        for phase_num in range(1, 6):
            gt_fname = row[f"A2_d{phase_num}"]
            gt_path = data_dir / f"d{phase_num}" / args.split / gt_fname
            if not gt_path.exists():
                continue

            gt_arr = load_tiff(gt_path)

            # Verify same shape
            if pre_arr.shape != gt_arr.shape:
                continue

            # Z-score normalize using the same stats as MAMA-SYNTH
            pre_norm = zscore(pre_arr, norm_mean, norm_std)
            gt_norm = zscore(gt_arr, norm_mean, norm_std)

            # Create empty mask (no tumor annotation available)
            mask = np.zeros_like(pre_norm, dtype=np.int16)

            # Output filename
            fname = f"LAB_{patient}_{roi}_{idx:04d}_p{phase_num}"

            # Skip if already exists (resume)
            if (mha_input_dir / f"{fname}.mha").exists():
                total += 1
                continue

            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(gt_norm), str(mha_gt_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(mha_mask_dir / f"{fname}.mha"))
            total += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"  Progress: {idx+1}/{len(rows)} rows, {total} files")

    logger.info(f"Done. {total} files written to {output_dir}")


if __name__ == "__main__":
    main()
