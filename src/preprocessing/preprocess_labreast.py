"""
preprocess_labreast.py — LA-Breast 2D preprocessing with breast masking.

LA-Breast is already 2D (448×448 TIFF, 16-bit). This script:
1. Loads d0 (pre-contrast) and peak post-contrast phase (d1-d5)
2. Applies breast mask (pre-generated)
3. Generates tumour mask from ROI coordinates in metadata CSV
4. Z-score normalises
5. Saves as .mha in the standard format

Input structure:
  breast_data/d0/train/*.tiff   (pre-contrast)
  breast_data/d1-d5/train/*.tiff (post-contrast phases)
  breast_data/metadata/train.csv (ROI coordinates)
  breast_masks/train/*.tiff      (binary breast masks)

Usage:
    python preprocess_labreast.py \
        --data_dir "/path/to/breast_data" \
        --mask_dir "/path/to/breast_masks/train" \
        --output_dir "/path/to/output" \
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def make_ellipse_mask(shape: tuple, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Create binary ellipse mask. cx/cy are center coords, rx/ry are radii."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
    return mask.astype(np.float32)


def load_roi_metadata(csv_path: Path) -> tuple:
    """Load ROI info and phase filename mapping from metadata CSV.

    Returns:
        roi_map: {d0_filename: [(cx, cy, rx, ry), ...]}
        phase_map: {d0_filename: {1: d1_filename, 2: d2_filename, ...}}
    """
    roi_map = {}
    phase_map = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d0_name = row.get("A2_d0", "").strip()
            if not d0_name:
                continue

            # ROI coordinates
            cx = float(row["Centro_x_d0"])
            cy = float(row["Centro_y_d0"])
            rx = float(row["Distancia_x_d0"]) / 2.0
            ry = float(row["Distancia_y_d0"]) / 2.0

            if d0_name not in roi_map:
                roi_map[d0_name] = []
            roi_map[d0_name].append((cx, cy, rx, ry))

            # Phase filename mapping (d0 -> d1..d5)
            if d0_name not in phase_map:
                phase_map[d0_name] = {}
                for phase in range(1, 6):
                    pname = row.get(f"A2_d{phase}", "").strip()
                    if pname:
                        phase_map[d0_name][phase] = pname

    return roi_map, phase_map


def main():
    parser = argparse.ArgumentParser(description="LA-Breast 2D preprocessing")
    parser.add_argument("--data_dir", required=True, help="breast_data root (contains d0/, d1/, ...)")
    parser.add_argument("--mask_dir", required=True, help="Breast mask TIFF directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--peak_phase", type=int, default=None,
                        help="Fixed post-contrast phase (1-5). If not set, selects per-slice max.")
    args = parser.parse_args()

    with open(args.global_stats) as f:
        stats = json.load(f)
    norm_mean, norm_std = float(stats["mean"]), float(stats["std"])

    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    mha_input_dir = output_dir / "mha" / "input"
    mha_gt_dir = output_dir / "mha" / "ground_truth"
    mha_mask_dir = output_dir / "mha" / "mask"
    mha_breast_dir = output_dir / "mha" / "breast_mask"
    for d in (mha_input_dir, mha_gt_dir, mha_mask_dir, mha_breast_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load ROI metadata and phase mapping
    metadata_csv = data_dir / "metadata" / f"{args.split}.csv"
    if metadata_csv.exists():
        roi_map, phase_map = load_roi_metadata(metadata_csv)
    else:
        roi_map, phase_map = {}, {}
    logger.info(f"ROI metadata: {len(roi_map)} slices with annotations, {len(phase_map)} with phase mapping")

    # List all d0 files for this split
    d0_dir = data_dir / "d0" / args.split
    tiff_files = sorted(d0_dir.glob("*.tiff"))
    logger.info(f"Found {len(tiff_files)} slices in {d0_dir}")

    # Post-contrast phase directories
    post_dirs = {}
    for phase in range(1, 6):
        pd = data_dir / f"d{phase}" / args.split
        if pd.exists():
            post_dirs[phase] = pd

    count = 0
    count_with_tumour = 0
    for tiff_path in tqdm(tiff_files, desc="Processing"):
        fname = tiff_path.stem  # e.g. Mri_101_R1_IM-0002-0085
        pid = f"LABREAST_{fname}"
        tiff_name = tiff_path.name

        # Load pre-contrast
        pre = np.array(Image.open(tiff_path)).astype(np.float32)
        h, w = pre.shape

        # Load breast mask
        mask_path = mask_dir / tiff_name
        if not mask_path.exists():
            continue
        breast_mask = (np.array(Image.open(mask_path)) > 127).astype(np.float32)

        # Generate tumour mask from ROI metadata
        tumour_mask = np.zeros((h, w), dtype=np.float32)
        if tiff_name in roi_map:
            for cx, cy, rx, ry in roi_map[tiff_name]:
                ellipse = make_ellipse_mask((h, w), cx, cy, rx, ry)
                tumour_mask = np.maximum(tumour_mask, ellipse)
            count_with_tumour += 1

        # Get phase filename mapping for this slice
        slice_phase_map = phase_map.get(tiff_name, {})

        # Select peak post-contrast phase
        if args.peak_phase is not None and args.peak_phase in slice_phase_map:
            peak = args.peak_phase
        else:
            # Find phase with highest mean intensity in breast region
            best_phase, best_val = None, -1.0
            for phase, pname in slice_phase_map.items():
                pf = post_dirs.get(phase)
                if pf is None:
                    continue
                pp = pf / pname
                if pp.exists():
                    post = np.array(Image.open(pp)).astype(np.float32)
                    val = float(np.mean(post[breast_mask > 0])) if breast_mask.sum() > 0 else 0
                    if val > best_val:
                        best_val = val
                        best_phase = phase
            peak = best_phase

        if peak is None:
            continue

        # Load peak post-contrast using mapped filename
        peak_fname = slice_phase_map.get(peak)
        if peak_fname is None:
            continue
        peak_path = post_dirs[peak] / peak_fname
        if not peak_path.exists():
            continue
        post = np.array(Image.open(peak_path)).astype(np.float32)

        # Apply breast mask
        pre_masked = pre * breast_mask
        post_masked = post * breast_mask

        # Z-score normalise
        pre_norm = zscore(pre_masked, norm_mean, norm_std)
        post_norm = zscore(post_masked, norm_mean, norm_std)

        # Save as MHA
        sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{pid}.mha"))
        sitk.WriteImage(sitk.GetImageFromArray(post_norm), str(mha_gt_dir / f"{pid}.mha"))
        sitk.WriteImage(sitk.GetImageFromArray(tumour_mask.astype(np.int16)), str(mha_mask_dir / f"{pid}.mha"))
        sitk.WriteImage(sitk.GetImageFromArray(breast_mask.astype(np.int16)), str(mha_breast_dir / f"{pid}.mha"))
        count += 1

    logger.info(f"Done. {count} slices processed ({count_with_tumour} with tumour ROI) → {output_dir}")


if __name__ == "__main__":
    main()
