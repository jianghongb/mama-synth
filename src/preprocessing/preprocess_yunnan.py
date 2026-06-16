"""
preprocess_yunnan.py — Yunnan dataset 3D→2D preprocessing with built-in breast mask.

Yunnan data structure (each zip):
  <id>/P0.nii.gz ... P5.nii.gz, Breast_mask.nii.gz, GT.nii.gz

Output: same format as mask_and_preprocess.py
  mha/{input, ground_truth, mask, breast_mask}/<patient_id>.mha

Usage:
    python preprocess_yunnan.py \
        --data_dir /path/to/8068383 \
        --output_dir /path/to/output \
        --global_stats src/preprocessing/training_pre_stats.json
"""
import argparse
import json
import logging
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def determine_slice_axis(shape: Tuple[int, ...]) -> int:
    if len(shape) != 3:
        raise ValueError(f"Expected 3D, got {shape}")
    size_to_axes = {}
    for ax, s in enumerate(shape):
        size_to_axes.setdefault(s, []).append(ax)
    n = len(size_to_axes)
    if n == 1:
        return 2
    if n == 3:
        raise ValueError(f"Ambiguous FOV: {shape}")
    return next(axes[0] for axes in size_to_axes.values() if len(axes) == 1)


def find_largest_slice(seg: np.ndarray, axis: int) -> int:
    others = tuple(i for i in range(seg.ndim) if i != axis)
    return int(np.argmax(np.sum(seg > 0, axis=others)))


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def process_case(case_dir: Path, patient_id: str, norm_mean: float, norm_std: float,
                 mha_input_dir: Path, mha_gt_dir: Path, mha_mask_dir: Path, mha_breast_dir: Path) -> dict:
    """Process one Yunnan case."""
    # Load phases
    phase_files = sorted(case_dir.glob("P*.nii.gz"))
    if len(phase_files) < 2:
        logger.warning(f"SKIP {patient_id}: <2 phases")
        return None

    phases = {}
    for pf in phase_files:
        idx = int(pf.stem.replace("P", ""))
        phases[idx] = load_nifti(pf)

    # Load breast mask and tumour GT
    breast_mask_file = case_dir / "Breast_mask.nii.gz"
    gt_file = case_dir / "GT.nii.gz"
    if not breast_mask_file.exists() or not gt_file.exists():
        logger.warning(f"SKIP {patient_id}: missing Breast_mask or GT")
        return None

    breast_mask = (load_nifti(breast_mask_file) > 0).astype(np.float32)
    tumour_seg = load_nifti(gt_file)

    # Apply breast mask to all phases
    masked_phases = {k: v * breast_mask for k, v in phases.items()}

    # Determine slice axis
    try:
        axis = determine_slice_axis(tumour_seg.shape)
    except ValueError as e:
        logger.warning(f"SKIP {patient_id}: {e}")
        return None

    # Find peak phase (highest mean tumour intensity)
    pre_phase = 0
    peak_phase = 0
    peak_val = -np.inf
    for k, vol in masked_phases.items():
        tumour_voxels = vol[tumour_seg > 0]
        if len(tumour_voxels) > 0:
            m = float(np.mean(tumour_voxels))
            if m > peak_val:
                peak_val = m
                peak_phase = k

    # Select slice with largest tumour area
    slice_idx = find_largest_slice(tumour_seg, axis)

    # Extract 2D slices
    pre_2d = np.take(masked_phases[pre_phase], slice_idx, axis=axis)
    peak_2d = np.take(masked_phases[peak_phase], slice_idx, axis=axis)
    mask_2d = np.take(tumour_seg, slice_idx, axis=axis).astype(np.int16)
    breast_2d = np.take(breast_mask, slice_idx, axis=axis).astype(np.int16)

    # Z-score normalise
    pre_norm = zscore(pre_2d, norm_mean, norm_std)
    peak_norm = zscore(peak_2d, norm_mean, norm_std)

    # Rotate 90° CCW (match convention)
    pre_norm = np.rot90(pre_norm, k=1)
    peak_norm = np.rot90(peak_norm, k=1)
    mask_2d = np.rot90(mask_2d, k=1)
    breast_2d = np.rot90(breast_2d, k=1)

    # Save
    pid = f"YUNNAN_{patient_id}"
    for arr, dirp, is_label in [
        (pre_norm, mha_input_dir, False),
        (peak_norm, mha_gt_dir, False),
        (mask_2d, mha_mask_dir, True),
        (breast_2d, mha_breast_dir, True),
    ]:
        out = np.rint(arr).astype(np.int16) if is_label else arr.astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(out), str(dirp / f"{pid}.mha"))

    logger.info(f"  ✓ {pid}: axis={axis}, slice={slice_idx}, peak=P{peak_phase}")
    return {"patient_id": pid, "pre_phase": pre_phase, "peak_phase": peak_phase,
            "slice_idx": slice_idx, "slice_axis": axis}


def main():
    parser = argparse.ArgumentParser(description="Yunnan dataset preprocessing")
    parser.add_argument("--data_dir", required=True, help="Directory with .zip files (8068383)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    args = parser.parse_args()

    with open(args.global_stats) as f:
        stats = json.load(f)
    norm_mean, norm_std = float(stats["mean"]), float(stats["std"])

    output_dir = Path(args.output_dir)
    mha_input_dir = output_dir / "mha" / "input"
    mha_gt_dir = output_dir / "mha" / "ground_truth"
    mha_mask_dir = output_dir / "mha" / "mask"
    mha_breast_dir = output_dir / "mha" / "breast_mask"
    for d in (mha_input_dir, mha_gt_dir, mha_mask_dir, mha_breast_dir):
        d.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    zips = sorted(data_dir.glob("*.zip"))
    logger.info(f"Found {len(zips)} zip files")

    results = []
    for zf in zips:
        patient_id = zf.stem  # e.g. "1", "23", "100"
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zf) as z:
                z.extractall(tmp)
            # Find the case directory inside
            case_dir = Path(tmp) / patient_id
            if not case_dir.exists():
                # Try finding any directory
                dirs = [d for d in Path(tmp).iterdir() if d.is_dir()]
                case_dir = dirs[0] if dirs else Path(tmp)

            result = process_case(case_dir, patient_id, norm_mean, norm_std,
                                  mha_input_dir, mha_gt_dir, mha_mask_dir, mha_breast_dir)
            if result:
                results.append(result)

    # Save report
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        report_path = output_dir / "report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"\nDone. {len(results)} cases → {output_dir}")


if __name__ == "__main__":
    main()
