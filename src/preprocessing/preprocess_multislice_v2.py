"""
preprocess_multislice_v2.py — Extract peak-performance slice from EACH phase.

For each patient:
1. Determine the slice axis (through-plane).
2. For each DCE phase, find the 2D slice where the tumour ROI has the
   highest mean intensity within that phase.
3. Output: pre-contrast slice + corresponding peak slice (from the overall
   peak phase) at the SAME slice position as the per-phase maximum.

Optimised with vectorised slice search (no Python loops over slices).

Usage:
    python preprocess_multislice_v2.py \
        --image_dir /path/to/images \
        --seg_dir /path/to/segmentations/automatic \
        --output_dir /path/to/output \
        --global_stats src/preprocessing/training_pre_stats.json \
        --skip_ambiguous_shapes \
        --workers 8
"""
import argparse
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return as float32 numpy array."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def determine_slice_axis(shape: Tuple[int, ...]) -> int:
    """Determine through-plane axis from shape heuristic."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D, got {shape}")
    size_to_axes: Dict[int, list] = {}
    for ax, s in enumerate(shape):
        size_to_axes.setdefault(s, []).append(ax)
    n = len(size_to_axes)
    if n == 1:
        return 2
    if n == 3:
        raise ValueError(f"Ambiguous FOV: {shape}")
    return next(axes[0] for axes in size_to_axes.values() if len(axes) == 1)


def find_peak_slice_vectorised(volume: np.ndarray, seg: np.ndarray, axis: int) -> Tuple[int, float]:
    """Find slice with highest mean tumour intensity — fully vectorised.

    Computes sum and count of tumour voxels per slice using array ops,
    then divides to get mean intensity per slice.
    """
    # Move the slice axis to position 0
    vol = np.moveaxis(volume, axis, 0)  # (n_slices, H, W)
    mask = np.moveaxis(seg, axis, 0) > 0  # (n_slices, H, W) boolean

    # Sum of intensities within mask per slice
    masked_vol = np.where(mask, vol, 0.0)
    slice_sums = masked_vol.reshape(vol.shape[0], -1).sum(axis=1)

    # Count of mask voxels per slice
    slice_counts = mask.reshape(mask.shape[0], -1).sum(axis=1)

    # Mean intensity (avoid div by zero)
    valid = slice_counts > 0
    if not valid.any():
        return 0, 0.0

    means = np.full(vol.shape[0], -np.inf)
    means[valid] = slice_sums[valid] / slice_counts[valid]

    best_idx = int(np.argmax(means))
    return best_idx, float(means[best_idx])


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score normalise."""
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def process_patient(
    patient_dir: Path,
    seg_dir: Path,
    mha_input_dir: Path,
    mha_gt_dir: Path,
    mha_mask_dir: Path,
    norm_mean: float,
    norm_std: float,
    min_mask_area: int,
    skip_ambiguous: bool,
) -> Tuple[str, int]:
    """Process a single patient. Returns (patient_id, num_slices_extracted)."""
    patient_id = patient_dir.name
    try:
        # Load all phases
        phase_files = sorted(patient_dir.glob(f"{patient_id}_*.nii.gz"))
        if len(phase_files) < 2:
            return patient_id, 0

        phases: Dict[int, np.ndarray] = {}
        for pf in phase_files:
            idx = int(pf.stem.replace(".nii", "").rsplit("_", 1)[1])
            phases[idx] = load_nifti(pf)

        # Load tumour segmentation
        seg_file = seg_dir / f"{patient_id}.nii.gz"
        if not seg_file.exists():
            candidates = list(seg_dir.glob(f"{patient_id}*"))
            seg_file = candidates[0] if candidates else None
        if seg_file is None or not seg_file.exists():
            return patient_id, 0
        tumour_seg = load_nifti(seg_file)

        # Determine slice axis
        try:
            axis = determine_slice_axis(tumour_seg.shape)
        except ValueError:
            if skip_ambiguous:
                return patient_id, 0
            raise

        # Identify pre-contrast and post-contrast phases
        pre_phase = min(phases.keys())
        post_phases = sorted(k for k in phases.keys() if k != pre_phase)

        if not post_phases:
            return patient_id, 0

        # Find global peak phase (vectorised)
        global_peak_phase = pre_phase
        global_peak_val = -np.inf
        for k, vol in phases.items():
            voxels = vol[tumour_seg > 0]
            if len(voxels) > 0:
                m = float(np.mean(voxels))
                if m > global_peak_val:
                    global_peak_val = m
                    global_peak_phase = k

        # For each post-contrast phase, find its best slice (vectorised)
        slices_extracted = set()
        n_extracted = 0
        for phase_num in post_phases:
            peak_slice_idx, _ = find_peak_slice_vectorised(
                phases[phase_num], tumour_seg, axis
            )

            # Skip duplicate slice positions
            if peak_slice_idx in slices_extracted:
                continue

            # Check mask area
            mask_2d = np.take(tumour_seg, peak_slice_idx, axis=axis)
            if mask_2d.sum() < min_mask_area:
                continue

            slices_extracted.add(peak_slice_idx)

            # Extract slices
            pre_2d = np.take(phases[pre_phase], peak_slice_idx, axis=axis)
            gt_2d = np.take(phases[global_peak_phase], peak_slice_idx, axis=axis)
            mask_out = np.rint(mask_2d).astype(np.int16)

            # Z-score normalize
            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            gt_norm = zscore(gt_2d, norm_mean, norm_std)

            # Rotate 90° CCW
            pre_norm = np.rot90(pre_norm, k=1)
            gt_norm = np.rot90(gt_norm, k=1)
            mask_out = np.rot90(mask_out, k=1)

            fname = f"{patient_id}_p{phase_num}"
            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(gt_norm), str(mha_gt_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(mask_out), str(mha_mask_dir / f"{fname}.mha"))
            n_extracted += 1

        return patient_id, n_extracted

    except Exception as e:
        return patient_id, -1


def main():
    parser = argparse.ArgumentParser(description="Multi-slice v2: peak slice from each phase (parallel)")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--min_mask_area", type=int, default=50,
                        help="Min tumor mask pixels to include a slice")
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
    parser.add_argument("--exclude_list", default=None,
                        help="Text file with patient IDs to exclude (one per line)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    with open(args.global_stats) as f:
        stats = json.load(f)
    norm_mean, norm_std = float(stats["mean"]), float(stats["std"])

    output_dir = Path(args.output_dir)
    mha_input_dir = output_dir / "mha" / "input"
    mha_gt_dir = output_dir / "mha" / "ground_truth"
    mha_mask_dir = output_dir / "mha" / "mask"
    for d in (mha_input_dir, mha_gt_dir, mha_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Exclusion list
    exclude_ids = set()
    if args.exclude_list:
        with open(args.exclude_list) as f:
            exclude_ids = {line.strip() for line in f if line.strip()}

    image_dir = Path(args.image_dir)
    seg_dir = Path(args.seg_dir)

    patients = sorted(d for d in image_dir.iterdir() if d.is_dir())
    if exclude_ids:
        patients = [p for p in patients if p.name not in exclude_ids]

    n_workers = args.workers or mp.cpu_count()
    logger.info(f"Found {len(patients)} patients, using {n_workers} workers")

    worker_fn = partial(
        process_patient,
        seg_dir=seg_dir,
        mha_input_dir=mha_input_dir,
        mha_gt_dir=mha_gt_dir,
        mha_mask_dir=mha_mask_dir,
        norm_mean=norm_mean,
        norm_std=norm_std,
        min_mask_area=args.min_mask_area,
        skip_ambiguous=args.skip_ambiguous_shapes,
    )

    total_slices = 0
    errors = 0
    with mp.Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(worker_fn, patients)
        for i, (pid, n) in enumerate(results, 1):
            if n < 0:
                errors += 1
            else:
                total_slices += n
            if i % 50 == 0 or i == len(patients):
                logger.info(f"  Progress: {i}/{len(patients)} patients, {total_slices} slices, {errors} errors")

    logger.info(f"\nDone. {total_slices} total slices from {len(patients)} patients ({errors} errors) → {output_dir}")


if __name__ == "__main__":
    main()
