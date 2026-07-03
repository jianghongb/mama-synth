"""
preprocess_multislice_v3.py — Peak slice ± N neighbors, global peak phase as GT.

For each patient:
1. Find global peak phase (highest mean tumor intensity across all voxels).
2. Find the slice with largest tumor area (peak slice).
3. Extract peak ± extra_slices, with GT always from global peak phase.
4. Z-score normalize and save as MHA.

This ensures GT consistency (all samples use peak enhancement) while maximizing
data volume through neighbor slices.

Parallelized with multiprocessing. Supports resume (skip existing files).

Usage:
    python preprocess_multislice_v3.py \
        --image_dir /path/to/images \
        --seg_dir /path/to/segmentations/automatic \
        --output_dir /path/to/data_multislice_v3 \
        --global_stats src/preprocessing/training_pre_stats.json \
        --extra_slices 2 \
        --min_mask_area 50 \
        --skip_ambiguous_shapes \
        --exclude_list src/preprocessing/motion_cases.txt \
        --workers 32
"""
import argparse
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

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


def find_largest_slice(seg: np.ndarray, axis: int) -> int:
    """Find slice with largest tumor area (vectorized)."""
    others = tuple(i for i in range(seg.ndim) if i != axis)
    return int(np.argmax(np.sum(seg > 0, axis=others)))


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
    extra_slices: int,
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

        # Find peak slice (largest tumor area)
        peak_idx = find_largest_slice(tumour_seg, axis)
        n_slices = tumour_seg.shape[axis]

        # Find global peak phase
        pre_phase = min(phases.keys())
        peak_phase = pre_phase
        peak_val = -np.inf
        for k, vol in phases.items():
            voxels = vol[tumour_seg > 0]
            if len(voxels) > 0:
                m = float(np.mean(voxels))
                if m > peak_val:
                    peak_val = m
                    peak_phase = k

        # Extract peak ± extra_slices
        n_extracted = 0
        for offset in range(-extra_slices, extra_slices + 1):
            slice_idx = peak_idx + offset
            if slice_idx < 0 or slice_idx >= n_slices:
                continue

            # Check mask area
            mask_2d = np.take(tumour_seg, slice_idx, axis=axis)
            if offset != 0 and mask_2d.sum() < min_mask_area:
                continue

            # Naming
            if offset == 0:
                fname = patient_id
            else:
                fname = f"{patient_id}_s{offset:+d}"

            # Skip if already exists (resume)
            if (mha_input_dir / f"{fname}.mha").exists():
                n_extracted += 1
                continue

            # Extract pre-contrast and GT (always global peak phase)
            pre_2d = np.take(phases[pre_phase], slice_idx, axis=axis)
            gt_2d = np.take(phases[peak_phase], slice_idx, axis=axis)
            mask_out = np.rint(mask_2d).astype(np.int16)

            # Z-score normalize
            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            gt_norm = zscore(gt_2d, norm_mean, norm_std)

            # Rotate 90° CCW
            pre_norm = np.rot90(pre_norm, k=1)
            gt_norm = np.rot90(gt_norm, k=1)
            mask_out = np.rot90(mask_out, k=1)

            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(gt_norm), str(mha_gt_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(mask_out), str(mha_mask_dir / f"{fname}.mha"))
            n_extracted += 1

        return patient_id, n_extracted

    except Exception as e:
        return patient_id, -1


def main():
    parser = argparse.ArgumentParser(description="Multi-slice v3: peak ± N, global peak phase GT (parallel)")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--extra_slices", type=int, default=2, help="± slices around peak (default: 2)")
    parser.add_argument("--min_mask_area", type=int, default=50)
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
    parser.add_argument("--exclude_list", default=None)
    parser.add_argument("--workers", type=int, default=None)
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
    logger.info(f"Found {len(patients)} patients, extra_slices={args.extra_slices}, workers={n_workers}")

    worker_fn = partial(
        process_patient,
        seg_dir=seg_dir,
        mha_input_dir=mha_input_dir,
        mha_gt_dir=mha_gt_dir,
        mha_mask_dir=mha_mask_dir,
        norm_mean=norm_mean,
        norm_std=norm_std,
        extra_slices=args.extra_slices,
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
