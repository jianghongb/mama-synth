"""
preprocess_multislice.py — Extract multiple slices around peak tumor slice.

Takes the peak tumor slice ± N adjacent slices, generating 2N+1 samples per patient.
Only includes slices where tumor mask area > min_area threshold.

Usage:
    python preprocess_multislice.py \
        --image_dir /path/to/images \
        --seg_dir /path/to/segmentations/automatic \
        --output_dir /path/to/output \
        --global_stats src/preprocessing/training_pre_stats.json \
        --extra_slices 2 \
        --min_mask_area 50 \
        --skip_ambiguous_shapes
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser(description="Multi-slice preprocessing")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--extra_slices", type=int, default=2, help="Number of extra slices on each side of peak")
    parser.add_argument("--min_mask_area", type=int, default=50, help="Min tumor mask pixels to include slice")
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
    parser.add_argument("--exclude_list", default=None)
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
    logger.info(f"Found {len(patients)} patients, extra_slices={args.extra_slices}")

    total_slices = 0
    for patient_dir in patients:
        patient_id = patient_dir.name
        try:
            # Load phases
            phase_files = sorted(patient_dir.glob(f"{patient_id}_*.nii.gz"))
            if len(phase_files) < 2:
                continue

            phases = {}
            for pf in phase_files:
                idx = int(pf.stem.replace(".nii", "").rsplit("_", 1)[1])
                phases[idx] = load_nifti(pf)

            # Load tumor seg
            seg_file = seg_dir / f"{patient_id}.nii.gz"
            if not seg_file.exists():
                candidates = list(seg_dir.glob(f"{patient_id}*"))
                seg_file = candidates[0] if candidates else None
            if seg_file is None or not seg_file.exists():
                continue
            tumour_seg = load_nifti(seg_file)

            # Determine axis
            try:
                axis = determine_slice_axis(tumour_seg.shape)
            except ValueError:
                if args.skip_ambiguous_shapes:
                    continue
                raise

            # Find peak slice
            peak_idx = find_largest_slice(tumour_seg, axis)
            n_slices = tumour_seg.shape[axis]

            # Find peak phase
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

            # Extract slices: peak ± extra_slices
            offsets = range(-args.extra_slices, args.extra_slices + 1)
            for offset in offsets:
                slice_idx = peak_idx + offset
                if slice_idx < 0 or slice_idx >= n_slices:
                    continue

                # Extract 2D
                mask_2d = np.take(tumour_seg, slice_idx, axis=axis)

                # Check min area (skip slices with tiny/no tumor)
                if offset != 0 and mask_2d.sum() < args.min_mask_area:
                    continue

                pre_2d = np.take(phases[pre_phase], slice_idx, axis=axis)
                peak_2d = np.take(phases[peak_phase], slice_idx, axis=axis)

                # Z-score normalize
                pre_norm = zscore(pre_2d, norm_mean, norm_std)
                peak_norm = zscore(peak_2d, norm_mean, norm_std)
                mask_out = np.rint(mask_2d).astype(np.int16)

                # Rotate 90° CCW
                pre_norm = np.rot90(pre_norm, k=1)
                peak_norm = np.rot90(peak_norm, k=1)
                mask_out = np.rot90(mask_out, k=1)

                # Naming: patient_s+0, patient_s-1, patient_s+2, etc.
                if offset == 0:
                    fname = patient_id
                else:
                    fname = f"{patient_id}_s{offset:+d}"

                sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{fname}.mha"))
                sitk.WriteImage(sitk.GetImageFromArray(peak_norm), str(mha_gt_dir / f"{fname}.mha"))
                sitk.WriteImage(sitk.GetImageFromArray(mask_out), str(mha_mask_dir / f"{fname}.mha"))
                total_slices += 1

            logger.info(f"  ✓ {patient_id}: axis={axis}, peak={peak_idx}, slices extracted")

        except Exception as e:
            logger.error(f"ERROR {patient_id}: {e}")
            continue

    logger.info(f"\nDone. {total_slices} total slices from {len(patients)} patients → {output_dir}")


if __name__ == "__main__":
    main()
