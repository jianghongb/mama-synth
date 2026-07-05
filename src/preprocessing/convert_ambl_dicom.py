"""
convert_ambl_dicom.py — Convert Advanced-MRI-Breast-Lesions DICOM to data_multislice_v3 format.

For each patient:
1. Load pre-contrast (MASK series) and multi-phase DCE (5 temporal phases).
2. Find global peak phase (highest mean intensity in tumor ROI or whole breast).
3. Find peak slice (largest tumor area or highest intensity).
4. Extract peak ± 2 slices, GT = global peak phase.
5. Z-score normalize and save as MHA.

Usage:
    python convert_ambl_dicom.py \
        --dicom_dir /path/to/Advanced-MRI-Breast-Lesions \
        --output_dir /path/to/data_multislice_v3/train \
        --global_stats src/preprocessing/training_pre_stats.json \
        --extra_slices 2 \
        --workers 16
"""
import argparse
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dicom_series(series_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load a DICOM series and return as 3D numpy array + metadata."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None, None

    # Read first file for metadata
    ds0 = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    rows, cols = int(ds0.Rows), int(ds0.Columns)

    # Read all slices, sort by slice location or instance number
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        loc = float(getattr(ds, 'SliceLocation', 0) or getattr(ds, 'InstanceNumber', 0))
        slices.append((loc, ds.pixel_array.astype(np.float32)))

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)  # (Z, H, W)

    meta = {
        'pixel_spacing': [float(x) for x in ds0.PixelSpacing] if hasattr(ds0, 'PixelSpacing') else [1.0, 1.0],
        'slice_thickness': float(getattr(ds0, 'SliceThickness', 2.0)),
    }
    return volume, meta


def load_multiphase_dicom(series_dir: Path, n_expected_phases: int = 5) -> Dict[int, np.ndarray]:
    """Load multi-phase DCE DICOM series, split by TemporalPositionIdentifier."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return {}

    # Group by temporal position
    phase_slices = {}
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        tp = int(getattr(ds, 'TemporalPositionIdentifier', 1))
        loc = float(getattr(ds, 'SliceLocation', 0) or getattr(ds, 'InstanceNumber', 0))
        if tp not in phase_slices:
            phase_slices[tp] = []
        phase_slices[tp].append((loc, ds.pixel_array.astype(np.float32)))

    # Sort each phase by slice location and stack into volumes
    phases = {}
    for tp, slices in phase_slices.items():
        slices.sort(key=lambda x: x[0])
        phases[tp] = np.stack([s[1] for s in slices], axis=0)  # (Z, H, W)

    return phases


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def find_peak_slice(volume: np.ndarray) -> int:
    """Find slice with highest mean intensity (proxy for tumor without mask)."""
    means = volume.mean(axis=(1, 2))
    return int(np.argmax(means))


def process_patient(
    patient_id: str,
    mask_dir: Path,
    multi_dir: Path,
    roi_dir: Path,
    mha_input_dir: Path,
    mha_gt_dir: Path,
    mha_mask_dir: Path,
    norm_mean: float,
    norm_std: float,
    extra_slices: int,
) -> Tuple[str, int]:
    """Process a single patient."""
    try:
        # Load pre-contrast
        pre_vol, meta = load_dicom_series(mask_dir)
        if pre_vol is None:
            return patient_id, 0

        # Load multi-phase DCE
        phases = load_multiphase_dicom(multi_dir)
        if not phases:
            return patient_id, 0

        n_slices = pre_vol.shape[0]

        # Find global peak phase
        peak_phase = 1
        peak_val = -np.inf
        for tp, vol in phases.items():
            m = float(vol.mean())
            if m > peak_val:
                peak_val = m
                peak_phase = tp

        peak_vol = phases[peak_phase]

        # Find peak slice using subtraction (enhancement)
        if peak_vol.shape == pre_vol.shape:
            enhancement = peak_vol - pre_vol
            slice_enhance = enhancement.mean(axis=(1, 2))
            peak_idx = int(np.argmax(slice_enhance))
        else:
            peak_idx = find_peak_slice(peak_vol)

        # Load ROI mask if available
        has_roi = False
        roi_vol = None
        if roi_dir and roi_dir.exists():
            # ROI is DICOM SEG - try to load
            seg_files = list(roi_dir.glob("*.dcm"))
            if seg_files:
                try:
                    ds = pydicom.dcmread(str(seg_files[0]))
                    if hasattr(ds, 'pixel_array'):
                        roi_vol = ds.pixel_array.astype(np.float32)
                        has_roi = True
                except:
                    pass

        # Extract peak ± extra_slices
        n_extracted = 0
        for offset in range(-extra_slices, extra_slices + 1):
            slice_idx = peak_idx + offset
            if slice_idx < 0 or slice_idx >= n_slices:
                continue
            if slice_idx >= peak_vol.shape[0]:
                continue

            # Extract 2D slices
            pre_2d = pre_vol[slice_idx]
            gt_2d = peak_vol[slice_idx]

            # Mask: use ROI if available, else zeros
            if has_roi and roi_vol is not None and roi_vol.ndim >= 2:
                if roi_vol.ndim == 3 and slice_idx < roi_vol.shape[0]:
                    mask_2d = (roi_vol[slice_idx] > 0).astype(np.int16)
                else:
                    mask_2d = np.zeros_like(pre_2d, dtype=np.int16)
            else:
                mask_2d = np.zeros_like(pre_2d, dtype=np.int16)

            # Z-score normalize
            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            gt_norm = zscore(gt_2d, norm_mean, norm_std)

            # Naming
            if offset == 0:
                fname = f"AMBL_{patient_id}"
            else:
                fname = f"AMBL_{patient_id}_s{offset:+d}"

            # Skip if exists (resume)
            if (mha_input_dir / f"{fname}.mha").exists():
                n_extracted += 1
                continue

            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(gt_norm), str(mha_gt_dir / f"{fname}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(mask_2d), str(mha_mask_dir / f"{fname}.mha"))
            n_extracted += 1

        return patient_id, n_extracted

    except Exception as e:
        logger.error(f"ERROR {patient_id}: {e}")
        return patient_id, -1


def main():
    parser = argparse.ArgumentParser(description="Convert Advanced-MRI-Breast-Lesions DICOM to MHA")
    parser.add_argument("--dicom_dir", required=True, help="Downloaded TCIA DICOM directory")
    parser.add_argument("--output_dir", required=True, help="Output (e.g. data_multislice_v3/train)")
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--extra_slices", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
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

    dicom_dir = Path(args.dicom_dir)

    # Map series UIDs to their descriptions by reading one DICOM per dir
    logger.info("Scanning DICOM directories...")
    series_dirs = [d for d in dicom_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(series_dirs)} series directories")

    # Classify series by reading first DICOM in each
    patient_data = {}  # patient_id -> {mask_dir, multi_dir, roi_dir}
    for sd in series_dirs:
        dcm_files = list(sd.glob("*.dcm"))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            pid = str(getattr(ds, 'PatientID', 'unknown'))
            desc = str(getattr(ds, 'SeriesDescription', ''))

            if pid not in patient_data:
                patient_data[pid] = {}

            if 'MASK' in desc.upper() and 'VIBRANT' in desc.upper():
                patient_data[pid]['mask_dir'] = sd
            elif 'MULTIPHASE' in desc.upper():
                patient_data[pid]['multi_dir'] = sd
            elif desc == 'ROI':
                patient_data[pid]['roi_dir'] = sd
        except:
            continue

    # Filter to patients with both MASK and MultiPhase
    valid_patients = {pid: data for pid, data in patient_data.items()
                      if 'mask_dir' in data and 'multi_dir' in data}
    logger.info(f"Valid patients (have MASK + MultiPhase): {len(valid_patients)}")

    # Process patients
    total_slices = 0
    errors = 0
    for i, (pid, data) in enumerate(sorted(valid_patients.items()), 1):
        pid_clean = pid.replace('-', '_')
        result_pid, n = process_patient(
            patient_id=pid_clean,
            mask_dir=data['mask_dir'],
            multi_dir=data['multi_dir'],
            roi_dir=data.get('roi_dir'),
            mha_input_dir=mha_input_dir,
            mha_gt_dir=mha_gt_dir,
            mha_mask_dir=mha_mask_dir,
            norm_mean=norm_mean,
            norm_std=norm_std,
            extra_slices=args.extra_slices,
        )
        if n < 0:
            errors += 1
        else:
            total_slices += n

        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(valid_patients)} patients, {total_slices} slices, {errors} errors")

    logger.info(f"\nDone. {total_slices} slices from {len(valid_patients)} patients ({errors} errors) → {output_dir}")


if __name__ == "__main__":
    main()
