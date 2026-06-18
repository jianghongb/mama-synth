"""
preprocess_ambl.py — AMBL dataset 3D→2D preprocessing with breast masking.

AMBL structure:
  images/PATIENT_phaseN/PATIENT_phaseN_0000.nii.gz  (T1 image)
  images/PATIENT_phaseN/PATIENT_phaseN_0001.nii.gz  (subtraction)
  segmentations/PATIENT_phaseN.nii.gz               (empty placeholders)

Phases: phase1=pre-contrast, phase2-5=post-contrast.

Pipeline:
1. Load phase1 (pre) and select peak from phase2-5 (post)
2. Run Exp4x breast segmentation (4ch)
3. Apply breast mask, select middle slice, z-score normalise
4. Save as .mha

Usage:
    python preprocess_ambl.py \
        --image_dir /path/to/ambl_nifti/images \
        --output_dir /path/to/output \
        --global_stats src/preprocessing/training_pre_stats.json \
        --breast_model_dir /path/to/exp4x/.../nnUNetTrainer__nnUNetPlans__3d_fullres
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")


class BreastSegmentor:
    """Lazy-loaded Exp4x breast segmentor."""
    BREAST_LABELS = {1, 2, 3}

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = model_dir
        self._predictor = None
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self._device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device_str = "mps"
            else:
                self._device_str = "cpu"
        else:
            self._device_str = device

    def _init(self):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device(self._device_str), verbose=False, allow_tqdm=True)
        self._predictor.initialize_from_trained_model_folder(
            self.model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        logger.info(f"Breast seg model loaded on {self._device_str}")

    def predict(self, p0, p1, d_early, d_late):
        if self._predictor is None:
            self._init()
        input_arr = np.stack([p0, p1, d_early, d_late]).astype(np.float32)
        props = {"sitk_stuff": {"spacing": (2.0, 0.703, 0.703), "origin": (0,0,0),
                  "direction": (1,0,0,0,1,0,0,0,1)}, "spacing": [2.0, 0.703, 0.703]}
        pred = self._predictor.predict_single_npy_array(input_arr, props, None, None, False)
        return np.isin(pred, list(self.BREAST_LABELS)).astype(np.float32)


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img


def determine_slice_axis(shape):
    size_to_axes = {}
    for ax, s in enumerate(shape):
        size_to_axes.setdefault(s, []).append(ax)
    n = len(size_to_axes)
    if n == 1:
        return 2
    if n == 3:
        raise ValueError(f"Ambiguous FOV: {shape}")
    return next(axes[0] for axes in size_to_axes.values() if len(axes) == 1)


def zscore(arr, mean, std):
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="AMBL preprocessing")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True)
    parser.add_argument("--breast_model_dir", required=True)
    parser.add_argument("--tumor_model_dir", default=None,
                        help="Path to full_image_dce_mri_tumor_segmentation model dir")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
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

    segmentor = BreastSegmentor(args.breast_model_dir, device=args.device)
    image_dir = Path(args.image_dir)

    # Tumor segmentation model (2D, single-channel T1, post-contrast input)
    tumor_predictor = None
    if args.tumor_model_dir and Path(args.tumor_model_dir).exists():
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        device_str = args.device
        if device_str == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"
        tumor_predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device(device_str), verbose=False, allow_tqdm=False)
        tumor_predictor.initialize_from_trained_model_folder(
            args.tumor_model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        logger.info("Tumor segmentation model loaded")

    # Group folders by patient
    all_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir() and "phase" in d.name)
    patients = {}
    for d in all_dirs:
        # AMBL-001_phase1 -> patient=AMBL-001, phase=1
        parts = d.name.rsplit("_phase", 1)
        if len(parts) == 2:
            patient, phase_str = parts
            patients.setdefault(patient, {})[int(phase_str)] = d

    logger.info(f"Found {len(patients)} patients")
    results = []

    for patient_id, phase_dirs in patients.items():
        try:
            if 1 not in phase_dirs:
                logger.warning(f"SKIP {patient_id}: no phase1 (pre-contrast)")
                continue

            # Load pre-contrast (phase1, channel 0000)
            pre_path = phase_dirs[1] / f"{patient_id}_phase1_0000.nii.gz"
            if not pre_path.exists():
                logger.warning(f"SKIP {patient_id}: {pre_path.name} not found")
                continue
            pre, _ = load_nifti(pre_path)

            # Load post-contrast phases
            post_phases = {}
            for phase_num in sorted(phase_dirs.keys()):
                if phase_num == 1:
                    continue
                pp = phase_dirs[phase_num] / f"{patient_id}_phase{phase_num}_0000.nii.gz"
                if pp.exists():
                    post_phases[phase_num], _ = load_nifti(pp)

            if not post_phases:
                logger.warning(f"SKIP {patient_id}: no post-contrast phases")
                continue

            # Select peak phase (highest mean intensity overall)
            peak_phase = max(post_phases, key=lambda k: float(post_phases[k].mean()))
            post = post_phases[peak_phase]

            # Breast segmentation (4ch)
            p1_peak = post
            d_early = p1_peak - pre
            # Use last available phase for d_late
            last_phase = post_phases[max(post_phases.keys())]
            d_late = last_phase - p1_peak

            logger.info(f"[{patient_id}] Breast segmentation...")
            breast_mask = segmentor.predict(pre, p1_peak, d_early, d_late)

            # Apply breast mask
            pre_masked = pre * breast_mask
            post_masked = post * breast_mask

            # Determine slice axis and select middle slice (no tumour seg available)
            try:
                axis = determine_slice_axis(pre.shape)
            except ValueError as e:
                if args.skip_ambiguous_shapes:
                    logger.warning(f"SKIP {patient_id}: {e}")
                    continue
                raise

            # Select middle slice of the volume
            n_slices = pre.shape[axis]
            slice_idx = n_slices // 2

            # Extract 2D
            pre_2d = np.take(pre_masked, slice_idx, axis=axis)
            post_2d = np.take(post_masked, slice_idx, axis=axis)
            breast_2d = np.take(breast_mask, slice_idx, axis=axis).astype(np.int16)

            # Z-score normalise
            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            post_norm = zscore(post_2d, norm_mean, norm_std)

            # Tumor segmentation on post-contrast 2D slice
            tumor_2d = np.zeros_like(breast_2d)
            if tumor_predictor is not None:
                t_input = post_2d[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
                props = {"sitk_stuff": {"spacing": (1.0, 1.0, 1.0), "origin": (0,0,0),
                          "direction": (1,0,0,0,1,0,0,0,1)}, "spacing": [1.0, 1.0, 1.0]}
                t_pred = tumor_predictor.predict_single_npy_array(t_input, props, None, None, False)
                while t_pred.ndim > 2:
                    t_pred = t_pred[0]
                tumor_2d = (t_pred > 0).astype(np.int16)

            # Rotate 90° CCW
            pre_norm = np.rot90(pre_norm, k=1)
            post_norm = np.rot90(post_norm, k=1)
            breast_2d = np.rot90(breast_2d, k=1)
            tumor_2d = np.rot90(tumor_2d, k=1)

            # Save
            pid = f"AMBL_{patient_id}"
            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f"{pid}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(post_norm), str(mha_gt_dir / f"{pid}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(tumor_2d), str(mha_mask_dir / f"{pid}.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(breast_2d), str(mha_breast_dir / f"{pid}.mha"))

            results.append({"patient_id": pid, "peak_phase": peak_phase, "slice_idx": slice_idx, "axis": axis})
            logger.info(f"  ✓ {pid}: axis={axis}, slice={slice_idx}, peak=phase{peak_phase}")

        except Exception as e:
            logger.error(f"ERROR {patient_id}: {e}")
            continue

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "report.csv", index=False)
        logger.info(f"\nDone. {len(results)} patients → {output_dir}")


if __name__ == "__main__":
    main()
