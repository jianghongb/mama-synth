"""
mask_and_preprocess.py — Breast-masked 3D→2D preprocessing pipeline.

Pipeline:
1. Run Exp4x (Dataset932, 4ch) nnUNet to get 3D breast segmentation.
2. Apply breast mask to all DCE phases (zero-out chest wall / background).
3. Select peak enhancement phase & largest tumour slice (same as preprocess.py).
4. Z-score normalise and save 2D .mha slices.

Usage:
    python mask_and_preprocess.py \
        --image_dir ./images \
        --seg_dir ./segmentations \
        --output_dir ./output_masked \
        --global_stats ./src/preprocessing/training_pre_stats.json \
        --breast_model_dir /path/to/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --skip_ambiguous_shapes
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# nnUNet env vars (needed for import)
os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnUNet_results")


class BreastSegmentor:
    """Lazy-loaded nnUNet breast segmentor (Dataset932, 4ch)."""

    BREAST_LABELS = {1, 2, 3}  # breast, FGT, tumor

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

    def _init_predictor(self):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = torch.device(self._device_str)
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=True,
            device=device, verbose=False, allow_tqdm=True,
        )
        self._predictor.initialize_from_trained_model_folder(
            self.model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth",
        )
        logger.info(f"Breast seg model loaded on {self._device_str}")

    def predict(self, p0: np.ndarray, p1: np.ndarray, d_early: np.ndarray, d_late: np.ndarray) -> np.ndarray:
        """Run 4ch segmentation, return binary breast mask (same shape as inputs)."""
        if self._predictor is None:
            self._init_predictor()

        # nnUNet expects (C, D, H, W)
        input_arr = np.stack([p0, p1, d_early, d_late]).astype(np.float32)
        props = {
            "sitk_stuff": {
                "spacing": (2.0, 0.703, 0.703),
                "origin": (0.0, 0.0, 0.0),
                "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
            },
            "spacing": [2.0, 0.703, 0.703],
        }
        pred = self._predictor.predict_single_npy_array(input_arr, props, None, None, False)
        return np.isin(pred, list(self.BREAST_LABELS)).astype(np.float32)


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI, return (float32 array, image object)."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img


def determine_slice_axis(shape: Tuple[int, ...]) -> int:
    """Return through-plane axis. Raises ValueError if ambiguous."""
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}")
    size_to_axes = {}
    for ax, s in enumerate(shape):
        size_to_axes.setdefault(s, []).append(ax)
    n = len(size_to_axes)
    if n == 1:
        return 2  # cubic fallback
    if n == 3:
        raise ValueError(f"Ambiguous FOV: all dims differ {shape}")
    return next(axes[0] for axes in size_to_axes.values() if len(axes) == 1)


def find_largest_slice(seg: np.ndarray, axis: int) -> int:
    """Return index of slice with largest label area along axis."""
    others = tuple(i for i in range(seg.ndim) if i != axis)
    return int(np.argmax(np.sum(seg > 0, axis=others)))


def zscore(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Breast-masked 3D→2D preprocessing")
    parser.add_argument("--image_dir", required=True, help="Per-patient phase volumes")
    parser.add_argument("--seg_dir", required=True, help="Tumour segmentation dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_stats", required=True, help="JSON with mean/std")
    parser.add_argument("--breast_model_dir", required=True,
                        help="Path to nnUNetTrainer__nnUNetPlans__3d_fullres folder")
    parser.add_argument("--skip_ambiguous_shapes", action="store_true", default=False)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--exclude_list", default=None,
                        help="Text file with patient IDs to exclude (one per line)")
    args = parser.parse_args()

    # Load exclusion list
    exclude_ids = set()
    if args.exclude_list:
        with open(args.exclude_list) as f:
            exclude_ids = {line.strip() for line in f if line.strip()}
        logger.info(f"Excluding {len(exclude_ids)} patients from {args.exclude_list}")

    with open(args.global_stats) as f:
        stats = json.load(f)
    norm_mean, norm_std = float(stats["mean"]), float(stats["std"])
    logger.info(f"Normalisation: mean={norm_mean:.4f}, std={norm_std:.4f}")

    output_dir = Path(args.output_dir)
    mha_input_dir = output_dir / "mha" / "input"
    mha_gt_dir = output_dir / "mha" / "ground_truth"
    mha_mask_dir = output_dir / "mha" / "mask"
    mha_breast_dir = output_dir / "mha" / "breast_mask"
    for d in (mha_input_dir, mha_gt_dir, mha_mask_dir, mha_breast_dir):
        d.mkdir(parents=True, exist_ok=True)

    segmentor = BreastSegmentor(args.breast_model_dir, device=args.device)

    image_dir = Path(args.image_dir)
    seg_dir = Path(args.seg_dir)
    results = []

    patients = sorted(d for d in image_dir.iterdir() if d.is_dir())
    logger.info(f"Found {len(patients)} patients")
    if exclude_ids:
        patients = [p for p in patients if p.name not in exclude_ids]
        logger.info(f"After exclusion: {len(patients)} patients")

    for patient_dir in patients:
        patient_id = patient_dir.name
        # Skip if already processed
        if (mha_input_dir / f"{patient_id}.mha").exists():
            continue
        try:
            # --- Load phases ---
            phase_files = sorted(
                f for f in patient_dir.glob(f"{patient_id}_*.nii.gz")
            )
            if len(phase_files) < 2:
                logger.warning(f"SKIP {patient_id}: <2 phases")
                continue

            phases = {}
            ref_img = None
            for pf in phase_files:
                idx = int(pf.stem.replace(".nii", "").rsplit("_", 1)[1])
                arr, img = load_nifti(pf)
                phases[idx] = arr
                if ref_img is None:
                    ref_img = img

            # --- Load tumour segmentation ---
            seg_file = seg_dir / f"{patient_id}.nii.gz"
            if not seg_file.exists():
                candidates = list(seg_dir.glob(f"{patient_id}*"))
                seg_file = candidates[0] if candidates else None
            if seg_file is None or not seg_file.exists():
                logger.warning(f"SKIP {patient_id}: no tumour seg")
                continue
            tumour_seg, _ = load_nifti(seg_file)

            # --- Step 1: Breast segmentation (4ch) ---
            p0 = phases[min(phases.keys())]
            p1 = phases[sorted(phases.keys())[1]]
            last = phases[max(phases.keys())]
            d_early = p1 - p0
            d_late = last - p1

            logger.info(f"[{patient_id}] Running breast segmentation...")
            breast_mask = segmentor.predict(p0, p1, d_early, d_late)
            logger.info(f"  Breast mask: {breast_mask.sum():.0f} voxels")

            # --- Step 2: Apply breast mask to all phases ---
            masked_phases = {k: v * breast_mask for k, v in phases.items()}

            # --- Step 3: Find peak phase & slice ---
            # Determine slice axis from tumour seg shape
            try:
                axis = determine_slice_axis(tumour_seg.shape)
            except ValueError as e:
                if args.skip_ambiguous_shapes:
                    logger.warning(f"SKIP {patient_id}: {e}")
                    continue
                raise

            # Peak phase = highest mean tumour intensity (on masked data)
            peak_phase = min(phases.keys())
            peak_val = -np.inf
            for k, vol in masked_phases.items():
                tumour_voxels = vol[tumour_seg > 0]
                if len(tumour_voxels) > 0:
                    m = float(np.mean(tumour_voxels))
                    if m > peak_val:
                        peak_val = m
                        peak_phase = k

            pre_phase = min(phases.keys())
            slice_idx = find_largest_slice(tumour_seg, axis)

            # Extract 2D slices
            pre_2d = np.take(masked_phases[pre_phase], slice_idx, axis=axis)
            peak_2d = np.take(masked_phases[peak_phase], slice_idx, axis=axis)
            mask_2d = np.take(tumour_seg, slice_idx, axis=axis).astype(np.int16)
            breast_2d = np.take(breast_mask, slice_idx, axis=axis).astype(np.int16)

            # --- Step 4: Z-score normalise ---
            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            peak_norm = zscore(peak_2d, norm_mean, norm_std)

            # Rotate 90° CCW (match existing preprocess.py convention)
            pre_norm = np.rot90(pre_norm, k=1)
            peak_norm = np.rot90(peak_norm, k=1)
            mask_2d = np.rot90(mask_2d, k=1)
            breast_2d = np.rot90(breast_2d, k=1)

            # --- Save ---
            for arr, dirp, label in [
                (pre_norm, mha_input_dir, False),
                (peak_norm, mha_gt_dir, False),
                (mask_2d, mha_mask_dir, True),
                (breast_2d, mha_breast_dir, True),
            ]:
                arr_out = np.rint(arr).astype(np.int16) if label else arr.astype(np.float32)
                sitk.WriteImage(sitk.GetImageFromArray(arr_out), str(dirp / f"{patient_id}.mha"))

            results.append({
                "patient_id": patient_id,
                "pre_phase": pre_phase,
                "peak_phase": peak_phase,
                "slice_idx": slice_idx,
                "slice_axis": axis,
                "breast_voxels": int(breast_mask.sum()),
                "peak_mean_tumour": peak_val,
            })
            logger.info(f"  ✓ {patient_id}: axis={axis}, slice={slice_idx}, peak=phase{peak_phase}")

        except Exception as e:
            logger.error(f"ERROR {patient_id}: {e}")
            continue

    # Save report
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        report_path = output_dir / "report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"\nDone. {len(results)} patients processed → {output_dir}")
        logger.info(f"Report: {report_path}")
    else:
        logger.warning("No patients processed successfully.")


if __name__ == "__main__":
    main()
