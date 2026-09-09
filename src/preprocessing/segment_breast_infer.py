#!/usr/bin/env python3
"""Breast segmentation inference with the 3D BreastDivider nnU-Net model.

The model (nnUNetTrainer__nnUNetPlans__3d_dac) takes a single-channel 3D MR
volume and outputs labels {0: background, 1: left breast, 2: right breast}.
This script runs it over a folder of volumes and writes, per case, a binary
breast mask (label > 0) as .nii.gz. Optionally also writes the raw left/right
label map.

No paths are hard-coded; pass them on the command line.

Prerequisites:
    - nnunetv2 installed in the active environment
    - the model folder contains: dataset.json, plans.json, and
      fold_<f>/checkpoint_final.pth

Example (on MAIA):
    python segment_breast_infer.py \
        --model_dir /home/maia-user/jh/BreastDividerModel \
        --input_dir  /path/to/volumes \
        --output_dir /path/to/masks \
        --fold 0 --checkpoint checkpoint_final.pth

Input naming:
    nnU-Net expects channel-suffixed files (<case>_0000.nii.gz). If your inputs
    are plain <case>.nii.gz, pass --auto_suffix to stage temporary symlinks with
    the _0000 suffix automatically.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def build_predictor(model_dir: str, fold: int, checkpoint: str, device_str: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device(device_str)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=(fold,), checkpoint_name=checkpoint,
    )
    return predictor


def stage_inputs(input_dir: Path, auto_suffix: bool) -> tuple[Path, bool]:
    """Return a directory whose files carry the nnU-Net _0000 channel suffix.

    If files already look like <case>_0000.nii.gz, use input_dir as is.
    Otherwise, when auto_suffix is set, create a temp dir of symlinks with the
    suffix added. Returns (dir, is_temp).
    """
    files = sorted(input_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"no .nii.gz files in {input_dir}")
    if all("_0000" in f.name for f in files):
        return input_dir, False
    if not auto_suffix:
        raise ValueError(
            "inputs are not channel-suffixed (<case>_0000.nii.gz). "
            "Re-run with --auto_suffix to stage them automatically."
        )
    tmp = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
    for f in files:
        case = f.name[: -len(".nii.gz")]
        (tmp / f"{case}_0000.nii.gz").symlink_to(f.resolve())
    return tmp, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="BreastDivider model folder (contains dataset.json, plans.json, fold_*/)")
    ap.add_argument("--input_dir", required=True, help="folder of 3D .nii.gz volumes")
    ap.add_argument("--output_dir", required=True, help="where breast masks are written")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--auto_suffix", action="store_true",
                    help="stage <case>.nii.gz as <case>_0000.nii.gz for nnU-Net")
    ap.add_argument("--save_labelmap", action="store_true",
                    help="also save the raw left/right label map (0/1/2)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged, is_temp = stage_inputs(input_dir, args.auto_suffix)
    raw_out = Path(tempfile.mkdtemp(prefix="nnunet_out_"))

    try:
        predictor = build_predictor(args.model_dir, args.fold, args.checkpoint, args.device)
        print(f"model loaded from {args.model_dir} (fold {args.fold}, {args.checkpoint})")
        print(f"segmenting {len(list(staged.glob('*_0000.nii.gz')))} volume(s) ...")

        predictor.predict_from_files(
            str(staged), str(raw_out),
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        # Post-process each label map into a binary breast mask (label > 0).
        n = 0
        for pred_path in sorted(raw_out.glob("*.nii.gz")):
            img = sitk.ReadImage(str(pred_path))
            arr = sitk.GetArrayFromImage(img)
            breast = (arr > 0).astype(np.uint8)  # left(1) + right(2) = breast
            mask_img = sitk.GetImageFromArray(breast)
            mask_img.CopyInformation(img)
            sitk.WriteImage(mask_img, str(output_dir / pred_path.name))
            if args.save_labelmap:
                sitk.WriteImage(img, str(output_dir / f"labelmap_{pred_path.name}"))
            frac = 100.0 * breast.sum() / breast.size
            print(f"  {pred_path.name}: breast = {frac:.1f}% of volume")
            n += 1
        print(f"done: {n} mask(s) -> {output_dir}")
    finally:
        shutil.rmtree(raw_out, ignore_errors=True)
        if is_temp:
            shutil.rmtree(staged, ignore_errors=True)


if __name__ == "__main__":
    main()
