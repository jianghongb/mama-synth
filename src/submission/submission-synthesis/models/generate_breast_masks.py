"""Batch-generate breast masks using nnUNet segmentation models.

Supports:
  - Dataset910 (1-channel T1 input, 10-class output)
  - Dataset932 (4-channel kinetic input: P0, P1, d_early, d_late)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Dataset910: labels that are "breast" (tissue, vessel, lesion, lymphnode, implant)
BREAST_LABELS_910 = {1, 2, 5, 6, 9}
# Dataset932: labels 1=breast, 2=FGT, 3=tumor → all are breast region
BREAST_LABELS_932 = {1, 2, 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="pre-contrast MHA dir")
    parser.add_argument("--gt_dir", default="", help="ground_truth (subtraction) dir, required for 4ch model")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_type", choices=["910", "932"], default="910",
                        help="910=single-channel T1, 932=4-channel kinetic")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "932" and not args.gt_dir:
        parser.error("--gt_dir required for 4-channel model (932)")
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model type: Dataset{args.model_type}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_dir,
        use_folds=(args.fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    print("Model loaded.")

    props = {
        'sitk_stuff': {
            'spacing': (1.0, 1.0, 1.0),
            'origin': (0.0, 0.0, 0.0),
            'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
        'spacing': [2.0, 0.703, 0.703] if args.model_type == "932" else [1.0, 1.0, 1.0],
    }

    breast_labels = BREAST_LABELS_932 if args.model_type == "932" else BREAST_LABELS_910
    mha_files = sorted(input_dir.glob("*.mha"))
    print(f"Processing {len(mha_files)} files...")

    for f in tqdm(mha_files):
        out_path = output_dir / f.name
        if out_path.exists():
            continue

        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        sl = arr.squeeze()  # (H, W)

        if args.model_type == "932":
            # 4-channel: P0, P1, d_early, d_late
            gt_path = gt_dir / f.name
            if not gt_path.exists():
                continue
            gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.float32).squeeze()

            p0 = sl                    # pre-contrast
            d_early = gt_arr           # subtraction (post - pre) = ground_truth
            p1 = p0 + d_early          # reconstruct post = pre + subtraction
            d_late = d_early            # approximate (only 1 phase available)

            # Shape: (1, 4, 1, H, W) for 3D model with single slice
            input_arr = np.stack([p0, p1, d_early, d_late])[:, np.newaxis, :, :]  # (4, 1, H, W)
        else:
            # 1-channel: T1 only
            input_arr = sl[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)

        # Squeeze to 2D
        while pred.ndim > 2:
            pred = pred[0]

        breast_mask = np.isin(pred, list(breast_labels)).astype(np.uint8)

        # Morphological post-processing: connect disconnected breast regions
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
        # Fill internal holes
        h, w = breast_mask.shape
        flood = np.zeros((h+2, w+2), np.uint8)
        mask_inv = breast_mask.copy()
        cv2.floodFill(mask_inv, flood, (0, 0), 1)
        breast_mask = (breast_mask | (1 - mask_inv)).astype(np.int16)

        if arr.ndim == 3:
            breast_mask = breast_mask[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(breast_mask)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

    print(f"Done. Masks saved to {output_dir}")


if __name__ == "__main__":
    main()
