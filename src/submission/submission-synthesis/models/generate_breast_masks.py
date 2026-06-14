"""Batch-generate breast masks for training data using Dataset910_BreastSegNet."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

os.environ.setdefault("nnUNet_raw", "/tmp/nnunet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnunet_preprocessed")
os.environ.setdefault("nnUNet_results", "/tmp/nnunet_results")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

BREAST_LABELS = {1, 2, 5, 6, 9}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        'spacing': [1.0, 1.0, 1.0],
    }

    mha_files = sorted(input_dir.glob("*.mha"))
    print(f"Processing {len(mha_files)} files...")

    for f in tqdm(mha_files):
        out_path = output_dir / f.name
        if out_path.exists():
            continue

        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        sl = arr.squeeze()

        # Shape for nnUNet: (1, 1, H, W)
        input_arr = sl[np.newaxis, np.newaxis, :, :]
        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)

        if pred.ndim == 3:
            pred = pred[0]

        breast_mask = np.isin(pred, list(BREAST_LABELS)).astype(np.int16)

        # Save with same metadata
        if arr.ndim == 3:
            breast_mask = breast_mask[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(breast_mask)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

    print(f"Done. Masks saved to {output_dir}")


if __name__ == "__main__":
    main()
