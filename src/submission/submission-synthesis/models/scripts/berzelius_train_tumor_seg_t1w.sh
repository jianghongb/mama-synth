#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -J tumor_seg_t1w
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Train a 2D nnUNet to segment tumor from pre-contrast T1w (P0).
#
# Purpose: Enable the GAN to know WHERE the tumor is at inference time,
# so it can focus contrast enhancement on the correct region.
#
# Data source: data_multislice_v3/train/mha/{input, mask}
# Input: pre-contrast MHA (1 channel)
# Label: binary tumor mask
#
# Steps:
#   1. Convert MHA → nnUNet format (NIfTI + dataset.json)
#   2. Plan + preprocess
#   3. Train fold 0 (2D)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
export TORCHDYNAMO_DISABLE=1

DATASET_ID=940
DATASET_NAME="Dataset${DATASET_ID}_TumorSegT1w"
DATASET_DIR=$nnUNet_raw/$DATASET_NAME

INPUT_DIR=$PROJ/data_multislice_v3/train/mha/input
MASK_DIR=$PROJ/data_multislice_v3/train/mha/mask

echo "=== Step 1: Convert MHA → nnUNet format ==="

python -c "
import json, os, sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path('$INPUT_DIR')
MASK_DIR = Path('$MASK_DIR')
DATASET_DIR = Path('$DATASET_DIR')
IMAGES_DIR = DATASET_DIR / 'imagesTr'
LABELS_DIR = DATASET_DIR / 'labelsTr'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Get all paired files (input + mask both exist)
input_files = sorted(INPUT_DIR.glob('*.mha'))
print(f'Total input files: {len(input_files)}')

count = 0
skipped_empty = 0
for f in tqdm(input_files, desc='Converting'):
    mask_path = MASK_DIR / f.name
    if not mask_path.exists():
        continue

    # Read mask, skip if empty (no tumor)
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img).squeeze()
    if mask_arr.sum() < 10:  # skip near-empty masks
        skipped_empty += 1
        continue

    # Read input
    input_img = sitk.ReadImage(str(f))
    input_arr = sitk.GetArrayFromImage(input_img).astype(np.float32).squeeze()

    # Case ID: replace dots and special chars
    case_id = f.stem.replace('.', '_').replace('+', 'p').replace('-', 'm')

    # Save as NIfTI (nnUNet format: 3D with single slice)
    # Shape: (1, H, W) for 2D nnUNet
    affine = np.eye(4)

    # Input: case_id_0000.nii.gz
    out_img = sitk.GetImageFromArray(input_arr[np.newaxis, :, :])
    sitk.WriteImage(out_img, str(IMAGES_DIR / f'{case_id}_0000.nii.gz'))

    # Label: case_id.nii.gz (binary: 0=background, 1=tumor)
    mask_out = mask_arr.astype(np.uint8)[np.newaxis, :, :]
    out_mask = sitk.GetImageFromArray(mask_out)
    sitk.WriteImage(out_mask, str(LABELS_DIR / f'{case_id}.nii.gz'))

    count += 1

# Write dataset.json
dataset_json = {
    'channel_names': {'0': 'T1w_pre_contrast'},
    'labels': {'background': 0, 'tumor': 1},
    'numTraining': count,
    'file_ending': '.nii.gz',
}
with open(DATASET_DIR / 'dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f'')
print(f'Done: {count} training cases (skipped {skipped_empty} empty masks)')
print(f'Output: {DATASET_DIR}')
"

echo ""
echo "=== Step 2: nnUNet Plan + Preprocess ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 2d --verify_dataset_integrity

echo ""
echo "=== Step 3: Train fold 0 (2D) ==="
nnUNetv2_train $DATASET_ID 2d 0 --npz

echo ""
echo "=== Complete ==="
echo "Model: $nnUNet_results/$DATASET_NAME/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
echo ""
echo "Next steps:"
echo "  1. Run inference on training data to get predicted tumor masks"
echo "  2. Train GAN with input_nc=3 (pre + breast_mask + tumor_mask)"
