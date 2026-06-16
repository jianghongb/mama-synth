#!/bin/bash
# Train a single-channel (T1 pre-contrast only) breast segmentation model
# using pseudo labels from Dataset932 (4ch model).
#
# Steps:
# 1. Generate pseudo labels from Dataset932 on training data
# 2. Prepare nnUNet dataset format (Dataset933)
# 3. Plan & preprocess
# 4. Train nnUNet 2d

set -e

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
MAMA=$PROJ/mama-synth/src/submission/submission-synthesis

# Paths
TRAIN_INPUT=$PROJ/data_split/train/mha/input
TRAIN_GT=$PROJ/data_split/train/mha/ground_truth
PSEUDO_LABELS=$PROJ/data_split/train/mha/breast_mask_932
MODEL_932=$PROJ/weights/breast_seg/nnUNetTrainer__nnUNetPlans__3d_fullres

# nnUNet dirs
export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

DATASET_ID=933
DATASET_NAME="Dataset${DATASET_ID}_BreastSeg1ch"
RAW_DIR=$nnUNet_raw/$DATASET_NAME

echo "=== Step 1: Generate pseudo labels from Dataset932 ==="
if [ ! -d "$PSEUDO_LABELS" ]; then
    python $MAMA/models/generate_breast_masks.py \
        --input_dir $TRAIN_INPUT \
        --gt_dir $TRAIN_GT \
        --output_dir $PSEUDO_LABELS \
        --model_dir $MODEL_932 \
        --model_type 932
fi
echo "Pseudo labels: $(ls $PSEUDO_LABELS | wc -l) files"

echo "=== Step 2: Prepare nnUNet raw dataset ==="
python -c "
import os, json, shutil, SimpleITK as sitk, numpy as np
from pathlib import Path

input_dir = Path('$TRAIN_INPUT')
label_dir = Path('$PSEUDO_LABELS')
raw_dir = Path('$RAW_DIR')

images_dir = raw_dir / 'imagesTr'
labels_dir = raw_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Convert MHA to NIfTI for nnUNet
files = sorted([f for f in os.listdir(str(input_dir)) if f.endswith('.mha')])
n = 0
for f in files:
    label_path = label_dir / f
    if not label_path.exists():
        continue
    case_id = f.replace('.mha', '')
    
    # Image: single channel _0000.nii.gz
    img = sitk.ReadImage(str(input_dir / f))
    sitk.WriteImage(img, str(images_dir / f'{case_id}_0000.nii.gz'))
    
    # Label: breast seg from pseudo labels
    lbl = sitk.ReadImage(str(label_path))
    sitk.WriteImage(lbl, str(labels_dir / f'{case_id}.nii.gz'))
    n += 1

# dataset.json
ds = {
    'channel_names': {'0': 'T1_pre'},
    'labels': {'background': 0, 'breast': 1, 'FGT': 2, 'tumor': 3},
    'numTraining': n,
    'file_ending': '.nii.gz',
    'name': '$DATASET_NAME'
}
with open(str(raw_dir / 'dataset.json'), 'w') as fh:
    json.dump(ds, fh, indent=2)

print(f'Prepared {n} cases in {raw_dir}')
"

echo "=== Step 3: Plan & Preprocess ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 2d --verify_dataset_integrity

echo "=== Step 4: Train ==="
nnUNetv2_train $DATASET_ID 2d 0 --npz
