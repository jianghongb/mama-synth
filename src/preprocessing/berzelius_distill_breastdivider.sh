#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J distill_930
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# BreastDivider 3D → 2D distillation on Berzelius
# Downloads BreastDividerDataset from HuggingFace, extracts 2D slices, trains nnUNet 2D
#
set -e

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
WORK=$PROJ/distill_breastdivider
mkdir -p $WORK

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

# Install deps if missing
pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub
pip show nibabel > /dev/null 2>&1 || pip install nibabel
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2

export nnUNet_raw="$WORK/nnUNet_raw"
export nnUNet_preprocessed="$WORK/nnUNet_preprocessed"
export nnUNet_results="$WORK/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# ============================================================
# Step 1: Download BreastDividerDataset from HuggingFace
# ============================================================
BD_DIR="$WORK/BreastDividerDataset"
if [ ! -d "$BD_DIR/imagesTr_batch1" ]; then
    echo "=== Step 1: Downloading BreastDividerDataset ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Bubenpo/BreastDividerDataset', repo_type='dataset', local_dir='$BD_DIR')
"
else
    echo "=== Step 1: Dataset already exists, skipping download ==="
fi

# ============================================================
# Step 2: Extract 2D slices → nnUNet format
# ============================================================
DATASET_DIR="$WORK/Dataset930_BreastDivider2D"
if [ ! -d "$DATASET_DIR/imagesTr" ]; then
    echo "=== Step 2: Extracting 2D slices ==="
    python -c "
import nibabel as nib
import numpy as np
import json
from pathlib import Path

bd_dir = Path('$BD_DIR')
out_dir = Path('$DATASET_DIR')
images_dir = out_dir / 'imagesTr'
labels_dir = out_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Process both batches
count = 0
THRESHOLD = 64  # skip images with min dim < 64

for batch in ['imagesTr_batch1', 'imagesTr_batch2']:
    img_batch = bd_dir / batch
    lbl_batch = bd_dir / batch.replace('imagesTr', 'labelsTr')
    if not img_batch.exists():
        continue
    for img_f in sorted(img_batch.glob('*_0000.nii.gz')):
        stem = img_f.name.replace('_0000.nii.gz', '')
        lbl_f = lbl_batch / f'{stem}.nii.gz'
        if not lbl_f.exists():
            continue

        img = nib.load(str(img_f))
        lbl = nib.load(str(lbl_f))
        img_data = img.get_fdata()
        lbl_data = lbl.get_fdata()

        # Find middle axial slice
        if img_data.ndim == 3:
            mid = img_data.shape[2] // 2
            img_2d = img_data[:, :, mid].astype(np.float32)
            lbl_2d = lbl_data[:, :, mid].astype(np.uint8)
        else:
            img_2d = img_data.astype(np.float32)
            lbl_2d = lbl_data.astype(np.uint8)

        # Skip too small
        if min(img_2d.shape[:2]) < THRESHOLD:
            continue

        # Binarize label (left=1, right=2 → breast=1)
        lbl_2d = (lbl_2d > 0).astype(np.uint8)

        # Save as (H, W, 1) for nnUNet 2D
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img_2d[:, :, np.newaxis], affine), str(images_dir / f'{stem}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(lbl_2d[:, :, np.newaxis], affine), str(labels_dir / f'{stem}.nii.gz'))
        count += 1

        if count % 1000 == 0:
            print(f'  {count} cases processed...')

# dataset.json
ds = {
    'channel_names': {'0': 'MRI'},
    'labels': {'background': 0, 'breast': 1},
    'numTraining': count,
    'file_ending': '.nii.gz'
}
with open(out_dir / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=2)
print(f'Done: {count} 2D cases extracted')
"
else
    echo "=== Step 2: Dataset already extracted, skipping ==="
fi

# Link to nnUNet_raw
ln -sfn $DATASET_DIR $nnUNet_raw/Dataset930_BreastDivider2D

# ============================================================
# Step 3: Plan + Preprocess + Train
# ============================================================
echo "=== Step 3: Plan and preprocess ==="
export TORCHDYNAMO_DISABLE=1
nnUNetv2_plan_and_preprocess -d 930 -c 2d --verify_dataset_integrity

echo "=== Step 4: Train fold 0 ==="
nnUNetv2_train 930 2d 0 --npz

echo "=== Done ==="
