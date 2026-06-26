#!/bin/bash
# Cleanup bad cases from Dataset930 and restart training
# Run on MAIA: bash cleanup_and_train.sh

set -e
cd /home/maia-user/jh

# Activate env
conda activate /home/maia-user/jh/envs/distill

export nnUNet_raw="/home/maia-user/jh/distill/nnUNet_raw"
export nnUNet_preprocessed="/home/maia-user/jh/distill/nnUNet_preprocessed"
export nnUNet_results="/home/maia-user/jh/distill/nnUNet_results"

# Step 1: Remove bad cases from raw data + preprocessed
echo "=== Step 1: Cleanup bad cases ==="
python -c "
import nibabel as nib
import os, json
from pathlib import Path

img_dir = Path('/home/maia-user/jh/distill/Dataset930/imagesTr')
lbl_dir = Path('/home/maia-user/jh/distill/Dataset930/labelsTr')
prep_dir = Path('/home/maia-user/jh/distill/nnUNet_preprocessed/Dataset930_BreastDivider2D/nnUNetPlans_2d')

# Find bad cases (min dim < 64)
THRESHOLD = 64
bad = []
for f in sorted(img_dir.glob('*.nii.gz')):
    img = nib.load(str(f))
    sz = img.shape[:2]
    if min(sz) < THRESHOLD:
        bad.append(f.stem.replace('_0000', ''))

print(f'Found {len(bad)} bad cases (min dim < {THRESHOLD})')

# Remove from raw
for stem in bad:
    for p in [img_dir / f'{stem}_0000.nii.gz', lbl_dir / f'{stem}.nii.gz']:
        if p.exists():
            os.remove(p)

# Remove from preprocessed
removed = 0
if prep_dir.exists():
    existing = {f.stem.replace('_0000','') for f in img_dir.glob('*.nii.gz')}
    for f in list(prep_dir.glob('*.b2nd')):
        stem = f.stem.replace('_seg','')
        if stem not in existing:
            os.remove(f)
            removed += 1

# Update dataset.json
remaining = len(list(img_dir.glob('*.nii.gz')))
ds_json = Path('/home/maia-user/jh/distill/Dataset930/dataset.json')
with open(ds_json) as f:
    d = json.load(f)
d['numTraining'] = remaining
with open(ds_json, 'w') as f:
    json.dump(d, f, indent=2)

print(f'Removed {len(bad)} raw + {removed} preprocessed files')
print(f'Remaining: {remaining} cases')
"

# Step 2: Delete splits (force regeneration)
echo "=== Step 2: Remove old splits ==="
rm -f /home/maia-user/jh/distill/nnUNet_preprocessed/Dataset930_BreastDivider2D/splits_final.json

# Step 3: Train
echo "=== Step 3: Start training ==="
nohup nnUNetv2_train 930 2d 0 --npz > /home/maia-user/jh/distill/train.log 2>&1 &
echo "Training started (PID: $!)"
echo "Monitor: tail -f /home/maia-user/jh/distill/train.log"
