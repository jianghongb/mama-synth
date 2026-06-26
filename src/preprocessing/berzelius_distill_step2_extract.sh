#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J distill_extract
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_extract_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
set -e
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
WORK=$PROJ/distill_breastdivider

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show nibabel > /dev/null 2>&1 || pip install nibabel

BD_DIR="$WORK/BreastDividerDataset"
DATASET_DIR="$WORK/Dataset930_BreastDivider2D"

echo "=== Extracting 2D slices ==="
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

count = 0
THRESHOLD = 64

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
        if img_data.ndim == 3:
            mid = img_data.shape[2] // 2
            img_2d = img_data[:, :, mid].astype(np.float32)
            lbl_2d = lbl_data[:, :, mid].astype(np.uint8)
        else:
            img_2d = img_data.astype(np.float32)
            lbl_2d = lbl_data.astype(np.uint8)
        if min(img_2d.shape[:2]) < THRESHOLD:
            continue
        lbl_2d = (lbl_2d > 0).astype(np.uint8)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img_2d[:, :, np.newaxis], affine), str(images_dir / f'{stem}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(lbl_2d[:, :, np.newaxis], affine), str(labels_dir / f'{stem}.nii.gz'))
        count += 1
        if count % 1000 == 0:
            print(f'  {count} cases...')

ds = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'breast': 1}, 'numTraining': count, 'file_ending': '.nii.gz'}
with open(out_dir / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=2)
print(f'Done: {count} cases')
"
echo "=== Extraction complete ==="
