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

echo "=== Extracting 2D slices (T1w only) + deleting non-T1w files ==="
python -c "
import nibabel as nib
import numpy as np
import json, os, csv
from pathlib import Path

bd_dir = Path('$BD_DIR')
out_dir = Path('$DATASET_DIR')
images_dir = out_dir / 'imagesTr'
labels_dir = out_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Load T1w case IDs
t1w_ids = set()
id_map = bd_dir / 'breastdivider_id_mapping.csv'
if id_map.exists():
    t1w_keywords = ['t1', 'vibrant', 'dyn', 'flash', 'thrive', 'pre', 'ax-dyn']
    t2w_keywords = ['t2', 'stir', 'tirm']
    dwi_keywords = ['dwi', 'diff', 'adc', 'b800', 'b1000']
    with open(id_map) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            bd_id, orig_id = row[0], row[1].lower()
            is_t1 = any(k in orig_id for k in t1w_keywords)
            is_t2 = any(k in orig_id for k in t2w_keywords)
            is_dwi = any(k in orig_id for k in dwi_keywords)
            if is_t1 and not is_t2 and not is_dwi:
                t1w_ids.add(bd_id)
    print(f'Loaded {len(t1w_ids)} T1w case IDs')
else:
    print('WARNING: No id_mapping found, using all cases')

count = 0
deleted = 0
THRESHOLD = 64

for batch in ['imagesTr_batch1', 'imagesTr_batch2']:
    img_batch = bd_dir / batch
    lbl_batch = bd_dir / batch.replace('imagesTr', 'labelsTr')
    if not img_batch.exists():
        continue
    for img_f in sorted(img_batch.glob('*_0000.nii.gz')):
        stem = img_f.name.replace('_0000.nii.gz', '')
        lbl_f = lbl_batch / f'{stem}.nii.gz'

        # Delete non-T1w files to free space
        if t1w_ids and stem not in t1w_ids:
            os.remove(img_f)
            if lbl_f.exists():
                os.remove(lbl_f)
            deleted += 1
            continue

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
            os.remove(img_f)
            os.remove(lbl_f)
            deleted += 1
            continue
        lbl_2d = (lbl_2d > 0).astype(np.uint8)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img_2d[:, :, np.newaxis], affine), str(images_dir / f'{stem}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(lbl_2d[:, :, np.newaxis], affine), str(labels_dir / f'{stem}.nii.gz'))
        count += 1
        if count % 1000 == 0:
            print(f'  {count} extracted, {deleted} deleted...')

        # Delete source after extraction to free space
        os.remove(img_f)
        os.remove(lbl_f)
        deleted += 1

ds = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'breast': 1}, 'numTraining': count, 'file_ending': '.nii.gz'}
with open(out_dir / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=2)
print(f'Done: {count} T1w cases extracted, {deleted} files deleted')
"
echo "=== Extraction complete ==="
