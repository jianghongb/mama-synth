#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J distill_ext
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_ext_%A_%a.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_ext_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#SBATCH --array=0-3
#
# Parallel extraction: 4 jobs, each handles 1/4 of the files
set -e
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
WORK=$PROJ/distill_breastdivider
BD_DIR="$WORK/BreastDividerDataset"
DATASET_DIR="$WORK/Dataset930_BreastDivider2D"

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

python -c "
import nibabel as nib
import numpy as np
import json, os, csv, sys
from pathlib import Path

TASK_ID = $SLURM_ARRAY_TASK_ID
N_TASKS = 4

bd_dir = Path('$BD_DIR')
out_dir = Path('$DATASET_DIR')
images_dir = out_dir / 'imagesTr'
labels_dir = out_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Load T1w IDs
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
            if any(k in orig_id for k in t1w_keywords) and not any(k in orig_id for k in t2w_keywords) and not any(k in orig_id for k in dwi_keywords):
                t1w_ids.add(bd_id)

# Collect all files
all_files = []
for batch in ['imagesTr_batch1', 'imagesTr_batch2']:
    img_batch = bd_dir / batch
    if img_batch.exists():
        all_files.extend(sorted(img_batch.glob('*_0000.nii.gz')))

# Split for this task
my_files = [f for i, f in enumerate(all_files) if i % N_TASKS == TASK_ID]
print(f'Task {TASK_ID}: processing {len(my_files)}/{len(all_files)} files', flush=True)

count = 0
deleted = 0
THRESHOLD = 64

for img_f in my_files:
    stem = img_f.name.replace('_0000.nii.gz', '')
    batch = img_f.parent.name
    lbl_f = bd_dir / batch.replace('imagesTr', 'labelsTr') / f'{stem}.nii.gz'

    if t1w_ids and stem not in t1w_ids:
        os.remove(img_f)
        if lbl_f.exists(): os.remove(lbl_f)
        deleted += 1
        continue

    if not lbl_f.exists():
        continue

    img = nib.load(str(img_f))
    img_data = img.get_fdata()
    lbl_data = nib.load(str(lbl_f)).get_fdata()

    if img_data.ndim == 3:
        mid = img_data.shape[2] // 2
        img_2d = img_data[:, :, mid].astype(np.float32)
        lbl_2d = lbl_data[:, :, mid].astype(np.uint8)
    else:
        img_2d = img_data.astype(np.float32)
        lbl_2d = lbl_data.astype(np.uint8)

    if min(img_2d.shape[:2]) < THRESHOLD:
        os.remove(img_f)
        if lbl_f.exists(): os.remove(lbl_f)
        deleted += 1
        continue

    lbl_2d = (lbl_2d > 0).astype(np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img_2d[:, :, np.newaxis], affine), str(images_dir / f'{stem}_0000.nii.gz'))
    nib.save(nib.Nifti1Image(lbl_2d[:, :, np.newaxis], affine), str(labels_dir / f'{stem}.nii.gz'))
    count += 1
    os.remove(img_f)
    os.remove(lbl_f)
    deleted += 1

    if count % 200 == 0:
        print(f'  Task {TASK_ID}: {count} extracted, {deleted} deleted', flush=True)

print(f'Task {TASK_ID} done: {count} extracted, {deleted} deleted', flush=True)
"
