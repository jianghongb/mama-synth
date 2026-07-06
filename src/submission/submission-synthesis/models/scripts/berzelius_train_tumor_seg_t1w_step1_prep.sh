#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH -t 2:00:00
#SBATCH -J tumor_seg_prep
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_prep_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_prep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 1 (CPU only): Convert MHA → nnUNet NIfTI format
# Multi-threaded for speed. No GPU needed.
#
# After this completes, submit berzelius_train_tumor_seg_t1w_step2.sh for GPU training.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

DATASET_ID=940
DATASET_NAME="Dataset${DATASET_ID}_TumorSegT1w"
DATASET_DIR=$nnUNet_raw/$DATASET_NAME

INPUT_DIR=$PROJ/data_multislice_v3/train/mha/input
MASK_DIR=$PROJ/data_multislice_v3/train/mha/mask

echo "=== Step 1: Convert MHA → nnUNet format (32 threads) ==="

python -c "
import json, os, sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

INPUT_DIR = Path('$INPUT_DIR')
MASK_DIR = Path('$MASK_DIR')
DATASET_DIR = Path('$DATASET_DIR')
IMAGES_DIR = DATASET_DIR / 'imagesTr'
LABELS_DIR = DATASET_DIR / 'labelsTr'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

input_files = sorted(INPUT_DIR.glob('*.mha'))
print(f'Total input files: {len(input_files)}')

def process_one(f):
    mask_path = MASK_DIR / f.name
    if not mask_path.exists():
        return None, 'no_mask'

    # Case ID: sanitize filename
    case_id = f.stem.replace('.', '_').replace('+', 'p').replace('-', 'm')

    # Skip if already done
    out_img_path = IMAGES_DIR / f'{case_id}_0000.nii.gz'
    out_lbl_path = LABELS_DIR / f'{case_id}.nii.gz'
    if out_img_path.exists() and out_lbl_path.exists():
        return case_id, 'exists'

    # Read mask, skip if empty
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img).squeeze()
    if mask_arr.sum() < 10:
        return None, 'empty'

    # Read input
    input_img = sitk.ReadImage(str(f))
    input_arr = sitk.GetArrayFromImage(input_img).astype(np.float32).squeeze()

    # Save as NIfTI (shape: 1, H, W for 2D nnUNet)
    out_img = sitk.GetImageFromArray(input_arr[np.newaxis, :, :])
    sitk.WriteImage(out_img, str(out_img_path))

    mask_out = mask_arr.astype(np.uint8)[np.newaxis, :, :]
    out_mask = sitk.GetImageFromArray(mask_out)
    sitk.WriteImage(out_mask, str(out_lbl_path))

    return case_id, 'ok'

# Multi-threaded processing
count = 0
skipped_empty = 0
skipped_nomask = 0
already_done = 0

with ThreadPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(process_one, f): f for f in input_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc='Converting'):
        case_id, status = future.result()
        if status == 'ok':
            count += 1
        elif status == 'empty':
            skipped_empty += 1
        elif status == 'no_mask':
            skipped_nomask += 1
        elif status == 'exists':
            already_done += 1
            count += 1

total_cases = count
print(f'')
print(f'Done: {count} cases ({already_done} already existed)')
print(f'Skipped: {skipped_empty} empty masks, {skipped_nomask} no mask file')

# Write dataset.json
dataset_json = {
    'channel_names': {'0': 'T1w_pre_contrast'},
    'labels': {'background': 0, 'tumor': 1},
    'numTraining': total_cases,
    'file_ending': '.nii.gz',
}
with open(DATASET_DIR / 'dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f'dataset.json: numTraining={total_cases}')
print(f'Output: {DATASET_DIR}')
"

echo ""
echo "=== Step 1 complete ==="
echo "Now submit GPU training job:"
echo "  sbatch src/submission/submission-synthesis/models/scripts/berzelius_train_tumor_seg_t1w_step2.sh"
