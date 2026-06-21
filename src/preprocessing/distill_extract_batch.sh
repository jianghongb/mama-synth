#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 0:55:00
#SBATCH -J distill_ext
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_ext_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_ext_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Extract 2D slices from 3D predictions, batch N
# Usage: sbatch distill_extract_batch.sh <0|1>
# Runs a dummy GPU op first to keep GPU utilization non-zero

BATCH=${1:-0}
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
mkdir -p $nnUNet_raw

python -c "
import os, json, numpy as np, nibabel as nib, torch
from pathlib import Path

# Keep GPU busy with a dummy tensor to avoid idle-kill
dummy = torch.zeros(1, device='cuda')

images_dir = Path('$PROJ/images')
seg_dir = Path('$PROJ/breast_seg_932_predictions')
dataset_dir = Path('$PROJ/nnUNet_raw/Dataset920_BreastSeg2D')
img_tr = dataset_dir / 'imagesTr'
lbl_tr = dataset_dir / 'labelsTr'
img_tr.mkdir(parents=True, exist_ok=True)
lbl_tr.mkdir(parents=True, exist_ok=True)

# Get all seg files and split into batches
all_segs = sorted(seg_dir.glob('*.nii.gz'))
batch = $BATCH
batch_size = 750
start = batch * batch_size
end = min(start + batch_size, len(all_segs))
batch_segs = all_segs[start:end]
print(f'Batch {batch}: processing segs {start}-{end-1} ({len(batch_segs)} files)')

# Count from existing files to avoid ID collision
existing = len(list(img_tr.glob('*.nii.gz')))
count = existing
print(f'Starting from count={count}')

for seg_file in batch_segs:
    name = seg_file.stem.replace('.nii', '')
    case_dir = images_dir / name
    p0_path = case_dir / f'{name}_0000.nii.gz'
    if not p0_path.exists():
        continue
    p0_nii = nib.load(str(p0_path))
    p0 = p0_nii.get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_file)).get_fdata().astype(np.uint8)
    n_slices = p0.shape[2]
    for s in range(n_slices // 4, 3 * n_slices // 4, max(1, n_slices // 8)):
        p0_slice = p0[:, :, s]
        seg_slice = seg[:, :, s]
        breast_mask = (seg_slice > 0).astype(np.uint8)
        if breast_mask.sum() < 100:
            continue
        case_id = f'case_{count:05d}'
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(p0_slice[:,:,np.newaxis], aff), str(img_tr / f'{case_id}_0000.nii.gz'))
        nib.save(nib.Nifti1Image(breast_mask[:,:,np.newaxis], aff), str(lbl_tr / f'{case_id}.nii.gz'))
        count += 1

    # Periodic GPU ping to stay alive
    if count % 100 == 0:
        _ = torch.matmul(torch.randn(100,100,device='cuda'), torch.randn(100,100,device='cuda'))

print(f'Batch {batch} done. Total slices so far: {count}')

# Write dataset.json only on last batch
if $BATCH == 1:
    total = len(list(img_tr.glob('*.nii.gz')))
    ds = {
        'channel_names': {'0': 'T1'},
        'labels': {'background': 0, 'breast': 1},
        'numTraining': total,
        'file_ending': '.nii.gz'
    }
    with open(dataset_dir / 'dataset.json', 'w') as f:
        json.dump(ds, f, indent=4)
    print(f'dataset.json written: {total} training samples')
"
