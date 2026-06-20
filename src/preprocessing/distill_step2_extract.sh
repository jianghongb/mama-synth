#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH -C cpu
#SBATCH -n 4
#SBATCH -t 2:00:00
#SBATCH -J distill_step2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_step2_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_step2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 2: Extract 2D axial slices from 3D predictions (CPU only)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
mkdir -p $nnUNet_raw

python -c "
import os, json, numpy as np, nibabel as nib
from pathlib import Path

images_dir = Path('$PROJ/images')
seg_dir = Path('$PROJ/breast_seg_932_predictions')
dataset_dir = Path('$PROJ/nnUNet_raw/Dataset920_BreastSeg2D')
img_tr = dataset_dir / 'imagesTr'
lbl_tr = dataset_dir / 'labelsTr'
img_tr.mkdir(parents=True, exist_ok=True)
lbl_tr.mkdir(parents=True, exist_ok=True)

count = 0
for seg_file in sorted(seg_dir.glob('*.nii.gz')):
    name = seg_file.stem.replace('.nii', '')
    case_dir = images_dir / name
    p0_path = case_dir / f'{name}_0000.nii.gz'
    if not p0_path.exists():
        continue

    p0_nii = nib.load(str(p0_path))
    p0 = p0_nii.get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_file)).get_fdata(dtype=np.uint8)

    # Extract multiple axial slices (not just mid) for more training data
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

ds = {
    'channel_names': {'0': 'T1'},
    'labels': {'background': 0, 'breast': 1},
    'numTraining': count,
    'file_ending': '.nii.gz'
}
with open(dataset_dir / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=4)

print(f'Step 2 done: {count} 2D training slices from {len(list(seg_dir.glob(\"*.nii.gz\")))} 3D cases.')
"
