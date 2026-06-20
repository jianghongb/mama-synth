#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_step3
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_step3_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_step3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 3: Train nnUNet 2D on distilled Dataset920 (GPU required)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

# === Extract 2D slices (fast, ~10min) then immediately start GPU training ===
echo "Extracting 2D slices from 3D predictions..."
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
print(f'Extracted {count} 2D slices.')
"

echo "Preprocessing..."
nnUNetv2_plan_and_preprocess -d 920 -c 2d --verify_dataset_integrity

echo "Training fold 0..."
nnUNetv2_train 920 2d 0 --npz

echo "Done! Model at: $nnUNet_results/Dataset920_BreastSeg2D/"
