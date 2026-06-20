#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_932_to_2d
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Distill Dataset932 (4ch 3D) breast segmentation into 2D nnUNet training data
# Step 1: Run 932 on all 3D cases → 3D breast labels
# Step 2: Extract axial mid-slices → 2D training pairs
# Step 3: Train nnUNet 2D (single channel T1, breast labels)

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
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

IMAGES_DIR=$PROJ/images
MODEL_932=$PROJ/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres
OUTPUT_3D=$PROJ/breast_seg_932_predictions
DATASET_DIR=$nnUNet_raw/Dataset920_BreastSeg2D

# ============ Step 1: Run 932 inference on all 3D cases ============
if [ ! -d "$OUTPUT_3D" ]; then
    echo "Step 1: Running Dataset932 3D inference..."
    mkdir -p $OUTPUT_3D

    python -c "
import os, numpy as np, nibabel as nib, torch
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

model_dir = '$MODEL_932'
images_dir = Path('$IMAGES_DIR')
output_dir = Path('$OUTPUT_3D')

device = torch.device('cuda')
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

cases = sorted([d for d in images_dir.iterdir() if d.is_dir()])
print(f'Processing {len(cases)} cases...')

for case_dir in tqdm(cases):
    name = case_dir.name
    out_path = output_dir / f'{name}.nii.gz'
    if out_path.exists():
        continue

    # Load P0 and P1
    p0_path = case_dir / f'{name}_0000.nii.gz'
    p1_path = case_dir / f'{name}_0001.nii.gz'
    if not p0_path.exists() or not p1_path.exists():
        continue

    p0_nii = nib.load(str(p0_path))
    p0 = p0_nii.get_fdata(dtype=np.float32)
    p1 = nib.load(str(p1_path)).get_fdata(dtype=np.float32)
    d_early = p1 - p0
    d_late = d_early  # approximate

    # 4ch input: (4, D, H, W)
    input_4ch = np.stack([p0, p1, d_early, d_late])

    props = {
        'sitk_stuff': {'spacing': tuple(p0_nii.header.get_zooms()[:3]),
                       'origin': (0.,0.,0.),
                       'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)},
        'spacing': list(p0_nii.header.get_zooms()[:3]),
    }

    pred = predictor.predict_single_npy_array(input_4ch, props, None, None, False)
    # Save as NIfTI
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), p0_nii.affine, p0_nii.header), str(out_path))

print('Step 1 done.')
"
fi

# ============ Step 2: Extract 2D slices for nnUNet training ============
if [ ! -d "$DATASET_DIR" ]; then
    echo "Step 2: Extracting 2D slices..."

    python -c "
import os, json, numpy as np, nibabel as nib
from pathlib import Path

images_dir = Path('$IMAGES_DIR')
seg_dir = Path('$OUTPUT_3D')
dataset_dir = Path('$DATASET_DIR')
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

    # Extract mid axial slice
    mid = p0.shape[2] // 2
    p0_slice = p0[:, :, mid]
    seg_slice = seg[:, :, mid]

    # Convert 932 labels (1=breast,2=FGT,3=tumor) → binary breast mask
    breast_mask = (seg_slice > 0).astype(np.uint8)

    # Skip if mask is empty
    if breast_mask.sum() < 100:
        continue

    # Save as 2D NIfTI (H, W, 1)
    case_id = f'case_{count:04d}'
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(p0_slice[:,:,np.newaxis], aff), str(img_tr / f'{case_id}_0000.nii.gz'))
    nib.save(nib.Nifti1Image(breast_mask[:,:,np.newaxis], aff), str(lbl_tr / f'{case_id}.nii.gz'))
    count += 1

# Write dataset.json
ds = {
    'channel_names': {'0': 'T1'},
    'labels': {'background': 0, 'breast': 1},
    'numTraining': count,
    'file_ending': '.nii.gz'
}
with open(dataset_dir / 'dataset.json', 'w') as f:
    json.dump(ds, f, indent=4)

print(f'Step 2 done: {count} 2D training slices.')
"
fi

# ============ Step 3: Train nnUNet 2D ============
echo "Step 3: Training nnUNet 2D..."
nnUNetv2_plan_and_preprocess -d 920 -c 2d --verify_dataset_integrity
nnUNetv2_train 920 2d 0 --npz

echo "Done! Model at: $nnUNet_results/Dataset920_BreastSeg2D/"
