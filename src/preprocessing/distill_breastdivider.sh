#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_breastdivider
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_bd_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_bd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Distill BreastDivider 3D model → 2D nnUNet for breast segmentation
# Teacher: ykirchhoff/BreastDividerModel (3D, multi-modality)
# Student: Dataset930 (2D nnUNet, single-channel T1)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

BD_MODEL=$PROJ/models/BreastDividerModel
PSEUDO_3D=$PROJ/pseudo_labels_breastdivider
DATASET_ID=930

# ============================================================
# Step 1: Download BreastDivider model
# ============================================================
if [ ! -d "$BD_MODEL" ]; then
    echo "=== Downloading BreastDivider model ==="
    huggingface-cli download ykirchhoff/BreastDividerModel --local-dir $BD_MODEL
fi

# ============================================================
# Step 2: Prepare 3D input (nnUNet format: patient_0000.nii.gz)
# ============================================================
INPUT_3D=$PROJ/breastdivider_input
if [ ! -d "$INPUT_3D" ] || [ $(ls "$INPUT_3D"/*.nii.gz 2>/dev/null | wc -l) -lt 10 ]; then
    echo "=== Preparing 3D input ==="
    mkdir -p $INPUT_3D
    python -c "
import os
from pathlib import Path

# Use pre-contrast volumes from images dir (phase 0)
img_dir = Path('$PROJ/images')
out_dir = Path('$INPUT_3D')

for patient_dir in sorted(img_dir.iterdir()):
    if not patient_dir.is_dir():
        continue
    pid = patient_dir.name
    # Find phase 0 (pre-contrast)
    phase0 = patient_dir / f'{pid}_0000.nii.gz'
    if not phase0.exists():
        phase0 = patient_dir / f'{pid}_0.nii.gz'
    if not phase0.exists():
        # Try first file
        files = sorted(patient_dir.glob('*.nii.gz'))
        if files:
            phase0 = files[0]
    if phase0.exists():
        dst = out_dir / f'{pid}_0000.nii.gz'
        if not dst.exists():
            os.symlink(phase0, dst)

print(f'Prepared {len(list(out_dir.glob(\"*.nii.gz\")))} input files')
"
fi

# ============================================================
# Step 3: Run BreastDivider 3D prediction
# ============================================================
if [ ! -d "$PSEUDO_3D" ] || [ $(ls "$PSEUDO_3D"/*.nii.gz 2>/dev/null | wc -l) -lt 10 ]; then
    echo "=== Running BreastDivider 3D prediction ==="
    mkdir -p $PSEUDO_3D
    nnUNetv2_predict_from_modelfolder \
        -i $INPUT_3D \
        -o $PSEUDO_3D \
        -m $BD_MODEL \
        --disable_tta
fi
echo "Pseudo labels: $(ls $PSEUDO_3D/*.nii.gz | wc -l) files"

# ============================================================
# Step 4: Extract 2D slices for nnUNet training
# ============================================================
DATASET_DIR=$nnUNet_raw/Dataset${DATASET_ID}_BreastDivider2D
if [ ! -d "$DATASET_DIR/imagesTr" ]; then
    echo "=== Extracting 2D slices ==="
    python -c "
import json, csv, numpy as np, nibabel as nib, SimpleITK as sitk
from pathlib import Path

pseudo_dir = Path('$PSEUDO_3D')
data_dir = Path('$PROJ/data_split_v4/train/mha')
dataset_dir = Path('$DATASET_DIR')
images_dir = dataset_dir / 'imagesTr'
labels_dir = dataset_dir / 'labelsTr'
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# For each 2D case in data_split_v4, find corresponding 3D pseudo label
# and extract the same slice
input_dir = data_dir / 'input'
count = 0
for mha_file in sorted(input_dir.glob('*.mha')):
    case_id = mha_file.stem
    # Find 3D pseudo label
    pseudo_file = pseudo_dir / f'{case_id}.nii.gz'
    if not pseudo_file.exists():
        # Try without prefix
        for p in pseudo_dir.glob(f'{case_id}*.nii.gz'):
            pseudo_file = p
            break
    if not pseudo_file.exists():
        continue

    # Load 2D input to get shape info
    img_2d = sitk.GetArrayFromImage(sitk.ReadImage(str(mha_file))).squeeze()

    # Load 3D pseudo label
    label_3d = nib.load(str(pseudo_file)).get_fdata()

    # Take middle slice (same axis as preprocess)
    # Since our 2D data is already extracted, we use the 3D label's middle slice
    # that best matches the 2D image size
    for axis in range(3):
        sl_shape = tuple(s for i, s in enumerate(label_3d.shape) if i != axis)
        if sl_shape == img_2d.shape:
            mid = label_3d.shape[axis] // 2
            label_2d = np.take(label_3d, mid, axis=axis)
            break
    else:
        # Fallback: take middle of last axis
        label_2d = label_3d[:, :, label_3d.shape[2]//2]

    # Binarize: any label > 0 = breast
    breast_mask = (label_2d > 0).astype(np.int16)

    # Save as nifti for nnUNet
    # Image
    img_nii = nib.Nifti1Image(img_2d[np.newaxis, :, :].astype(np.float32), np.eye(4))
    nib.save(img_nii, str(images_dir / f'{case_id}_0000.nii.gz'))
    # Label
    lbl_nii = nib.Nifti1Image(breast_mask[np.newaxis, :, :].astype(np.int16), np.eye(4))
    nib.save(lbl_nii, str(labels_dir / f'{case_id}.nii.gz'))
    count += 1

# Write dataset.json
dataset_json = {
    'channel_names': {'0': 'T1'},
    'labels': {'background': 0, 'breast': 1},
    'numTraining': count,
    'file_ending': '.nii.gz'
}
with open(dataset_dir / 'dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=2)

print(f'Dataset{${DATASET_ID}}: {count} training cases')
"
fi

# ============================================================
# Step 5: Train 2D nnUNet
# ============================================================
echo "=== Training 2D nnUNet (Dataset${DATASET_ID}) ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 2d --verify_dataset_integrity
nnUNet_compile=0 nnUNetv2_train $DATASET_ID 2d 0 --npz

echo "=== Done! ==="
echo "Model: $nnUNet_results/Dataset${DATASET_ID}_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
