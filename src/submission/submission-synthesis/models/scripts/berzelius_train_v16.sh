#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J mamasynth_v16
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v16: v14 config but with ensemble breast mask (ResEncUNetL fold_0 OR PlainConvUNet fold_4)
# Quick test: 50 epochs only (niter=25 + niter_decay=25)
# Data: data_split_v4 axial-only (2528 cases)
# Breast mask: ensemble of two models (union)
# Intensity aug: yes

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

# Use the PlainConvUNet 2D fold_4 model for breast masks

# Create filtered dataset (axial only, exclude ISPY1/NACT/AMBL)
FILTERED=$PROJ/data_split_v4_axial/train/mha
if [ ! -d "$FILTERED/input" ]; then
    mkdir -p $FILTERED/input $FILTERED/ground_truth $FILTERED/mask
    for sub in input ground_truth mask; do
        src=$PROJ/data_split_v4/train/mha/$sub
        if [ -d "$src" ]; then
            for f in $src/*.mha; do
                name=$(basename "$f")
                case "$name" in
                    ISPY1_*|NACT_*|AMBL-*) continue ;;
                    *) ln -s "$f" "$FILTERED/$sub/$name" ;;
                esac
            done
        fi
    done
    echo "Filtered dataset (axial only): $(ls $FILTERED/input/ | wc -l) cases"
fi

# Generate breast masks with ensemble (OR of two models)
BREAST_MASK_DIR=$PROJ/data_split_v4_axial/train/mha/breast_mask_ensemble
if [ ! -d "$BREAST_MASK_DIR" ]; then
    echo "Generating ensemble breast masks..."
    # Model 1: ResEncUNetL fold_0 (checkpoint_best)
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $FILTERED/input \
        --output_dir ${BREAST_MASK_DIR}_resenc \
        --model_dir $PROJ/weights/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetResEncUNetLPlans__2d \
        --fold 0 \
        --checkpoint checkpoint_best.pth
    # Model 2: PlainConvUNet fold_4 (checkpoint_final)
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $FILTERED/input \
        --output_dir ${BREAST_MASK_DIR}_plain \
        --model_dir $PROJ/weights/breast_seg/nnUNetTrainer__nnUNetPlans__2d \
        --fold 4 \
        --checkpoint checkpoint_final.pth
    # Combine with OR
    mkdir -p $BREAST_MASK_DIR
    python -c "
import os, numpy as np, SimpleITK as sitk
from pathlib import Path
d1 = Path('${BREAST_MASK_DIR}_resenc')
d2 = Path('${BREAST_MASK_DIR}_plain')
out = Path('$BREAST_MASK_DIR')
for f in sorted(d1.glob('*.mha')):
    m1 = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
    f2 = d2 / f.name
    if f2.exists():
        m2 = sitk.GetArrayFromImage(sitk.ReadImage(str(f2)))
        combined = ((m1 > 0) | (m2 > 0)).astype(np.int16)
    else:
        combined = (m1 > 0).astype(np.int16)
    img = sitk.GetImageFromArray(combined)
    img.CopyInformation(sitk.ReadImage(str(f)))
    sitk.WriteImage(img, str(out / f.name))
print(f'Ensemble masks: {len(list(out.glob(\"*.mha\")))} files')
"
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v16 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $PROJ/data_split_v4_axial/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 9 \
  --norm instance \
  --batchSize 8 \
  --niter 25 \
  --niter_decay 25 \
  --lr 0.0002 \
  --lambda_feat 10 \
  --lambda_gan 1.0 \
  --tumor_weight 10 \
  --lambda_mssc 50 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --lambda_vgg 10 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $BREAST_MASK_DIR \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0
