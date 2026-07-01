#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J mamasynth_v18
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v18: v14 config + Dataset920 breast mask (distilled from 932)
# Data: data_split_v4 axial-only (2528 cases)
# Breast mask: Dataset920 (PlainConvUNet 2D, distilled from 932 4ch 3D)
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

# Create filtered dataset (axial only)
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
fi

# Generate breast masks with Dataset920
BREAST_MASK_DIR=$PROJ/data_split_v4_axial/train/mha/breast_mask_920
if [ ! -d "$BREAST_MASK_DIR" ]; then
    echo "Generating breast masks with Dataset920..."
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $FILTERED/input \
        --output_dir $BREAST_MASK_DIR \
        --model_dir $PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d \
        --fold 0 \
        --checkpoint checkpoint_best.pth
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v18 \
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
  --niter 100 \
  --niter_decay 100 \
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
