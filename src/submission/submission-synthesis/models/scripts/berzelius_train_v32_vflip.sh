#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v32_vflip
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v32: Per-Image Norm + Vertical Flip Augmentation
# =================================================
# Base: v31_pinorm (best model so far)
# Change: add --vflip for random vertical flip (50% probability, independent of hflip)
#
# Motivation (from TTA experiments on v31_pinorm):
#   - hflip TTA (2x): SSIM_tumor 0.379→0.495, Dice 0.371→0.558
#   - 4flip TTA (4x): SSIM_tumor 0.518, FRD 22.99 (adds vflip+hvflip)
#   - Training already has hflip, but no vflip
#   - Adding vflip should help the model learn vertical invariance natively
#     without needing TTA at inference time
#
# Architecture: same as v31 (ngf=64, n_blocks=12, residual_mode)
# Data: data_multislice_v3/train (per-image norm + breast mask)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

cd $PROJ/mama-synth
git pull origin dev

MASK_OUTPUT=$PROJ/data_multislice_v3/train/mha/breast_mask

echo "=== Starting v32 training (v31_pinorm + vflip augmentation) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v32_vflip \
  --model pix2pixHD \
  --dataset_mode mha_perimage_norm \
  --dataroot $PROJ/data_multislice_v3/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --noise_aug \
  --vflip \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 12 \
  --norm instance \
  --batchSize 16 \
  --nThreads 16 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0003 \
  --lambda_feat 10 \
  --lambda_gan 1.0 \
  --tumor_weight 10 \
  --lambda_mssc 50 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --lambda_vgg 10 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $MASK_OUTPUT \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v32_vflip/latest_net_G.pth"
