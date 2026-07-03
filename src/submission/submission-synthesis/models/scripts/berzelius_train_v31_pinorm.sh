#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v31_pinorm
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v31: Per-Image Z-Score Normalization
# ====================================
# Instead of global z-score (same mean/std for all images), each image is
# independently normalized using its own breast-region mean/std.
#
# Benefits:
# - Eliminates cross-scanner intensity variation
# - Prevents outlier cases (like DUKE_055) from having extreme values
# - Model learns in canonical space: all inputs have mean≈0, std≈1
#
# At inference:
#   input_gc → per-image normalize → model → de-normalize back to GC space
#
# Base: v22 architecture (ngf=64, n_blocks=12, batch=16, noise_aug, lr=0.0003)
# Data: data_multislice_v2/train (with breast mask)
# Key change: --dataset_mode mha_perimage_norm

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

echo "=== Starting v31 training (per-image z-score, ngf=64, n_blocks=12, batch=16) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v31_pinorm \
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

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v31_pinorm/latest_net_G.pth"
echo ""
echo "=== Inference: remember to use per-image norm + de-norm ==="
echo "See: models/infer_perimage_norm.py"
