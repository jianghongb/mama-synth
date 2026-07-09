#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v33_vflip_auroc
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v33: v32_vflip + v31_auroc loss settings
# ==========================================
# Combines the two best versions:
#   - v32_vflip: vflip augmentation → HD95=104.9 (best), LPIPS=0.117 (best)
#   - v31_auroc: tumor_weight=20, MSSC=100, no GT clip → Dice=0.565 (best)
#
# Expected: Dice ≈ 0.565 + HD95 ≈ 105 (both top performers)
#
# Architecture: same as v31 (ngf=64, n_blocks=12, residual_mode, per-image norm)
# Data: data_multislice_v3/train (breast mask)

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

echo "=== Starting v33 training (v32_vflip + v31_auroc: vflip + tumor=20 + mssc=100) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v33_vflip_auroc \
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
  --tumor_weight 20 \
  --lambda_mssc 100 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --lambda_vgg 10 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $MASK_OUTPUT \
  --save_epoch_freq 50 \
  --no_html \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v33_vflip_auroc/latest_net_G.pth"
echo "Inference: use infer_perimage_norm.py"
