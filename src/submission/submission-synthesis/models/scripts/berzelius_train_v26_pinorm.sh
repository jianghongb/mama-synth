#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v26_pinorm
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v26_pinorm: Wider network (ngf=96, n_blocks=12) + per-image z-score norm.
# Combines v26's lower MSE (larger network) with v31's better LPIPS/SSIM/Dice.
# Data: data_multislice_v3/train (~9504 samples, with AMBL).

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

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

echo "=== Starting v26_pinorm (ngf=96 + per-image norm) ==="
echo "Dataset: $PROJ/data_multislice_v3/train"
echo "Samples: $(ls $PROJ/data_multislice_v3/train/mha/input/*.mha 2>/dev/null | wc -l)"

python train.py \
  --name mamasynth_v26_pinorm \
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
  --ngf 96 \
  --n_blocks_global 12 \
  --norm instance \
  --batchSize 8 \
  --nThreads 16 \
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
  --breast_mask_dir $PROJ/data_multislice_v3/train/mha/breast_mask \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v26_pinorm/latest_net_G.pth"
echo "Inference: use infer_perimage_norm.py with --ngf 96"
