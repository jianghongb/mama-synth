#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J refiner_v17
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v17: Stage 2 — SDEdit diffusion refiner on top of frozen v14 GAN
# Input: [noisy_gt, pre, gan_output] (3ch) → predict noise ε
# GAN: mamasynth_v14 (data_split_v4, breast mask, intensity_aug)
# Data: data_split_v4/train (same as v14)
# Model: RefinerUNet ~18M params, DDPM training, cosine schedule

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

# Ensure v14 GAN weights exist
GAN_WEIGHTS=$PROJ/checkpoints/mamasynth_v14/latest_net_G.pth
if [ ! -f "$GAN_WEIGHTS" ]; then
    echo "ERROR: v14 GAN weights not found at $GAN_WEIGHTS"
    echo "Please wait for v14 training to finish first."
    exit 1
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train_refiner.py \
    --dataroot $PROJ/data_split_v4/train \
    --gan_weights $GAN_WEIGHTS \
    --checkpoints_dir $PROJ/checkpoints \
    --name refiner_v17 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --T 1000 \
    --image_size 512 \
    --save_freq 10 \
    --gpu_ids 0

echo "Done! Refiner weights: $PROJ/checkpoints/refiner_v17/refiner_latest.pth"
