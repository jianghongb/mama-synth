#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J resrefiner_v19
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v19: Residual Refiner — simple UNet predicts Δ, output = pre + Δ
# Supervisor suggestion: predict residual directly, add pre back
# Base GAN: v14 (best for SDEdit), single-step refinement (no diffusion)
# Faster training + faster inference (1 forward pass, no iterative denoising)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

GAN_WEIGHTS=$PROJ/checkpoints/mamasynth_v14/latest_net_G.pth
if [ ! -f "$GAN_WEIGHTS" ]; then
    echo "ERROR: v14 GAN weights not found"
    exit 1
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train_residual_refiner.py \
    --dataroot $PROJ/data_split_v4/train \
    --gan_weights $GAN_WEIGHTS \
    --checkpoints_dir $PROJ/checkpoints \
    --name resrefiner_v19 \
    --epochs 100 \
    --batch_size 8 \
    --lr 2e-4 \
    --image_size 512 \
    --save_freq 10 \
    --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/resrefiner_v19/refiner_latest.pth"
