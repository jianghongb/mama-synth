#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J refiner_v18
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v18: Stage 2 — SDEdit diffusion refiner on top of frozen v16 GAN
# v16 base is stronger than v14 (Dice 0.722 vs 0.703), expect v18 > v17
# Data: data_split_v4_axial (same as v16 training data)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

# v16 GAN weights
GAN_WEIGHTS=$PROJ/checkpoints/mamasynth_v16/latest_net_G.pth
if [ ! -f "$GAN_WEIGHTS" ]; then
    echo "ERROR: v16 GAN weights not found at $GAN_WEIGHTS"
    exit 1
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train_refiner.py \
    --dataroot $PROJ/data_split_v4_axial/train \
    --gan_weights $GAN_WEIGHTS \
    --checkpoints_dir $PROJ/checkpoints \
    --name refiner_v18 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --T 1000 \
    --image_size 512 \
    --save_freq 10 \
    --gpu_ids 0

echo "Done! Refiner weights: $PROJ/checkpoints/refiner_v18/refiner_latest.pth"
