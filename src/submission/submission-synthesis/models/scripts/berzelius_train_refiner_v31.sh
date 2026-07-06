#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -J refiner_v31
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# SDEdit Refiner on v31_pinorm
# ============================
# Stage 2 diffusion refiner trained on top of frozen v31 GAN.
#
# Key difference from v17 refiner:
# - Base GAN uses per-image normalization (v31_pinorm)
# - GAN inference includes per-image normalize + de-normalize
# - Data: data_multislice_v3 (peak GT, more diverse)
#
# Pipeline (training):
#   1. pre → per-image normalize → frozen v31 GAN → de-normalize → gan_out
#   2. Sample t, add noise to gt: noisy = √ᾱt·gt + √(1-ᾱt)·ε
#   3. RefinerUNet([noisy, pre, gan_out], t) → predict ε
#   4. Loss = MSE(ε̂, ε)
#
# Pipeline (inference):
#   1. pre → v31 GAN (with per-image norm) → gan_out
#   2. gan_out + 30% noise → DDIM 20 steps → refined output
#
# Expected: LPIPS ↓ (sharper tumor boundaries), Dice ↑ (better segmentation)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

# v31 GAN weights (per-image norm)
GAN_WEIGHTS=$PROJ/checkpoints/mamasynth_v31_pinorm/latest_net_G.pth
if [ ! -f "$GAN_WEIGHTS" ]; then
    echo "ERROR: v31 GAN weights not found at $GAN_WEIGHTS"
    exit 1
fi

DATAROOT=$PROJ/data_multislice_v3/train
BREAST_MASK_DIR=$PROJ/data_multislice_v3/train/mha/breast_mask

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

echo "=== Training SDEdit Refiner on v31_pinorm ==="
echo "GAN weights: $GAN_WEIGHTS"
echo "Data: $DATAROOT"
echo ""

python train_refiner_v31.py \
    --dataroot $DATAROOT \
    --gan_weights $GAN_WEIGHTS \
    --breast_mask_dir $BREAST_MASK_DIR \
    --checkpoints_dir $PROJ/checkpoints \
    --name refiner_v31_pinorm \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --T 1000 \
    --image_size 512 \
    --save_freq 10 \
    --gpu_ids 0

echo ""
echo "Done! Refiner weights: $PROJ/checkpoints/refiner_v31_pinorm/refiner_latest.pth"
echo ""
echo "=== Inference ==="
echo "python infer_refiner_v31.py \\"
echo "  --gan_weights $GAN_WEIGHTS \\"
echo "  --refiner_weights $PROJ/checkpoints/refiner_v31_pinorm/refiner_latest.pth \\"
echo "  --input_dir $PROJ/data_multislice_v3/test/mha/input \\"
echo "  --breast_mask_dir $PROJ/data_multislice_v3/test/mha/breast_mask \\"
echo "  --output_dir $PROJ/predictions_v31_refined"
