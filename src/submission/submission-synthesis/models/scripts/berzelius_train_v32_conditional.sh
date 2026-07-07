#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v32_cond
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v32: Conditional GAN with T1w Tumor Segmentation (Supervisor Proposal)
# =====================================================================
# Input: 3 channels = pre-contrast + breast_mask + predicted_tumor_mask
# The model EXPLICITLY knows where the tumor is → focuses enhancement there.
#
# Key differences from v31:
# - input_nc=3 (vs 1)
# - Uses predicted tumor mask from Dataset940 (not GT)
# - Per-image z-score normalization (from v31)
#
# Architecture: GlobalGenerator, ngf=64, n_blocks=12 (same as v22/v31)
# Data: data_multislice_v3/train
# Inference: BreastDivider → TumorSeg → GAN(3ch) → breast_mask composite

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics
export TORCHDYNAMO_DISABLE=1

cd $PROJ/mama-synth
git pull origin dev

DATAROOT=$PROJ/data_multislice_v3/train
BREAST_MASK=$PROJ/data_multislice_v3/train/mha/breast_mask
TUMOR_MASK=$PROJ/data_multislice_v3/train/mha/predicted_tumor

# Verify predicted tumor masks exist
NUM_TUMOR=$(ls $TUMOR_MASK/*.mha 2>/dev/null | wc -l)
if [ "$NUM_TUMOR" -lt 100 ]; then
    echo "ERROR: Only $NUM_TUMOR predicted tumor masks. Run berzelius_tumor_seg_infer.sh first!"
    exit 1
fi
echo "=== v32: Conditional GAN (pre + breast + tumor) ==="
echo "Predicted tumor masks: $NUM_TUMOR"
echo ""

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v32_conditional \
  --model pix2pixHD \
  --dataset_mode mha_perimage_norm \
  --dataroot $DATAROOT \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 3 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --noise_aug \
  --mask_as_input \
  --tumor_mask_as_input \
  --tumor_mask_dir $TUMOR_MASK \
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
  --breast_mask_dir $BREAST_MASK \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo ""
echo "Done! Weights: $PROJ/checkpoints/mamasynth_v32_conditional/latest_net_G.pth"
echo ""
echo "=== Inference pipeline ==="
echo "1. BreastDivider → breast_mask"
echo "2. Dataset940 TumorSeg → predicted_tumor_mask"
echo "3. GAN(concat(pre, breast_mask, tumor_mask)) → synthetic"
echo "4. output = breast_mask × synthetic + (1-breast_mask) × pre"
