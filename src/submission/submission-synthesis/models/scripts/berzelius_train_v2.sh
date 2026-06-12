#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J mamasynth_v2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Pix2PixHD v2 improved training for MAMA-SYNTH
# Key changes vs v1:
#   - Full 1074 cases (all resized to 512) instead of ~50 native-512 cases
#   - lambda_vgg 20 (was 10): improve LPIPS
#   - lambda_ssim 10 (new): directly optimize SSIM
#   - lambda_gan 0.2 (was 1.0): reduce blurriness
#   - tumor_weight 20 (was 10): improve Dice/SSIM_tumor
#   - lambda_mssc 10 (new): Multi-Scale Subtraction Consistency (Zhang 2025 thesis)

set -e

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

# Redirect conda/pip cache to proj (avoid home quota)
export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"

# Create env if not exists
ENV_DIR=$PROJ/envs/gan
if [ ! -d $ENV_DIR ]; then
  conda create -p $ENV_DIR python=3.10 -y
fi
conda activate $ENV_DIR

# Install deps if needed
pip show torch > /dev/null 2>&1 || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip show SimpleITK > /dev/null 2>&1 || pip install SimpleITK numpy Pillow dominate tqdm
pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

# Paths
WORK_DIR=${PROJ}/SimulatingDCE/synthesis/pix2pixHD
DATA_TRAIN=${PROJ}/data_split/train

cd ${WORK_DIR}

python train.py \
  --name mamasynth_v2_improved \
  --dataroot ${DATA_TRAIN} \
  --checkpoints_dir ${PROJ}/checkpoints \
  --dataset_mode mha \
  --input_nc 1 \
  --output_nc 1 \
  --label_nc 0 \
  --no_instance \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --netG global \
  --ngf 64 \
  --n_downsample_global 4 \
  --n_blocks_global 9 \
  --norm instance \
  --residual_mode \
  --batchSize 8 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0002 \
  --beta1 0.5 \
  --lambda_feat 10 \
  --lambda_vgg 20 \
  --lambda_ssim 10 \
  --lambda_gan 0.2 \
  --tumor_weight 20 \
  --lambda_mssc 10 \
  --mssc_levels 3 \
  --num_D 2 \
  --n_layers_D 3 \
  --nThreads 8 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  --display_freq 200 \
  --tf_log \
  --gpu_ids 0
