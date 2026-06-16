#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J mamasynth_v12
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v12: v9 params (MSEC=50 + breast mask) on data_split (1074 cases)
# Uses Dataset932 (4ch) for better breast segmentation

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

# Step 1: Generate breast masks using Dataset932 (4ch, better quality)
if [ ! -d "$PROJ/data_split/train/mha/breast_mask_932" ]; then
    echo "Generating breast masks with Dataset932..."
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $PROJ/data_split/train/mha/input \
        --gt_dir $PROJ/data_split/train/mha/ground_truth \
        --output_dir $PROJ/data_split/train/mha/breast_mask_932 \
        --model_dir $PROJ/weights/breast_seg/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --model_type 932
fi

# Step 2: Train
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v12 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $PROJ/data_split/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 --input_nc 1 --output_nc 1 \
  --no_instance --residual_mode \
  --resize_or_crop resize --loadSize 512 --fineSize 512 \
  --n_downsample_global 4 --ngf 64 --n_blocks_global 9 --norm instance \
  --batchSize 8 --niter 100 --niter_decay 100 --lr 0.0002 \
  --lambda_feat 10 --lambda_gan 1.0 --lambda_vgg 10 \
  --lambda_ssim 0 --lambda_mssc 50 --mssc_levels 3 \
  --tumor_weight 10 --num_D 2 --n_layers_D 3 \
  --breast_mask_dir $PROJ/data_split/train/mha/breast_mask_932 \
  --save_epoch_freq 5 --print_freq 100 --gpu_ids 0
