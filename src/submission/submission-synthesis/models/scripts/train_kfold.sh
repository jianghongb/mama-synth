#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J kfold_%a
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/kfold_%A_%a.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/kfold_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#SBATCH --array=0-4
#
# K-fold cross-validation training (v14 config)
# Submit with: sbatch train_kfold.sh
# Automatically runs 5 jobs (fold 0-4) via SLURM array

FOLD=${SLURM_ARRAY_TASK_ID}
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

# Paths
SPLITS_CSV=$PROJ/mama-synth/kfold_splits.csv
ALL_DATA=$PROJ/data_split_v4/train/mha
FOLD_DIR=$PROJ/kfold/fold_${FOLD}

# Create fold-specific train/test directories (symlinks)
echo "=== Setting up fold $FOLD ==="
python $PROJ/mama-synth/src/preprocessing/setup_kfold_dir.py \
    --splits_csv $SPLITS_CSV \
    --data_dir $ALL_DATA \
    --output_dir $FOLD_DIR \
    --fold $FOLD

echo "Train: $(ls $FOLD_DIR/train/mha/input/ | wc -l) cases"
echo "Test:  $(ls $FOLD_DIR/test/mha/input/ | wc -l) cases"

# Train (v14 config)
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name kfold_f${FOLD} \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $FOLD_DIR/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 9 \
  --norm instance \
  --batchSize 8 \
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
  --breast_mask_dir $FOLD_DIR/train/mha/breast_mask \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo "=== Fold $FOLD training done ==="
echo "Weights: $PROJ/checkpoints/kfold_f${FOLD}/latest_net_G.pth"
