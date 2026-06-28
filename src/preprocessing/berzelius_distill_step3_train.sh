#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J distill_train
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
set -e
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
WORK=$PROJ/distill_breastdivider

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2

export nnUNet_raw="$WORK/nnUNet_raw"
export nnUNet_preprocessed="$WORK/nnUNet_preprocessed"
export nnUNet_results="$WORK/nnUNet_results"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# Link dataset
ln -sfn $WORK/Dataset930_BreastDivider2D $nnUNet_raw/Dataset930_BreastDivider2D

echo "=== Preprocess ==="
nnUNetv2_plan_and_preprocess -d 930 -c 2d --verify_dataset_integrity

echo "=== Train fold 0 ==="
nnUNetv2_train 930 2d 0 --npz --c

echo "=== Done ==="
