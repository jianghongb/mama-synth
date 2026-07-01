#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_train
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 3: nnUNet preprocess + train (pure GPU operations)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

echo "Preprocessing Dataset920..."
nnUNetv2_plan_and_preprocess -d 920 -c 2d --verify_dataset_integrity

echo "Training fold 0..."
nnUNet_compile=0 nnUNetv2_train 920 2d 0 --npz

echo "Done! Model at: $nnUNet_results/Dataset920_BreastSeg2D/"
