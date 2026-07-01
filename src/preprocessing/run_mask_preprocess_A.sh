#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J mask_preprocess_A
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/mask_preprocess_A_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/mask_preprocess_A_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# 方案A: mask_and_preprocess.py (Dataset932, 4ch 3D, 一体化)
# 更精确但慢 (~19h)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

export nnUNet_raw="/tmp/nnUNet_raw"
export nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
export nnUNet_results="/tmp/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

python ${PROJ}/mama-synth/src/preprocessing/mask_and_preprocess.py \
    --image_dir ${PROJ}/images \
    --seg_dir ${PROJ}/segmentations/automatic \
    --output_dir ${PROJ}/data_split_v5/train \
    --global_stats ${PROJ}/mama-synth/src/preprocessing/training_pre_stats.json \
    --breast_model_dir ${PROJ}/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --skip_ambiguous_shapes \
    --exclude_list ${PROJ}/mama-synth/src/preprocessing/motion_cases.txt \
    --start ${BATCH_START:-0} \
    --end ${BATCH_END:-1507} \
    --device cuda

echo "Done!"
