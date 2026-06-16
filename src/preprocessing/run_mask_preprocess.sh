#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J mask_preprocess
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/mask_preprocess_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/mask_preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se

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

MAMA=$PROJ/mama-synth/src/submission/submission-synthesis/models
OUTPUT=$PROJ/data_split_v5/train

# Step 1: Preprocess without breast mask (uses GPU for nothing heavy, but fast)
echo "=== Step 1: Preprocess ==="
python $PROJ/mama-synth/src/preprocessing/preprocess.py \
    --image_dir ${PROJ}/images \
    --seg_dir ${PROJ}/segmentations/automatic \
    --output_dir ${OUTPUT} \
    --global_stats ${PROJ}/mama-synth/src/preprocessing/training_pre_stats.json \
    --skip_ambiguous_shapes \
    --exclude_list ${PROJ}/mama-synth/src/preprocessing/motion_cases.txt

# Step 2: Generate breast masks using Dataset910 (2D, fast ~2s/case)
echo "=== Step 2: Generate breast masks (Dataset910, 2D) ==="
python $MAMA/generate_breast_masks.py \
    --input_dir ${OUTPUT}/mha/input \
    --output_dir ${OUTPUT}/mha/breast_mask \
    --model_dir $PROJ/weights/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetResEncUNetLPlans__2d \
    --model_type 910

echo "=== Done! ==="
echo "Output: ${OUTPUT}"
echo "Breast masks: ${OUTPUT}/mha/breast_mask"
