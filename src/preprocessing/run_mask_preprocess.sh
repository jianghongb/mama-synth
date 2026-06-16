#!/bin/bash
#SBATCH -A berzbiomedicalimagingkth
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J mask_preprocess
#SBATCH --output=slurm-%j.out

module load Anaconda/2023.09-0
source activate nnunet  # or your conda env with nnunetv2

BASE="/proj/berzbiomedicalimagingkth/users/x_honji"

export nnUNet_raw="/tmp/nnUNet_raw"
export nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
export nnUNet_results="/tmp/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# Sync mask_and_preprocess.py from mama-synth repo (or copy it over)
SCRIPT="${BASE}/mama-synth/src/preprocessing/mask_and_preprocess.py"

python $SCRIPT \
    --image_dir ${BASE}/images \
    --seg_dir ${BASE}/segmentations/automatic \
    --output_dir ${BASE}/data_split_v5/train \
    --global_stats ${BASE}/mama-synth/src/preprocessing/training_pre_stats.json \
    --breast_model_dir ${BASE}/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres \
    --skip_ambiguous_shapes \
    --exclude_list ${BASE}/mama-synth/src/preprocessing/motion_cases.txt \
    --device cuda

echo "Done!"
