#!/bin/bash
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth

python src/preprocessing/generate_kfold_splits.py \
    --data_dir $PROJ/data_split_v4/train/mha/input \
    --mask_dir $PROJ/data_split_v4/train/mha/mask \
    --output_csv kfold_splits.csv \
    --k 5
