#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J kfold_v20_eval2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_eval2_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_eval2_%j.err
#
# Step 2: Evaluation (needs GPU for nnUNet segmentation)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip install xgboost scikit-learn --quiet 2>/dev/null

cd $PROJ/mama-synth

echo "Predictions: $(ls $PROJ/predictions_kfold_v20_ensemble/*.mha | wc -l)"

MAMA_PREDICTIONS_DIR=$PROJ/predictions_kfold_v20_ensemble \
MAMA_GT_DIR=$PROJ/data_split_v4/train/mha \
MAMA_MASKS_DIR=$PROJ/data_split_v4/train/mha/mask \
MAMA_PRECONTRAST_DIR=$PROJ/data_split_v4/train/mha/input \
MAMA_MODELS_DIR=src/evaluation/models \
MAMA_OUTPUT_DIR=$PROJ/eval_kfold_v20_ensemble \
python src/evaluation/evaluate.py

echo "=== Results ==="
python -c "
import json
d = json.load(open('$PROJ/eval_kfold_v20_ensemble/metrics.json'))
agg = d['aggregates']
for k in sorted(agg):
    v = agg[k]
    print(f'  {k}: {v[\"mean\"] if isinstance(v,dict) else v:.4f}')
"
