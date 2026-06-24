#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J kfold_v20_eval
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_eval_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# K-fold v20 ensemble inference + evaluation

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip install xgboost scikit-learn --quiet 2>/dev/null

cd $PROJ/mama-synth

echo "=== K-Fold v20 Ensemble Inference + Evaluation ==="

python -c "
import os, sys, csv, torch, numpy as np, SimpleITK as sitk, json
from pathlib import Path
from tqdm import tqdm
from functools import partial
import torch.nn as nn

sys.path.insert(0, 'src/submission/submission-synthesis')
from models.networks import GlobalGenerator

device = torch.device('cuda')
norm_layer = partial(nn.InstanceNorm2d, affine=False)

# Load splits
fold_map = {}
with open('kfold_splits_no_labreast.csv') as f:
    for row in csv.DictReader(f):
        fold_map[row['case_id']] = int(row['fold'])

# Load all 5 models
models = {}
for fold in range(5):
    w = f'$PROJ/checkpoints/kfold_v20_f{fold}/latest_net_G.pth'
    if not Path(w).exists():
        print(f'WARNING: fold {fold} weights not found, skipping')
        continue
    netG = GlobalGenerator(1,1,64,4,9,norm_layer,residual_mode=True)
    netG.load_state_dict(torch.load(w, map_location=device))
    netG.to(device).eval()
    models[fold] = netG
print(f'Loaded {len(models)} models')

data_dir = Path('$PROJ/data_split_v4/train/mha')
output_dir = Path('$PROJ/predictions_kfold_v20_ensemble')
output_dir.mkdir(exist_ok=True)

all_cases = sorted(fold_map.keys())
print(f'Ensembling {len(all_cases)} cases...')

for case_id in tqdm(all_cases):
    out_path = output_dir / f'{case_id}.mha'
    if out_path.exists():
        continue
    input_path = data_dir / 'input' / f'{case_id}.mha'
    if not input_path.exists():
        continue
    img = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    oh, ow = sl.shape
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
    if oh != 512 or ow != 512:
        t = torch.nn.functional.interpolate(t, (512,512), mode='bilinear', align_corners=False)

    # Ensemble: models that did NOT see this case during training
    case_fold = fold_map[case_id]
    ensemble_folds = [f for f in models.keys() if f != case_fold]

    preds = []
    with torch.no_grad():
        for f in ensemble_folds:
            preds.append(models[f](t))
    result = torch.stack(preds).mean(dim=0)

    if oh != 512 or ow != 512:
        result = torch.nn.functional.interpolate(result, (oh,ow), mode='bilinear', align_corners=False)
    res = result[0,0].cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        res = res[np.newaxis,...]
    out_img = sitk.GetImageFromArray(res)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

print(f'Done. {len(list(output_dir.glob(\"*.mha\")))} ensemble predictions')
"

echo ""
echo "=== Running Evaluation ==="

MAMA_PREDICTIONS_DIR=$PROJ/predictions_kfold_v20_ensemble \
MAMA_GT_DIR=$PROJ/data_split_v4/train/mha \
MAMA_MASKS_DIR=$PROJ/data_split_v4/train/mha/mask \
MAMA_PRECONTRAST_DIR=$PROJ/data_split_v4/train/mha/input \
MAMA_MODELS_DIR=src/evaluation/models \
MAMA_OUTPUT_DIR=$PROJ/eval_kfold_v20_ensemble \
python src/evaluation/evaluate.py

echo ""
echo "=== Results ==="
python -c "
import json
d = json.load(open('$PROJ/eval_kfold_v20_ensemble/metrics.json'))
agg = d['aggregates']
for k in sorted(agg.keys()):
    v = agg[k]
    val = v['mean'] if isinstance(v, dict) else v
    print(f'  {k}: {val:.4f}')
"
