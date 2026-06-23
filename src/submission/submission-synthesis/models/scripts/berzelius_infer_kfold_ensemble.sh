#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J infer_kfold_ens
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/infer_kfold_ens_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/infer_kfold_ens_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# K-fold ensemble inference: average predictions from 5 fold models
# Evaluates on data_split/test (199 cases)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

cd $PROJ/mama-synth

TEST_DIR=$PROJ/data_split/test/mha
PRED_DIR=$PROJ/mama-synth/predictions_kfold_ensemble
EVAL_DIR=$PROJ/mama-synth/eval_results_kfold_ensemble

mkdir -p $PRED_DIR

echo "=== K-fold Ensemble Inference (5 models) ==="
python -c "
import sys, torch, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from functools import partial
import torch.nn as nn

sys.path.insert(0, 'src/submission/submission-synthesis')
from models.networks import GlobalGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load 5 fold models
norm_layer = partial(nn.InstanceNorm2d, affine=False)
models = []
for fold in range(5):
    wpath = '$PROJ/checkpoints/kfold_f{}/latest_net_G.pth'.format(fold)
    netG = GlobalGenerator(input_nc=1, output_nc=1, ngf=64, n_downsampling=4,
                           n_blocks=9, norm_layer=norm_layer, residual_mode=True)
    netG.load_state_dict(torch.load(wpath, map_location=device))
    netG.to(device).eval()
    models.append(netG)
    print(f'  Loaded fold {fold}')

input_dir = Path('$TEST_DIR/input')
pred_dir = Path('$PRED_DIR')
files = sorted(input_dir.glob('*.mha'))
print(f'Processing {len(files)} cases with 5-model ensemble...')

for f in tqdm(files):
    out_path = pred_dir / f.name
    if out_path.exists():
        continue

    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    orig_h, orig_w = sl.shape

    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float().to(device)
    if orig_h != 512 or orig_w != 512:
        t = torch.nn.functional.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)

    # Ensemble: average 5 predictions
    preds = []
    with torch.no_grad():
        for netG in models:
            preds.append(netG(t))
    result = torch.stack(preds).mean(0)

    if orig_h != 512 or orig_w != 512:
        result = torch.nn.functional.interpolate(result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    out_arr = result[0, 0].cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        out_arr = out_arr[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

print(f'Done. {len(list(pred_dir.glob(\"*.mha\")))} predictions')
"

# Evaluate
echo "=== Evaluating kfold ensemble ==="
export MAMA_PREDICTIONS_DIR=$PRED_DIR
export MAMA_GT_DIR=$TEST_DIR
export MAMA_MASKS_DIR=$TEST_DIR/mask
export MAMA_OUTPUT_DIR=$EVAL_DIR
export MAMA_MODELS_DIR=$PROJ/mama-synth/src/evaluation/models
mkdir -p $EVAL_DIR
python src/evaluation/evaluate.py

echo "--- results ---"
python -m json.tool $EVAL_DIR/metrics.json | grep -A2 '"mean"' | head -20
echo ""
echo "=== Done. Results: $EVAL_DIR/metrics.json ==="
