#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J sdedit_sweep
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Grid search over SDEdit strength to find optimal value
# Tests: 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip install xgboost scikit-learn --quiet 2>/dev/null

TEST_INPUT=$PROJ/data_split/test/mha/input
TEST_GT=$PROJ/data_split/test/mha
GAN_WEIGHTS=$PROJ/checkpoints/mamasynth_v14/latest_net_G.pth
REFINER_WEIGHTS=$PROJ/checkpoints/refiner_v17/refiner_latest.pth

cd $PROJ/mama-synth

echo "=== SDEdit Strength Grid Search ==="
echo ""

for STRENGTH in 0.1 0.15 0.2 0.25 0.3 0.4 0.5; do
    PRED_DIR=$PROJ/predictions_sdedit_s${STRENGTH}
    EVAL_DIR=$PROJ/eval_sdedit_s${STRENGTH}

    if [ -f "$EVAL_DIR/metrics.json" ]; then
        echo "[$STRENGTH] Already done, skipping."
        continue
    fi

    echo "[$STRENGTH] Running inference..."
    mkdir -p $PRED_DIR

    python -c "
import os, sys, torch, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from functools import partial
import torch.nn as nn

sys.path.insert(0, 'src/submission/submission-synthesis')
from models.networks import GlobalGenerator
from models.refiner_network import RefinerUNet
from inference import cosine_alpha_bar, sdedit_refine

device = torch.device('cuda')
norm_layer = partial(nn.InstanceNorm2d, affine=False)
netG = GlobalGenerator(1,1,64,4,9,norm_layer,residual_mode=True)
netG.load_state_dict(torch.load('$GAN_WEIGHTS', map_location=device))
netG.to(device).eval()

refiner = RefinerUNet(3,1,64,(1,2,4,4))
refiner.load_state_dict(torch.load('$REFINER_WEIGHTS', map_location=device))
refiner.to(device).eval()

input_dir = Path('$TEST_INPUT')
output_dir = Path('$PRED_DIR')
strength = $STRENGTH

for f in tqdm(sorted(input_dir.glob('*.mha'))):
    out_path = output_dir / f.name
    if out_path.exists():
        continue
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()
    oh, ow = sl.shape
    t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
    if oh != 512 or ow != 512:
        t = torch.nn.functional.interpolate(t, (512,512), mode='bilinear', align_corners=False)
    with torch.no_grad():
        gan_out = netG(t)
        result = sdedit_refine(refiner, gan_out, t, device, strength=strength, num_steps=20)
    if oh != 512 or ow != 512:
        result = torch.nn.functional.interpolate(result, (oh,ow), mode='bilinear', align_corners=False)
    res = result[0,0].cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        res = res[np.newaxis,...]
    out_img = sitk.GetImageFromArray(res)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))
print(f'Done: {len(list(output_dir.glob(\"*.mha\")))} predictions')
"

    echo "[$STRENGTH] Running evaluation..."
    MAMA_PREDICTIONS_DIR=$PRED_DIR \
    MAMA_GT_DIR=$TEST_GT \
    MAMA_MASKS_DIR=$TEST_GT/mask \
    MAMA_PRECONTRAST_DIR=$TEST_INPUT \
    MAMA_MODELS_DIR=src/evaluation/models \
    MAMA_OUTPUT_DIR=$EVAL_DIR \
    python src/evaluation/evaluate.py

    echo "[$STRENGTH] Results:"
    python -c "
import json
with open('$EVAL_DIR/metrics.json') as f:
    d = json.load(f)
agg = d['aggregates']
for k in sorted(agg.keys()):
    v = agg[k]
    val = v['mean'] if isinstance(v, dict) else v
    print(f'  {k}: {val:.4f}')
"
    echo ""
done

echo "=== Summary ==="
python -c "
import json
from pathlib import Path

print(f'{\"strength\":<10} {\"MSE\":<8} {\"LPIPS\":<8} {\"SSIM\":<8} {\"FRD\":<8} {\"Dice\":<8} {\"HD95\":<8}')
print('-'*58)
for s in ['0.1','0.15','0.2','0.25','0.3','0.4','0.5']:
    p = Path('$PROJ/eval_sdedit_s' + s + '/metrics.json')
    if not p.exists():
        continue
    d = json.load(open(p))['aggregates']
    def g(k):
        v = d.get(k, 0)
        return v['mean'] if isinstance(v, dict) else v
    print(f'{s:<10} {g(\"mse\"):<8.4f} {g(\"lpips\"):<8.4f} {g(\"ssim_tumor\"):<8.4f} {g(\"frd\"):<8.4f} {g(\"dice\"):<8.4f} {g(\"hausdorff_95\"):<8.1f}')
"
