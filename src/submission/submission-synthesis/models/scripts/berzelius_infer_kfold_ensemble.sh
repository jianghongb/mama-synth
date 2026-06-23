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
# K-fold ensemble inference: each fold model predicts its own held-out test set.
# All predictions are merged into a single directory covering all training cases.

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

KFOLD_DIR=$PROJ/kfold
PRED_DIR=$PROJ/mama-synth/predictions_kfold_ensemble
GT_DIR=$PROJ/mama-synth/gt_kfold_ensemble
MASK_DIR=$PROJ/mama-synth/mask_kfold_ensemble
EVAL_DIR=$PROJ/mama-synth/eval_results_kfold_ensemble

mkdir -p $PRED_DIR $GT_DIR $MASK_DIR

echo "=== K-fold Ensemble Inference (each fold → its own test set) ==="
python -c "
import sys, torch, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from functools import partial
import torch.nn as nn
import shutil

sys.path.insert(0, 'src/submission/submission-synthesis')
from models.networks import GlobalGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
norm_layer = partial(nn.InstanceNorm2d, affine=False)

kfold_dir = Path('$KFOLD_DIR')
pred_dir = Path('$PRED_DIR')
gt_dir = Path('$GT_DIR')
mask_dir = Path('$MASK_DIR')

total = 0
for fold in range(5):
    print(f'--- Fold {fold} ---')
    # Load model
    wpath = '$PROJ/checkpoints/kfold_f{}/latest_net_G.pth'.format(fold)
    netG = GlobalGenerator(input_nc=1, output_nc=1, ngf=64, n_downsampling=4,
                           n_blocks=9, norm_layer=norm_layer, residual_mode=True)
    netG.load_state_dict(torch.load(wpath, map_location=device))
    netG.to(device).eval()

    # This fold's test set
    test_input = kfold_dir / f'fold_{fold}' / 'test' / 'mha' / 'input'
    test_gt = kfold_dir / f'fold_{fold}' / 'test' / 'mha' / 'ground_truth'
    test_mask = kfold_dir / f'fold_{fold}' / 'test' / 'mha' / 'mask'

    files = sorted(test_input.glob('*.mha'))
    print(f'  {len(files)} test cases')

    for f in tqdm(files, desc=f'fold{fold}'):
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

        with torch.no_grad():
            result = netG(t)

        if orig_h != 512 or orig_w != 512:
            result = torch.nn.functional.interpolate(result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        out_arr = result[0, 0].cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            out_arr = out_arr[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

        # Symlink GT and mask for evaluation
        gt_src = test_gt / f.name
        mask_src = test_mask / f.name
        if gt_src.exists() and not (gt_dir / f.name).exists():
            (gt_dir / f.name).symlink_to(gt_src.resolve())
        if mask_src.exists() and not (mask_dir / f.name).exists():
            (mask_dir / f.name).symlink_to(mask_src.resolve())

    total += len(files)
    del netG
    torch.cuda.empty_cache()

print(f'Total predictions: {total}')
"

# Evaluate
echo "=== Evaluating kfold ensemble ==="
export MAMA_PREDICTIONS_DIR=$PRED_DIR
export MAMA_GT_DIR=$GT_DIR
export MAMA_MASKS_DIR=$MASK_DIR
export MAMA_OUTPUT_DIR=$EVAL_DIR
export MAMA_MODELS_DIR=$PROJ/mama-synth/src/evaluation/models
mkdir -p $EVAL_DIR
python src/evaluation/evaluate.py

echo "--- results ---"
python -m json.tool $EVAL_DIR/metrics.json | grep -A2 '"mean"' | head -20
echo ""
echo "=== Done. Results: $EVAL_DIR/metrics.json ==="
