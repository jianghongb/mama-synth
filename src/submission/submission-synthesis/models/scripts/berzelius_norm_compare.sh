#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 6:00:00
#SBATCH -J norm_compare
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/norm_compare_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/norm_compare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Quick comparison: Global z-score vs Per-image z-score normalization
# Both use v22 params (ngf=64, blocks=12, batch=16, lr=0.0003)
# 50 epochs each on data_multislice_v3 → evaluate on test set
# Total time estimate: ~4-5 hours (2x ~2.5h each)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
DATAROOT=$PROJ/data_multislice_v3/train
TESTROOT=$PROJ/data_multislice_v3/test
MASK_OUTPUT=$PROJ/data_multislice_v3/train/mha/breast_mask
MASK_TEST=$PROJ/data_multislice_v3/test/mha/breast_mask

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

cd $PROJ/mama-synth
git pull origin dev
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

COMMON_ARGS="--model pix2pixHD \
  --dataroot $DATAROOT \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 --input_nc 1 --output_nc 1 --no_instance \
  --residual_mode --intensity_aug --noise_aug \
  --resize_or_crop resize --loadSize 512 --fineSize 512 \
  --n_downsample_global 4 --ngf 64 --n_blocks_global 12 --norm instance \
  --batchSize 16 --nThreads 16 \
  --niter 50 --niter_decay 0 --lr 0.0003 \
  --lambda_feat 10 --lambda_gan 1.0 --tumor_weight 10 \
  --lambda_mssc 50 --mssc_levels 3 --lambda_ssim 0 --lambda_vgg 10 \
  --num_D 2 --n_layers_D 3 \
  --breast_mask_dir $MASK_OUTPUT \
  --save_epoch_freq 50 --print_freq 100 --gpu_ids 0"

# ============================================================
# Experiment A: Global z-score (standard mha dataset)
# ============================================================
echo ""
echo "=========================================="
echo "=== Exp A: Global z-score (50 epochs) ==="
echo "=========================================="
echo ""

python train.py --name norm_compare_global --dataset_mode mha $COMMON_ARGS

# ============================================================
# Experiment B: Per-image z-score
# ============================================================
echo ""
echo "============================================="
echo "=== Exp B: Per-image z-score (50 epochs) ==="
echo "============================================="
echo ""

python train.py --name norm_compare_perimage --dataset_mode mha_perimage_norm $COMMON_ARGS

# ============================================================
# Inference + Evaluate both
# ============================================================
echo ""
echo "=========================================="
echo "=== Inference & Evaluation ==="
echo "=========================================="
echo ""

python -c "
import torch, sys, os, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, '.')
from networks import GlobalGenerator
import torch.nn.functional as F
import functools

device = torch.device('cuda')
norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False)

TEST_DIR = '$TESTROOT/mha/input'
MASK_DIR = '$MASK_TEST'

def load_model(weights_path):
    netG = GlobalGenerator(1, 1, 64, 4, 12, norm_layer, residual_mode=True)
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    return netG.to(device).eval()

def infer_global(netG, files, pred_dir):
    '''Standard inference (no re-normalization).'''
    os.makedirs(pred_dir, exist_ok=True)
    for f in tqdm(files, desc='Global'):
        out_path = Path(pred_dir) / f.name
        if out_path.exists(): continue
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
        h, w = arr.shape
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
        t = F.interpolate(t, size=(512,512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = netG(t)
        result = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
        result = result[0,0].cpu().numpy()
        if sitk.GetArrayFromImage(img).ndim == 3:
            result = result[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

def infer_perimage(netG, files, pred_dir, mask_dir):
    '''Per-image norm inference: normalize → model → de-normalize.'''
    os.makedirs(pred_dir, exist_ok=True)
    for f in tqdm(files, desc='PerImage'):
        out_path = Path(pred_dir) / f.name
        if out_path.exists(): continue
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
        h, w = arr.shape

        # Load breast mask
        bm_path = Path(mask_dir) / f.name
        if bm_path.exists():
            bm = sitk.GetArrayFromImage(sitk.ReadImage(str(bm_path))).astype(np.float32).squeeze()
            breast_mask = (bm > 0).astype(np.float32)
        else:
            breast_mask = (arr > -0.3).astype(np.float32)

        # Per-image normalize
        fg = arr[breast_mask > 0.5]
        if fg.size > 100:
            mu, sigma = float(fg.mean()), max(float(fg.std()), 1e-8)
        else:
            mu, sigma = 0.0, 1.0
        arr_norm = (arr - mu) / sigma * breast_mask

        t = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        t = F.interpolate(t, size=(512,512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = netG(t)
        result_norm = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
        result_norm = result_norm[0,0].cpu().numpy()

        # De-normalize
        result = result_norm * sigma + mu

        if sitk.GetArrayFromImage(img).ndim == 3:
            result = result[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(result.astype(np.float32))
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))

files = sorted(Path(TEST_DIR).glob('*.mha'))
print(f'Test cases: {len(files)}')

# Exp A: Global
netG_global = load_model('$PROJ/checkpoints/norm_compare_global/latest_net_G.pth')
infer_global(netG_global, files, '$PROJ/predictions_norm_global')
del netG_global
torch.cuda.empty_cache()

# Exp B: Per-image
netG_pi = load_model('$PROJ/checkpoints/norm_compare_perimage/latest_net_G.pth')
infer_perimage(netG_pi, files, '$PROJ/predictions_norm_perimage', '$MASK_TEST')
del netG_pi
torch.cuda.empty_cache()

print('Inference done.')
"

# Evaluate both
echo ""
echo "=== Evaluating Global z-score ==="
export MAMA_PREDICTIONS_DIR=$PROJ/predictions_norm_global
export MAMA_GT_DIR=$TESTROOT/mha/ground_truth
export MAMA_MASKS_DIR=$TESTROOT/mha/mask
export MAMA_MODELS_DIR=$PROJ/mama-synth/src/evaluation/models
export MAMA_OUTPUT_DIR=$PROJ/eval_norm_global
mkdir -p $MAMA_OUTPUT_DIR
cd $PROJ/mama-synth
python src/evaluation/evaluate.py
python -c "import json; d=json.load(open('$PROJ/eval_norm_global/metrics.json')); print(json.dumps(d['aggregates'], indent=2))"

echo ""
echo "=== Evaluating Per-image z-score ==="
export MAMA_PREDICTIONS_DIR=$PROJ/predictions_norm_perimage
export MAMA_OUTPUT_DIR=$PROJ/eval_norm_perimage
mkdir -p $MAMA_OUTPUT_DIR
python src/evaluation/evaluate.py
python -c "import json; d=json.load(open('$PROJ/eval_norm_perimage/metrics.json')); print(json.dumps(d['aggregates'], indent=2))"

echo ""
echo "=========================================="
echo "=== COMPARISON COMPLETE ==="
echo "=========================================="
echo "Global:   $PROJ/eval_norm_global/metrics.json"
echo "PerImage: $PROJ/eval_norm_perimage/metrics.json"
