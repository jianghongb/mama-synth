#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J infer_bilat_v21
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/infer_bilat_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/infer_bilat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Batch inference using bilateral-split pipeline on test data.
# Model trained on unilateral crops → split bilateral test → predict → stitch.
#
# Usage: sbatch berzelius_infer_bilateral.sh [VERSION]

VERSION=${1:-21}
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

cd $PROJ/mama-synth

# Config
WEIGHTS=$PROJ/checkpoints/mamasynth_v${VERSION}/latest_net_G.pth
BREAST_SEG=$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d
TEST_DIR=$PROJ/data_split_v2/test/mha
PRED_DIR=$PROJ/mama-synth/predictions_v${VERSION}_bilat
EVAL_DIR=$PROJ/mama-synth/eval_results_v${VERSION}_bilat

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Weights not found: $WEIGHTS"
    exit 1
fi

mkdir -p $PRED_DIR

# ============================================================
# Batch inference: split → predict → stitch for each test case
# ============================================================
echo "=== Bilateral-split inference v${VERSION} ==="
echo "  Weights: $WEIGHTS"
echo "  Test input: $TEST_DIR/input"
echo "  Output: $PRED_DIR"

python -c "
import os, sys
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, 'src/submission/submission-synthesis')
sys.path.insert(0, 'src/preprocessing')

os.environ['MAMA_WEIGHTS_PATH'] = '$WEIGHTS'
os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'

from models.networks import GlobalGenerator
from stitch_bilateral import BilateralSplitter
from split_bilateral import find_chest_cut_row
from functools import partial
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build generator
norm_layer = partial(nn.InstanceNorm2d, affine=False)
netG = GlobalGenerator(input_nc=1, output_nc=1, ngf=64, n_downsampling=4,
                       n_blocks=9, norm_layer=norm_layer, residual_mode=True)
netG.load_state_dict(torch.load('$WEIGHTS', map_location=device))
netG.to(device).eval()

# Build breast predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy import ndimage as ndi

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$BREAST_SEG', use_folds=(0,), checkpoint_name='checkpoint_final.pth')

splitter = BilateralSplitter(target_size=512, pad_ratio=0.03)

input_dir = Path('$TEST_DIR/input')
pred_dir = Path('$PRED_DIR')
files = sorted(input_dir.glob('*.mha'))
print(f'Processing {len(files)} test cases...')

props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
         'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}

for f in tqdm(files):
    out_path = pred_dir / f.name
    if out_path.exists():
        continue

    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sl = arr.squeeze()

    # Breast mask
    input_arr = sl[np.newaxis, np.newaxis, :, :]
    pred_mask = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred_mask.ndim > 2:
        pred_mask = pred_mask[0]
    bm = (pred_mask > 0).astype(np.uint8)
    labeled, n = ndi.label(bm)
    if n > 1:
        sizes = ndi.sum(bm, labeled, range(1, n+1))
        thresh = sizes.max() * 0.1
        bm = np.zeros_like(bm, dtype=np.uint8)
        for i, s in enumerate(sizes):
            if s >= thresh:
                bm[labeled == (i+1)] = 1
    breast_mask = bm.astype(np.float32)

    # Chest wall removal
    cut_row = find_chest_cut_row(breast_mask, pad=20)
    if cut_row is not None and cut_row < breast_mask.shape[0]:
        breast_mask[cut_row:, :] = 0

    # Split
    left, right, meta = splitter.split(sl, breast_mask)

    # Predict each side
    def synthesize(crop):
        if crop is None:
            return None
        t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
        t = torch.nn.functional.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = netG(t)
        return out[0, 0].cpu().numpy()

    pred_l = synthesize(left)
    pred_r = synthesize(right)

    # Stitch + mask
    result = splitter.stitch(pred_l, pred_r, meta)
    result = breast_mask * result + (1 - breast_mask) * sl

    if arr.ndim == 3:
        result = result[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(result.astype(np.float32))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

print(f'Done. {len(list(pred_dir.glob(\"*.mha\")))} predictions in {pred_dir}')
"

# ============================================================
# Evaluate
# ============================================================
echo "=== Evaluating v${VERSION} bilateral-split ==="
export MAMA_PREDICTIONS_DIR=$PRED_DIR
export MAMA_GT_DIR=$TEST_DIR
export MAMA_MASKS_DIR=$TEST_DIR/mask
export MAMA_OUTPUT_DIR=$EVAL_DIR
export MAMA_MODELS_DIR=$PROJ/mama-synth/src/evaluation/models

mkdir -p $EVAL_DIR
python src/evaluation/evaluate.py

echo ""
echo "=== Results ==="
cat $EVAL_DIR/metrics.json | python -m json.tool | head -30
echo ""
echo "Full results: $EVAL_DIR/metrics.json"
