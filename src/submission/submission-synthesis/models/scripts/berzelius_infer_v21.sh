#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J infer_v21
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/infer_v21_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/infer_v21_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v21 inference: bilateral split → unilateral predict → stitch
# Evaluates on both data_split/test (199 cases) and data_split_v2/test (150 cases)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures
pip show scipy > /dev/null 2>&1 || pip install scipy

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

cd $PROJ/mama-synth

WEIGHTS=$PROJ/checkpoints/mamasynth_v21/latest_net_G.pth
BREAST_SEG=$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Weights not found: $WEIGHTS"
    exit 1
fi

# ============================================================
# Inference function
# ============================================================
run_inference() {
    local TEST_DIR=$1
    local PRED_DIR=$2
    local LABEL=$3

    echo "=== [$LABEL] Inference v21 ==="
    echo "  Weights: $WEIGHTS"
    echo "  Input: $TEST_DIR/input"
    echo "  Output: $PRED_DIR"
    mkdir -p $PRED_DIR

    python -c "
import os, sys, torch, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from functools import partial
import torch.nn as nn

sys.path.insert(0, 'src/submission/submission-synthesis')
sys.path.insert(0, 'src/preprocessing')

from models.networks import GlobalGenerator
from stitch_bilateral import BilateralSplitter
from split_bilateral import find_chest_cut_row
from scipy import ndimage as ndi

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load generator
norm_layer = partial(nn.InstanceNorm2d, affine=False)
netG = GlobalGenerator(input_nc=1, output_nc=1, ngf=64, n_downsampling=4,
                       n_blocks=9, norm_layer=norm_layer, residual_mode=True)
netG.load_state_dict(torch.load('$WEIGHTS', map_location=device))
netG.to(device).eval()

# Load breast seg
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$BREAST_SEG', use_folds=(0,), checkpoint_name='checkpoint_final.pth')

splitter = BilateralSplitter(target_size=512, pad_ratio=0.03)

input_dir = Path('$TEST_DIR/input')
pred_dir = Path('$PRED_DIR')
files = sorted(input_dir.glob('*.mha'))
print(f'[$LABEL] Processing {len(files)} cases...')

props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
         'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}

for f in tqdm(files, desc='$LABEL'):
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

    # Stitch
    result = splitter.stitch(pred_l, pred_r, meta)

    # Re-generate full breast mask (without chest cut) for final compositing
    breast_mask_full = bm.astype(np.float32)
    result = breast_mask_full * result + (1 - breast_mask_full) * sl

    if arr.ndim == 3:
        result = result[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(result.astype(np.float32))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

print(f'[$LABEL] Done. {len(list(pred_dir.glob(\"*.mha\")))} predictions')
"
}

# ============================================================
# Run inference
# ============================================================
run_inference "$PROJ/data_split/test/mha" "$PROJ/mama-synth/predictions_v21" "ds1_test"

# ============================================================
# Evaluate
# ============================================================
echo "=== Evaluating v21 on data_split/test ==="
export MAMA_PREDICTIONS_DIR=$PROJ/mama-synth/predictions_v21
export MAMA_GT_DIR=$PROJ/data_split/test/mha
export MAMA_MASKS_DIR=$PROJ/data_split/test/mha/mask
export MAMA_OUTPUT_DIR=$PROJ/mama-synth/eval_results_v21
export MAMA_MODELS_DIR=$PROJ/mama-synth/src/evaluation/models
mkdir -p $MAMA_OUTPUT_DIR
python src/evaluation/evaluate.py
echo "--- results ---"
python -m json.tool $MAMA_OUTPUT_DIR/metrics.json | grep -A2 '"mean"' | head -20

echo ""
echo "=== Done ==="
echo "Results: eval_results_v21/metrics.json"
