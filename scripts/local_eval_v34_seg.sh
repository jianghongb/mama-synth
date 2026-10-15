#!/usr/bin/env bash
# LOCAL eval of GAN v34_seg (segmentation-consistency loss) on the
# data_multislice_v3 test set (299 slices), so it is directly comparable to the
# FM / I2SB / GAN runs in the DSB repo.
#
# Two steps:
#   1. batch prediction (scripts/predict_v34_seg.py) reusing precomputed masks
#   2. MAMA-SYNTH evaluator -> metrics.json
#
# Uses the mama-synth .venv (torch+MPS, scipy, SimpleITK, radiomics). Run with
# MPS and do NOT run two evals at once (nnU-Net segmentation degrades past
# ~250 cases under memory pressure).
#
# Usage:
#   bash scripts/local_eval_v34_seg.sh
set -e

MAMA=/Users/ehogjig/git/kth/mama-synth
DATA=/Users/ehogjig/git/kth/data_multislice_v3
TEST_ROOT=$DATA/test/mha
NAME=v34_seg
WEIGHTS=${MAMA_WEIGHTS_PATH:-$MAMA/weights_v34/latest_net_G.pth}
PRED_DIR=$MAMA/predictions_${NAME}
EVAL_DIR=$MAMA/eval_${NAME}
DEVICE=${DEVICE:-mps}

# libomp loader-path workaround (xgboost in the evaluator)
LIBOMP=/Users/ehogjig/homebrew/Cellar/libomp/22.1.8/lib
[ -d "$LIBOMP" ] && export DYLD_LIBRARY_PATH="$LIBOMP:$DYLD_LIBRARY_PATH"

cd "$MAMA"
source .venv/bin/activate

if [ ! -f "$WEIGHTS" ]; then
  echo "ERROR: weights not found: $WEIGHTS" >&2
  exit 1
fi

echo "=============================================================="
echo " LOCAL eval: GAN $NAME"
echo "   weights = $WEIGHTS"
echo "   test    = $TEST_ROOT  (299 slices, data_multislice_v3)"
echo "   device  = $DEVICE"
echo "=============================================================="

# ---- Step 1: batch prediction ----
MAMA_WEIGHTS_PATH="$WEIGHTS" TEST_ROOT="$TEST_ROOT" OUT_DIR="$PRED_DIR" DEVICE="$DEVICE" \
  python scripts/predict_v34_seg.py
echo "[Step 1] prediction done -> $PRED_DIR"

# ---- Step 2: MAMA-SYNTH evaluator ----
export MAMA_PREDICTIONS_DIR=$PRED_DIR
export MAMA_GT_DIR=$TEST_ROOT/ground_truth
export MAMA_MASKS_DIR=$TEST_ROOT/mask
export MAMA_MODELS_DIR=$MAMA/src/evaluation/models
export MAMA_OUTPUT_DIR=$EVAL_DIR
export MAMA_ENSEMBLE=True

mkdir -p "$EVAL_DIR"
python "$MAMA/src/evaluation/evaluate.py"

echo ""
echo " metrics -> $EVAL_DIR/metrics.json"
if [ -f "$EVAL_DIR/metrics.json" ]; then
  python -c "
import json
m = json.load(open('$EVAL_DIR/metrics.json')); agg = m.get('aggregates', {})
print('=== $NAME key metrics ===')
for k in ['mse','lpips','ssim_tumor','frd','dice','hausdorff_95','auroc_contrast']:
    v = agg.get(k, {})
    if isinstance(v, dict) and 'mean' in v: print(f'  {k:16s}: {v[\"mean\"]:.4f}')
"
fi
