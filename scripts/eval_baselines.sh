#!/bin/bash
# Score the reference baselines that bound every metric, one at a time.
# ============================================================================
# WHY THESE EXIST
#   Over 130 evaluations are recorded in this repository and none of them is a
#   baseline. Every number has been read against other trained models, so no
#   metric has a known floor, and two readings have already had to be retracted
#   for that reason:
#
#     AUROC_contrast -- the two highest-scoring runs turned out to have the two
#       worst SSIM and MSE values, so it was not measuring fidelity.
#     FRD -- the INR run collapsed to the identity map and scored 10.47 against
#       25.8-28.7 for every model that synthesises anything, beating the whole
#       field by 2.5x while predicting no enhancement at all.
#
#   The INR collapse also showed how close to the floor a "working" run can sit:
#   its MSE of 1.6905 is 0.3 % better than copying the pre-contrast image, after
#   17.85 h of training.
#
# THE FOUR BASELINES
#   identity      prediction = pre-contrast, verbatim. The floor. Any metric on
#                 which a trained model fails to beat this is not measuring
#                 synthesis.
#
#   oracle_const  pre-contrast plus a constant 3.75 sigma inside the *true*
#                 tumour mask. Perfect localisation, zero texture. Dice and HD95
#                 are computed by segmenting the prediction, so if a flat block
#                 in the right place scores well, those metrics are measuring
#                 localisation rather than synthesis quality -- which would
#                 change how the GAN's Dice 0.624 against I²SB's 0.479 should be
#                 described. 3.75 sigma is the measured mean true tumour
#                 enhancement, so this does not win by being unusually bright.
#
#   const_offset  pre-contrast plus 0.7749 sigma everywhere inside the breast,
#                 the measured mean enhancement there. No spatial modelling at
#                 all. Separates "knowing enhancement happens" from "knowing
#                 where".
#
#   gt_ceiling    prediction = ground truth. MSE, LPIPS and FRD are trivially
#                 zero, so this is only informative for AUROC_contrast, where it
#                 gives the ceiling. Without it there is no way to tell whether
#                 I²SB's 0.760 is close to the maximum or far from it.
#
# WHY SERIAL
#   docs/HANDOFF.md records that the nnUNet segmentation step degrades badly past
#   ~250 cases under memory pressure, and that two concurrent evaluations drove
#   per-case time from ~8 s to ~90 min. Each evaluation here takes ~45 min, so
#   the four run back to back rather than together.
#
# Usage:
#   bash scripts/eval_baselines.sh                 # all four
#   bash scripts/eval_baselines.sh identity        # a subset

set -e

MAMA=/Users/ehogjig/git/kth/mama-synth
DATA=/Users/ehogjig/git/kth/data_multislice_v3/test/mha

ALL="identity oracle_const const_offset gt_ceiling"
RUNS="${*:-$ALL}"

if pgrep -f 'evaluate.py' > /dev/null; then
  echo "An evaluation is already running. Wait for it to finish:" >&2
  pgrep -fl 'evaluate.py' | sed 's/^/  /' >&2
  echo "Running two at once drives per-case segmentation time to ~90 min." >&2
  exit 1
fi

# MAMA_SKIP_SEG left over from a previous shell silently drops Dice and HD95.
# That already happened once on v34_seg, which produced a metrics.json with no
# segmentation fields and a 41 s total runtime.
unset MAMA_SKIP_SEG

for run in $RUNS; do
  PRED="$MAMA/predictions_$run"
  OUT="$MAMA/eval_results_$run"
  if [ ! -d "$PRED" ]; then
    echo "skip $run: no $PRED" >&2
    continue
  fi
  if [ -f "$OUT/metrics.json" ]; then
    echo "skip $run: $OUT/metrics.json already exists" >&2
    continue
  fi
  mkdir -p "$OUT"
  echo "=============================================================="
  echo " $run   ($(ls -1 "$PRED"/*.mha | wc -l | tr -d ' ') cases)"
  echo " -> $OUT"
  echo "=============================================================="
  MAMA_PREDICTIONS_DIR="$PRED" \
  MAMA_GT_DIR="$DATA/ground_truth" \
  MAMA_MASKS_DIR="$DATA/mask" \
  MAMA_PRECONTRAST_DIR="$DATA/input" \
  MAMA_MODELS_DIR="$MAMA/src/evaluation/models" \
  MAMA_OUTPUT_DIR="$OUT" \
  MAMA_ENSEMBLE=True \
    caffeinate -is "$MAMA/.venv/bin/python" -u "$MAMA/src/evaluation/evaluate.py" \
    > "$OUT/eval.log" 2>&1
  echo " done: $(grep -c 'empty mask' "$OUT/eval.log" || true) empty-mask warnings"
done

echo ""
echo "Compare with:  python scripts/compare_baselines.py"
