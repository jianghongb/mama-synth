#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v34_seg
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v34-seg: segmentation-consistency loss, single variable against v32_no_msec
# ===========================================================================
# PURPOSE
#   The intensity losses ask the synthetic image to look right. This run asks
#   whether the lesion is still *findable*: a frozen tumour segmenter is applied
#   to the generated image and an asymmetric Tversky term compares its tumour
#   probability against the ground-truth mask, with gradients flowing back
#   through the frozen network into the generator. That targets Dice and HD95
#   directly, which the intensity terms only reach indirectly.
#
#   The idea is adapted from Anguinus Sculpturae (MAMA-SYNTH submission), which
#   reports Dice 0.69 and HD95 62.5 on its Duke split with a comparable term.
#
# BASELINE AND THE SINGLE CHANGE
#   Derived from berzelius_train_v32_no_msec.sh, i.e. row (5) of the paper's
#   Table 2, with exactly one group of flags added:
#
#       --lambda_seg 25 --seg_loss_weights ... --seg_tversky_alpha/beta
#
#   Row (5) rather than row (4) is the baseline because the MSEC ablation already
#   showed that term redundant once per-image normalisation and the tumour mask
#   are present, so building on the configuration without it keeps the new
#   result attributable to one addition rather than two.
#
#   lambda_seg = 0 is a clean off-switch, the same pattern as lambda_mssc:
#     models/pix2pixHD_model.py  self.use_seg_loss = self.lambda_seg > 0
#     models/pix2pixHD_model.py  if self.use_seg_loss and mask is not None:
#   The criterion is not even constructed when the weight is zero, so existing
#   runs stay bit-comparable.
#
# WHICH SEGMENTER, AND WHY IT MATTERS
#   The challenge scores Dice and HD95 with Dataset101_mamasynth_firstpost
#   (src/evaluation/models/segmentation). Routing the loss through that same
#   network would optimise the metric rather than the synthesis, and the
#   resulting Dice would not mean what it appears to.
#
#   This run therefore uses Dataset940_TumorSegT1w, the tumour segmenter trained
#   for this work on the 9,504 train-split slices alone. Different training set,
#   different patch size (448 against 256), no test patient in its data. Keeping
#   the loss network separate from the scoring network is what makes "remains
#   segmentable" an independent claim. Do not point --seg_loss_weights at the
#   evaluation directory.
#
# lambda_seg = 25 COMES FROM MEASUREMENT, NOT GUESSWORK
#   scripts/calibrate_seg_loss.py ran the term over predictions_v32_no_msec, the
#   outputs of this very baseline, on all 299 test slices:
#
#     cases where the term is non-zero : 166/299 (55.5 %), mean 0.1766
#     cases already as segmentable as the real image : 133
#     cases the segmenter misses even in the real image : 211/299 (70.6 %)
#
#   At lambda_seg = 25 the contribution is about 4.4 on the cases it acts on,
#   the same decade as lambda_vgg = 10 and tumor_weight = 10 times their typical
#   values. At 1 it would be 0.18 and drowned; at 50, 8.8 and liable to dominate.
#
# TWO PROPERTIES WORTH KNOWING BEFORE READING THE RESULT
#   The term is baselined as relu(L_fake - L_real), so nothing is demanded beyond
#   what the segmenter already sees in the real image. This matters because its
#   median Tversky on real images is 0.98 in this single-pass full-image setting:
#   it misses most real lesions here, and an absolute target would push the
#   generator to over-enhance in pursuit of something unreachable. The relu also
#   means the term goes silent once a lesion is as findable as the real one, so
#   it cannot chase a non-truth optimum.
#
#   Supervision therefore covers about 55 % of slices. Expect any Dice gain to be
#   concentrated there and diluted in the aggregate. Report that fraction
#   alongside the metrics.
#
# BEFORE SUBMITTING
#   python scripts/check_seg_loss.py       # 7 checks, all should pass
#   python scripts/calibrate_seg_loss.py   # re-measure if the baseline changed
#
# AFTER TRAINING
#   Run inference and evaluation with the same settings as row (5), then compare
#   against eval_results_v32_no_msec:
#     MSE 1.120  LPIPS 0.125  SSIM_tum 0.520  FRD 25.83  Dice 0.624  HD95 69.8
#   Dice and HD95 are the metrics this term targets; the others are the cost side
#   of the trade. Use scripts/compare_paired.py in DSB_for_chanllenge for a
#   paired test rather than comparing aggregate means, since differences at this
#   scale can sit inside the noise.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics
export TORCHDYNAMO_DISABLE=1
# The segmenter's transient activations fragment the pool; the first attempt
# died with 652 MiB reserved-but-unallocated on a 40 GiB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $PROJ/mama-synth
git pull origin dev

DATAROOT=$PROJ/data_multislice_v3/train
BREAST_MASK=$PROJ/data_multislice_v3/train/mha/breast_mask
TUMOR_MASK=$PROJ/data_multislice_v3/train/mha/predicted_tumor

# The segmenter for the loss lives in different places on different machines:
# the nnU-Net results tree on the cluster, where it was trained, and a copy under
# weights/ locally for the probe and calibration scripts. Large .pth files do not
# travel in git, so neither location is guaranteed. Take the first candidate that
# has all three pieces the loss needs.
SEG_CANDIDATES=(
  "$PROJ/nnUNet_results/Dataset940_TumorSegT1w/nnUNetTrainer__nnUNetPlans__2d"
  "$PROJ/mama-synth/src/submission/submission-synthesis/weights/tumor_seg_940"
  "$PROJ/mama-synth/src/submission/submission-synthesis/weights/tumor_seg"
)
SEG_W=""
for c in "${SEG_CANDIDATES[@]}"; do
  if [ -f "$c/plans.json" ] && [ -f "$c/dataset.json" ] && \
     ls "$c"/fold_0/checkpoint_*.pth >/dev/null 2>&1; then
    SEG_W="$c"; break
  fi
done

NUM_TUMOR=$(ls $TUMOR_MASK/*.mha 2>/dev/null | wc -l)
if [ "$NUM_TUMOR" -lt 100 ]; then
    echo "ERROR: Only $NUM_TUMOR predicted tumor masks. Run berzelius_tumor_seg_infer.sh first!"
    exit 1
fi

# The loss is useless if the frozen segmenter is absent or is the scoring one.
if [ -z "$SEG_W" ]; then
    echo "ERROR: no usable segmenter for --lambda_seg. Looked for plans.json," >&2
    echo "       dataset.json and fold_0/checkpoint_*.pth under:" >&2
    for c in "${SEG_CANDIDATES[@]}"; do
        echo "         $c" >&2
        for p in plans.json dataset.json; do
            [ -f "$c/$p" ] || echo "           missing $p" >&2
        done
        ls "$c"/fold_0/checkpoint_*.pth >/dev/null 2>&1 || \
            echo "           missing fold_0/checkpoint_*.pth" >&2
    done
    exit 1
fi
if python - "$SEG_W/plans.json" <<'PYEOF'
import json, sys
name = json.load(open(sys.argv[1])).get("dataset_name", "")
sys.exit(0 if name == "Dataset101_mamasynth_firstpost" else 1)
PYEOF
then
    echo "ERROR: --seg_loss_weights points at the segmenter used for scoring." >&2
    echo "       The loss would optimise Dice/HD95 instead of the synthesis." >&2
    exit 1
fi

echo "=== v34-seg: row (5) + segmentation-consistency loss ==="
echo "Predicted tumor masks : $NUM_TUMOR"
echo "Loss segmenter path   : $SEG_W"
echo "Loss segmenter        : $(python -c "import json;print(json.load(open('$SEG_W/plans.json'))['dataset_name'])")"
echo "lambda_seg            : 25  (alpha 0.3, beta 0.7, gt-baselined)"
echo "Changed vs v32_no_msec: --lambda_seg only"
echo ""

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v34_seg \
  --model pix2pixHD \
  --dataset_mode mha_perimage_norm \
  --dataroot $DATAROOT \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 3 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --noise_aug \
  --mask_as_input \
  --tumor_mask_as_input \
  --tumor_mask_dir $TUMOR_MASK \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 12 \
  --norm instance \
  --batchSize 16 \
  --nThreads 16 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0003 \
  --lambda_feat 10 \
  --lambda_gan 1.0 \
  --tumor_weight 10 \
  --lambda_mssc 0 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --lambda_vgg 10 \
  --lambda_seg 25 \
  --seg_loss_weights $SEG_W \
  --seg_loss_fold 0 \
  --seg_tversky_alpha 0.3 \
  --seg_tversky_beta 0.7 \
  --seg_loss_max_batch 4 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $BREAST_MASK \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo ""
echo "Done! Weights: $PROJ/checkpoints/mamasynth_v34_seg/latest_net_G.pth"
echo ""
echo "Confirm in the log that G_Seg appears among the loss names and that its"
echo "value is of order 0.1-0.2 before the lambda is applied. If it sits at zero"
echo "throughout, the term never engaged and the run says nothing."
echo ""
echo "=== Next: inference + evaluation, then the paired comparison ==="
echo "  v32_no_msec (row 5): MSE 1.120  LPIPS 0.125  SSIM 0.520  FRD 25.83  Dice 0.624  HD95 69.8"
echo "  v34_seg            : <fill in from this run>"
