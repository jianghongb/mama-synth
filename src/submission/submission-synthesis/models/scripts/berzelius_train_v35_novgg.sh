#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v35_novgg
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v35-noVGG: VGG perceptual loss ablation, single variable against v32_no_msec
# ============================================================================
# WHY THIS RUN
#   lambda_vgg has been 10 in every run since v3 and has never been isolated.
#   A feature probe (scripts/probe_dino_features.py, n=40 real pre/post pairs)
#   gives a concrete reason to question it:
#
#     feature            separation   Spearman(distance, tumour delta)
#     VGG-19 (current)       1.22        +0.117   p = 0.47
#     DINOv2 ViT-S/14        1.37        -0.070   p = 0.67
#
#   Separation is ||f(post) - f(pre)|| divided by ||f(neighbour slice) - f(pre)||,
#   so 1.0 means the feature cannot tell enhancement from moving one slice. VGG
#   manages 1.22, with a minimum of 1.04 across cases.
#
#   More to the point, VGG feature distance does not track enhancement magnitude
#   at all: rho = +0.117, p = 0.47. A perceptual loss works by pushing feature
#   distance down, so if that distance is uncorrelated with whether the
#   enhancement is right, this term is constraining something else -- texture
#   realism, global structure -- and not the quantity being predicted.
#
#   That is a hypothesis, not a conclusion. The probe measures real pre/post
#   pairs, while the loss acts between synthetic and real, and it measures whole
#   slices, while the loss is restricted to the breast via fake_masked. This run
#   settles it directly.
#
#   The same reasoning already paid off once: MSEC was inherited, never isolated,
#   and turned out redundant once per-image normalisation and the tumour mask
#   were present (row 5 improved SSIM 0.509 -> 0.520 and FRD 27.06 -> 25.83).
#
# BASELINE AND THE SINGLE CHANGE
#   Derived from berzelius_train_v32_no_msec.sh, i.e. row (5) of Table 2, with
#   one flag added: --no_vgg_loss. Row (5) is the baseline rather than row (4)
#   because MSEC is already established as redundant, so this keeps the result
#   attributable to removing VGG alone. v34_seg used the same baseline.
#
#   --no_vgg_loss is a clean off-switch, not a zeroed weight:
#     pix2pixHD_model.py:93   if not opt.no_vgg_loss: <build criterionVGG>
#     pix2pixHD_model.py:267  if not self.opt.no_vgg_loss: <compute>
#   The criterion is never constructed and the loss name is never registered, so
#   no VGG forward pass happens at all. Passing --lambda_vgg 0 instead would
#   still run VGG-19 on every step and multiply the result by zero.
#
# WHAT EACH OUTCOME MEANS
#   If the metrics hold or improve, VGG is redundant here as MSEC was, and 20 M
#   parameters plus a forward and backward pass per step come off the training
#   cost for free. That also removes the main reason to consider swapping in a
#   DINO perceptual loss: there would be nothing for it to improve on.
#
#   If LPIPS degrades while the others hold, VGG is doing exactly the job a
#   perceptual loss should and the probe's null correlation reflects its own
#   limits rather than the loss's. A DINO swap then becomes worth testing, since
#   DINO separated enhancement 12 % better on the same pairs.
#
#   Either way the answer is cheaper than the DINO route, which needs the same
#   48 h plus either a gated license or an in-domain pretraining effort.
#
# AFTER TRAINING
#   Inference and evaluation exactly as for row (5), then a paired test rather
#   than a comparison of aggregate means:
#     eval_results_v32_no_msec: MSE 1.120  LPIPS 0.125  SSIM 0.520
#                               FRD 25.83  Dice 0.624   HD95 69.8
#   LPIPS is the metric most directly at stake. Differences at this scale can
#   sit inside the noise: v3e in the diffusion line looked like a 3 % Dice
#   regression on aggregates and came out at p = 0.75 when paired.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $PROJ/mama-synth
git pull origin dev

DATAROOT=$PROJ/data_multislice_v3/train
BREAST_MASK=$PROJ/data_multislice_v3/train/mha/breast_mask
TUMOR_MASK=$PROJ/data_multislice_v3/train/mha/predicted_tumor

NUM_TUMOR=$(ls $TUMOR_MASK/*.mha 2>/dev/null | wc -l)
if [ "$NUM_TUMOR" -lt 100 ]; then
    echo "ERROR: Only $NUM_TUMOR predicted tumor masks. Run berzelius_tumor_seg_infer.sh first!"
    exit 1
fi

echo "=== v35-noVGG: row (5) without the VGG perceptual loss ==="
echo "Predicted tumor masks : $NUM_TUMOR"
echo "Changed vs v32_no_msec: --no_vgg_loss only"
echo "Probe motivating this : VGG separation 1.22, rho(distance, delta) = +0.117 (p 0.47)"
echo ""

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v35_novgg \
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
  --no_vgg_loss \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $BREAST_MASK \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo ""
echo "Done! Weights: $PROJ/checkpoints/mamasynth_v35_novgg/latest_net_G.pth"
echo ""
echo "Confirm in the log that G_VGG is absent from the loss names. If it still"
echo "appears, --no_vgg_loss did not take effect and the run is not an ablation."
echo ""
echo "=== Next: inference + evaluation, then the paired comparison ==="
echo "  v32_no_msec (row 5): MSE 1.120  LPIPS 0.125  SSIM 0.520  FRD 25.83  Dice 0.624  HD95 69.8"
echo "  v35_novgg          : <fill in from this run>"
