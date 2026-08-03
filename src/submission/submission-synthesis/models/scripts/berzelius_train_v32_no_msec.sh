#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v32_nomsec
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v32-noMSEC: MSEC ablation for the MAMA-SYNTH paper (reviewer request)
# =====================================================================
# PURPOSE
#   Reviewer 1 asked for an ablation isolating the MSEC loss contribution.
#   This script is a byte-for-byte copy of berzelius_train_v32_conditional.sh
#   with EXACTLY ONE changed hyperparameter:
#
#       --lambda_mssc 50   →   --lambda_mssc 0
#
#   Everything else (data, architecture, all other loss weights, schedule,
#   augmentation, seed-relevant flags) is identical, so the resulting delta
#   is attributable to MSEC alone.
#
# WHY lambda_mssc 0 IS A CLEAN OFF-SWITCH
#   models/pix2pixHD_model.py:82   self.use_mssc_loss = self.lambda_mssc > 0
#   models/pix2pixHD_model.py:273  if self.use_mssc_loss: <compute MSSC>
#   With lambda_mssc=0 the branch is skipped entirely — the term contributes
#   no value and no gradient (it is not merely multiplied by zero).
#
# DO NOT CHANGE ANYTHING ELSE IN THIS FILE.
#   Any additional difference from v32_conditional would confound the ablation
#   and make the result unusable as evidence. Note in particular that the
#   existing lambda_mssc=100 runs (v31_auroc, v33_vflip_auroc) are NOT valid
#   MSEC ablations because they also use --input_nc 1 instead of 3.
#
# AFTER TRAINING — run inference + evaluation with the SAME settings used to
# produce the paper's row (3) "+ 3ch conditional", i.e. no blending/TTA, so the
# comparison target is eval_results_v32_conditional (MSE 1.1595, LPIPS 0.1165,
# FRD 28.4561, Dice 0.6138).

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

cd $PROJ/mama-synth
git pull origin dev

DATAROOT=$PROJ/data_multislice_v3/train
BREAST_MASK=$PROJ/data_multislice_v3/train/mha/breast_mask
TUMOR_MASK=$PROJ/data_multislice_v3/train/mha/predicted_tumor

# Verify predicted tumor masks exist
NUM_TUMOR=$(ls $TUMOR_MASK/*.mha 2>/dev/null | wc -l)
if [ "$NUM_TUMOR" -lt 100 ]; then
    echo "ERROR: Only $NUM_TUMOR predicted tumor masks. Run berzelius_tumor_seg_infer.sh first!"
    exit 1
fi
echo "=== v32-noMSEC: Conditional GAN (pre + breast + tumor), MSEC DISABLED ==="
echo "Predicted tumor masks: $NUM_TUMOR"
echo "lambda_mssc = 0  (ablation; v32_conditional baseline uses 50)"
echo ""

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v32_no_msec \
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
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $BREAST_MASK \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo ""
echo "Done! Weights: $PROJ/checkpoints/mamasynth_v32_no_msec/latest_net_G.pth"
echo ""
echo "=== Next: inference + evaluation (match row (3) of paper Table 2) ==="
echo "Compare against eval_results_v32_conditional:"
echo "  v32_conditional (lambda_mssc=50): MSE 1.1595  LPIPS 0.1165  FRD 28.4561  Dice 0.6138"
echo "  v32_no_msec     (lambda_mssc=0 ): <fill in from this run>"
