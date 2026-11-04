#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH -J v36_chest
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v36-chest: a second generator for the region the first one never sees
# ============================================================================
# WHY
#   Every GAN version so far confines its losses to the breast
#   (pix2pixHD_model.py:224-228 multiplies fake and real by the mask), and
#   inference then copies everything outside from the pre-contrast slice
#   (inference.py:180). Measured on the 299 test slices, that leaves most of the
#   error in a region no model touches:
#
#     squared error inside the breast    25.5 %
#     squared error outside the breast    74.5 %
#     MSE if the outside were perfect    0.821 -> 0.210
#
#   The chest wall and heart do enhance: mean |post - pre| outside the breast is
#   0.34 against 0.87 inside, so 39 % of the inside magnitude, not zero. Copying
#   the pre-contrast slice discards all of it.
#
#   Flow matching already showed the two-generator route pays. Its composite plus
#   lesion boost took HD95 from 71 to 62.7 and Dice from 0.61 to 0.622 as pure
#   post-processing, matching Anguinus's reported 62.52. The GAN has not had the
#   composite half of that at all.
#
# WHAT THIS TRAINS
#   Generator B, supervised on chest = foreground AND NOT breast, plus a 16 px
#   band straddling the breast edge so both generators see the seam that
#   merge_two_generators.py blends over. Measured on test slices: breast is
#   13.3 % of the frame, this mask is 46.0 %, and the overlap is the band alone,
#   so B does not compete for the breast interior that A already handles.
#
#   Generator A stays exactly as it is. This is deliberately not the alternative
#   of retraining one model on the whole foreground: that would dilute lesion
#   supervision across 53 % of the frame instead of 13 %, putting the Dice 0.624
#   and HD95 69.8 that are already in hand at risk for an unmeasured gain.
#
# THE ONE CHANGE
#   Derived from berzelius_train_v32_no_msec.sh, row (5), with --breast_mask_dir
#   pointed at chest_mask instead of breast_mask. Nothing else differs. The mask
#   enters only at pix2pixHD_model.py:224-228 and at the per-image z-score in
#   mha_perimage_norm_dataset.py:63-73, so redirecting it moves both the loss and
#   the normalisation statistics to the chest region, which is what a chest
#   specialist should be normalised on.
#
# PREREQUISITE
#   scripts/make_chest_mask.py must have run on the train split first:
#     python scripts/make_chest_mask.py --split train --rim 16
#   It needs foreground/ and breast_mask/, both 9,504 files on Berzelius.
#
# AFTER TRAINING
#   Inference with this checkpoint, then composite:
#     breast = generator A (row 5), chest = this model, background = pre-contrast
#   via DSB_for_chanllenge/scripts/merge_two_generators.py --blend_sigma 5.
#   Score the composite, not this model alone: on its own it will look poor on
#   every tumour metric, because it is not trained to synthesise lesions.
#
#   Judge with compare_paired.py over the 299 cases rather than aggregate means.
#   Baseline eval_results_v32_no_msec: MSE 1.120, LPIPS 0.125, SSIM 0.520,
#   FRD 25.83, Dice 0.624, HD95 69.8.
#
#   Note when reporting FRD: the identity baseline scores 9.90 against 25.83 for
#   this GAN and 20.3 for FM, so FRD runs opposite to synthesis quality in this
#   setup and should not be quoted as evidence of improvement.

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
CHEST_MASK=$DATAROOT/mha/chest_mask
TUMOR_MASK=$DATAROOT/mha/predicted_tumor

NUM_CHEST=$(ls $CHEST_MASK/*.mha 2>/dev/null | wc -l)
if [ "$NUM_CHEST" -lt 9000 ]; then
    echo "ERROR: only $NUM_CHEST chest masks in $CHEST_MASK (expected 9504)."
    echo "Run first:  python scripts/make_chest_mask.py --split train --rim 16"
    exit 1
fi

echo "=== v36-chest: generator B, loss on foreground minus breast plus rim ==="
echo "Chest masks  : $NUM_CHEST"
echo "Tumour masks : $(ls $TUMOR_MASK/*.mha 2>/dev/null | wc -l)"
echo "Changed vs v32_no_msec: --breast_mask_dir -> chest_mask"
echo ""

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v36_chest \
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
  --lambda_vgg 10 \
  --lambda_gan 1.0 \
  --tumor_weight 10 \
  --lambda_mssc 0 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $CHEST_MASK \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo ""
echo "Done. Weights: $PROJ/checkpoints/mamasynth_v36_chest/latest_net_G.pth"
echo ""
echo "This model is not meant to be scored alone. Fetch it, run inference, then"
echo "composite with generator A before evaluating:"
echo "  breast = predictions_v32_no_msec, chest = this model, bg = pre-contrast"
