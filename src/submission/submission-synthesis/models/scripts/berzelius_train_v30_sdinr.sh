#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J v30_sdinr
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v30: Semi-Disentangled INR-inspired Generator
# =============================================
# Core idea: Decompose synthesis into anatomy-lock + gated enhancement
#   output = pre + gate(x,y) * enhancement(x,y)
#
# - SharedEncoder extracts features from pre-contrast
# - EnhancementDecoder (heavy, with skip connections) predicts intensity delta
# - GateDecoder (lightweight, smooth) predicts spatial attention [0,1]
# - Anatomy is preserved by identity + gate gating
# - breast_mask hard-constraints gate=0 outside breast
#
# Expected improvements over v21:
# - No background hallucination (gate * breast_mask kills it)
# - Better tumor-region fidelity (all capacity focused on enhancement)
# - Lower MSE variance (anatomy lock eliminates intensity drift)

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

MASK_OUTPUT=$PROJ/data_multislice_v2/mha/breast_mask

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

echo "=== Starting v30: Semi-Disentangled INR-inspired Generator ==="
echo "Architecture: SharedEncoder + EnhancementDecoder + GateDecoder"
echo "Formula: output = pre + gate * enhancement"
echo "Base params from v26: ngf=96, data_multislice_v2, tumor=10, MSSC=50"

python train_semi_disentangled.py \
  --name mamasynth_v30_sdinr \
  --dataroot $PROJ/data_multislice_v2/train \
  --checkpoints_dir $PROJ/checkpoints \
  --breast_mask_dir $MASK_OUTPUT \
  --ngf 96 \
  --n_downsampling 4 \
  --n_encoder_blocks 9 \
  --n_enhance_blocks 3 \
  --n_gate_blocks 2 \
  --loadSize 512 \
  --batchSize 8 \
  --num_workers 8 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0002 \
  --lambda_enhance_l1 10.0 \
  --lambda_anatomy 5.0 \
  --lambda_gate_sparsity 0.5 \
  --lambda_gate_tv 0.1 \
  --lambda_tumor 10.0 \
  --lambda_vgg 10.0 \
  --lambda_gan 1.0 \
  --lambda_feat 10.0 \
  --lambda_mssc 50.0 \
  --mssc_levels 3 \
  --num_D 2 \
  --n_layers_D 3 \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v30_sdinr/"
echo ""
echo "=== Inference command ==="
echo "python infer_semi_disentangled.py \\"
echo "  --weights $PROJ/checkpoints/mamasynth_v30_sdinr/latest_net_G.pth \\"
echo "  --input_dir $PROJ/data_multislice_v2/test/mha/input \\"
echo "  --output_dir $PROJ/predictions_v30_sdinr \\"
echo "  --breast_seg_model $PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d \\"
echo "  --ngf 96 --n_encoder_blocks 9 --n_enhance_blocks 3 --n_gate_blocks 2 \\"
echo "  --save_gate"
