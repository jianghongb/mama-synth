#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J mamasynth_v16
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v16: v14 config but with ensemble breast mask (ResEncUNetL fold_0 OR PlainConvUNet fold_4)
# Quick test: 50 epochs only (niter=25 + niter_decay=25)
# Data: data_split_v4 axial-only (2528 cases)
# Breast mask: ensemble of two models (union)
# Intensity aug: yes

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show torchmetrics > /dev/null 2>&1 || pip install torchmetrics
pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

# Use the PlainConvUNet 2D fold_4 model for breast masks

# Create filtered dataset (axial only, exclude ISPY1/NACT/AMBL)
FILTERED=$PROJ/data_split_v4_axial/train/mha
if [ ! -d "$FILTERED/input" ]; then
    mkdir -p $FILTERED/input $FILTERED/ground_truth $FILTERED/mask
    for sub in input ground_truth mask; do
        src=$PROJ/data_split_v4/train/mha/$sub
        if [ -d "$src" ]; then
            for f in $src/*.mha; do
                name=$(basename "$f")
                case "$name" in
                    ISPY1_*|NACT_*|AMBL-*) continue ;;
                    *) ln -s "$f" "$FILTERED/$sub/$name" ;;
                esac
            done
        fi
    done
    echo "Filtered dataset (axial only): $(ls $FILTERED/input/ | wc -l) cases"
fi

# Generate breast masks with ensemble (OR of two models)
BREAST_MASK_DIR=$PROJ/data_split_v4_axial/train/mha/breast_mask
if [ ! -d "$BREAST_MASK_DIR" ]; then
    echo "Generating ensemble breast masks..."
    # Model 1: ResEncUNetL fold_0 (checkpoint_best)
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $FILTERED/input \
        --output_dir ${BREAST_MASK_DIR}_resenc \
        --model_dir $PROJ/weights/Dataset910_BreastSegNet/nnUNetTrainer__nnUNetResEncUNetLPlans__2d \
        --fold 0 \
        --checkpoint checkpoint_best.pth
    # Model 2: PlainConvUNet fold_4 (checkpoint_best)
    python $PROJ/mama-synth/src/submission/submission-synthesis/models/generate_breast_masks.py \
        --input_dir $FILTERED/input \
        --output_dir ${BREAST_MASK_DIR}_plain \
        --model_dir $PROJ/weights/breast_seg/nnUNetTrainer__nnUNetPlans__2d \
        --fold 4 \
        --checkpoint checkpoint_best.pth
    # Combine with OR + post-processing
    mkdir -p $BREAST_MASK_DIR
    python -c "
import os, numpy as np, SimpleITK as sitk, cv2
from pathlib import Path
from scipy import ndimage as ndi

def postprocess_mask(mask):
    mask = mask.astype(np.uint8)
    # Drop small components (<10% of largest)
    labeled, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labeled, range(1, n+1))
        max_size = sizes.max()
        mask = np.zeros_like(mask, dtype=np.uint8)
        for idx, s in enumerate(sizes):
            if s >= max_size * 0.1:
                mask[labeled == (idx+1)] = 1
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Fill holes
    h, w = mask.shape
    flood = np.zeros((h+2, w+2), np.uint8)
    inv = mask.copy()
    cv2.floodFill(inv, flood, (0,0), 1)
    mask = (mask | (1 - inv)).astype(np.uint8)
    # Handle multiple components
    labeled2, n2 = ndi.label(mask)
    if n2 > 1:
        sizes2 = ndi.sum(mask, labeled2, range(1, n2+1))
        top2_idx = np.argsort(sizes2)[-2:] + 1
        centroids = ndi.center_of_mass(mask, labeled2, top2_idx)
        c1, c2 = centroids[0], centroids[1]
        dy, dx = abs(c1[0]-c2[0]), abs(c1[1]-c2[1])
        if dx > dy:
            mask_top2 = np.isin(labeled2, top2_idx).astype(np.uint8)
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
            temp = mask_top2.copy()
            for _ in range(30):
                temp = cv2.dilate(temp, dk)
                if ndi.label(temp)[1] <= 1: break
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
            temp = cv2.erode(temp, ek)
            mask = ((mask > 0) | (temp > 0)).astype(np.uint8)
        else:
            largest = top2_idx[np.argmax([sizes2[i-1] for i in top2_idx])]
            mask = (labeled2 == largest).astype(np.uint8)
    return mask

d1 = Path('${BREAST_MASK_DIR}_resenc')
d2 = Path('${BREAST_MASK_DIR}_plain')
out = Path('$BREAST_MASK_DIR')
for f in sorted(d1.glob('*.mha')):
    m1 = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).squeeze()
    f2 = d2 / f.name
    if f2.exists():
        m2 = sitk.GetArrayFromImage(sitk.ReadImage(str(f2))).squeeze()
        combined = ((m1 > 0) | (m2 > 0)).astype(np.uint8)
    else:
        combined = (m1 > 0).astype(np.uint8)
    final = postprocess_mask(combined)
    out_arr = final.astype(np.int16)
    if sitk.GetArrayFromImage(sitk.ReadImage(str(f))).ndim == 3:
        out_arr = out_arr[np.newaxis,...]
    img = sitk.GetImageFromArray(out_arr)
    img.CopyInformation(sitk.ReadImage(str(f)))
    sitk.WriteImage(img, str(out / f.name))
print(f'Ensemble masks: {len(list(out.glob(\"*.mha\")))} files')
"
fi

cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v16 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $PROJ/data_split_v4_axial/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --intensity_aug \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 9 \
  --norm instance \
  --batchSize 8 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0002 \
  --lambda_feat 10 \
  --lambda_gan 1.0 \
  --tumor_weight 10 \
  --lambda_mssc 50 \
  --mssc_levels 3 \
  --lambda_ssim 0 \
  --lambda_vgg 10 \
  --num_D 2 \
  --n_layers_D 3 \
  --breast_mask_dir $BREAST_MASK_DIR \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0
