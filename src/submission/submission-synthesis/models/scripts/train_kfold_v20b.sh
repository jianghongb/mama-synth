#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J kfold_v20b_%a
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20b_%A_%a.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20b_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#SBATCH --array=0-4
#
# K-fold v20b: v20 config + Dataset920 breast mask (correctly generated)
# Train: DUKE + ISPY2 + YUNNAN + LABREAST (all sources)
# Test: DUKE + ISPY2 + YUNNAN only (5% per fold)
# Breast mask: Dataset920 (distilled 2D model)

FOLD=${SLURM_ARRAY_TASK_ID}
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

SPLITS_CSV=$PROJ/mama-synth/kfold_splits_v2.csv
FOLD_DIR=$PROJ/kfold_v20b/fold_${FOLD}

# ============================================================
# Step 0: Generate splits if not exists
# ============================================================
if [ ! -f "$SPLITS_CSV" ]; then
    echo "Generating kfold splits..."
    python $PROJ/mama-synth/src/preprocessing/generate_kfold_splits_v2.py \
        --data_dir $PROJ/data_split_v4/train/mha/input \
        --mask_dir $PROJ/data_split_v4/train/mha/mask \
        --output_csv $SPLITS_CSV \
        --k 5 --test_ratio 0.05 \
        --train_only_sources LABREAST
fi

# ============================================================
# Step 1: Setup fold directories
# ============================================================
if [ ! -d "$FOLD_DIR/train/mha/input" ] || [ $(ls "$FOLD_DIR/train/mha/input/"*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "Setting up fold $FOLD directories..."
    rm -rf $FOLD_DIR
    python $PROJ/mama-synth/src/preprocessing/setup_kfold_dir.py \
        --splits_csv $SPLITS_CSV \
        --data_dir $PROJ/data_split_v4/train/mha \
        --output_dir $FOLD_DIR \
        --fold $FOLD
fi

echo "Fold $FOLD: train=$(ls $FOLD_DIR/train/mha/input/*.mha | wc -l), test=$(ls $FOLD_DIR/test/mha/input/*.mha | wc -l)"

# ============================================================
# Step 2: Generate breast masks with Dataset920
# ============================================================
MASK_DIR=$FOLD_DIR/train/mha/breast_mask_920
TRAIN_INPUT=$FOLD_DIR/train/mha/input
SEG_MODEL=$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d/fold_0

if [ ! -d "$MASK_DIR" ] || [ $(ls "$MASK_DIR"/*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Generating Dataset920 breast masks ==="
    mkdir -p $MASK_DIR

    python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
os.environ['nnUNet_raw'] = '/tmp/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/tmp/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/tmp/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

BREAST_LABELS = {1, 2, 3}
device = torch.device('cuda')

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$SEG_MODEL', use_folds=(0,), checkpoint_name='checkpoint_final.pth')
print('Dataset920 model loaded')

input_dir = Path('$TRAIN_INPUT')
output_dir = Path('$MASK_DIR')
props = {'sitk_stuff': {'spacing': (1,1,1), 'origin': (0,0,0),
          'direction': (1,0,0,0,1,0,0,0,1)}, 'spacing': [1,1,1]}

files = sorted(input_dir.glob('*.mha'))
print(f'Generating masks for {len(files)} files...')
for f in tqdm(files):
    out_path = output_dir / f.name
    if out_path.exists():
        continue
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
    input_arr = arr[np.newaxis, np.newaxis, :, :]
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    mask = np.isin(pred, list(BREAST_LABELS)).astype(np.int16)
    if img.GetDimension() == 3 or sitk.GetArrayFromImage(img).ndim == 3:
        mask = mask[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(mask)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))
print(f'Done: {len(list(output_dir.glob(\"*.mha\")))} masks')
"
    echo "=== Mask generation complete: $(ls $MASK_DIR/*.mha | wc -l) masks ==="
fi

# ============================================================
# Step 3: Train
# ============================================================
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name kfold_v20b_f${FOLD} \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $FOLD_DIR/train \
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
  --breast_mask_dir $MASK_DIR \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo "=== Fold $FOLD done. Weights: $PROJ/checkpoints/kfold_v20b_f${FOLD}/latest_net_G.pth ==="
