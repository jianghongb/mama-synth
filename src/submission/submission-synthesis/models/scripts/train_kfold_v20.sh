#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH -J kfold_v20_%a
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_%A_%a.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/kfold_v20_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#SBATCH --array=0-4
#
# K-fold cross-validation with v20 config (Dataset920 2D breast mask).
# Step 1: Generate breast masks for this fold's train set using Dataset920
# Step 2: Train with v20 config
# Submit with: sbatch train_kfold_v20.sh

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

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

KFOLD_DIR=$PROJ/kfold
FOLD_DIR=$KFOLD_DIR/fold_${FOLD}
SPLITS_CSV=$PROJ/mama-synth/kfold_splits.csv

# ============================================================
# Step 0: Setup fold directories (skip LABREAST)
# ============================================================
if [ ! -d "$FOLD_DIR/train/mha/input" ] || [ $(ls "$FOLD_DIR/train/mha/input/"*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Setting up fold $FOLD directories ==="
    python $PROJ/mama-synth/src/preprocessing/setup_kfold_dir.py \
        --splits_csv $SPLITS_CSV \
        --data_dir $PROJ/data_split_v4/train/mha \
        --output_dir $FOLD_DIR \
        --fold $FOLD
fi

echo "Fold $FOLD: train=$(ls $FOLD_DIR/train/mha/input/*.mha | wc -l), test=$(ls $FOLD_DIR/test/mha/input/*.mha | wc -l)"

# ============================================================
# Step 1: Generate Dataset920 breast masks for train set
# ============================================================
MASK_OUTPUT=$FOLD_DIR/train/mha/breast_mask_2d
TRAIN_INPUT=$FOLD_DIR/train/mha/input

if [ ! -d "$MASK_OUTPUT" ] || [ $(ls "$MASK_OUTPUT"/*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Generating Dataset920 breast masks for fold $FOLD train ==="
    mkdir -p $MASK_OUTPUT

    python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$TRAIN_INPUT')
output_dir = Path('$MASK_OUTPUT')

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d/fold_0',
    use_folds=(0,), checkpoint_name='checkpoint_final.pth')

mha_files = sorted(input_dir.glob('*.mha'))
print(f'Generating masks for {len(mha_files)} files...')

for mha in tqdm(mha_files):
    out_path = output_dir / mha.name
    if out_path.exists():
        continue
    img = sitk.ReadImage(str(mha))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    input_arr = arr[np.newaxis] if arr.ndim == 2 else arr
    props = {
        'sitk_stuff': {
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection(),
        },
        'spacing': list(img.GetSpacing()),
    }
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    mask = (pred > 0).astype(np.float32)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(out_path))

print(f'Done. {len(list(output_dir.glob(\"*.mha\")))} masks generated.')
"
    echo "=== Mask generation complete ==="
else
    echo "=== Masks already exist for fold $FOLD, skipping ==="
fi

# ============================================================
# Step 2: Train with v20 config
# ============================================================
echo "=== Starting v20 kfold training (fold $FOLD) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name kfold_v20_f${FOLD} \
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
  --breast_mask_dir $MASK_OUTPUT \
  --save_epoch_freq 50 \
  --print_freq 100 \
  --gpu_ids 0

echo "=== Fold $FOLD done! Weights: $PROJ/checkpoints/kfold_v20_f${FOLD}/latest_net_G.pth ==="
