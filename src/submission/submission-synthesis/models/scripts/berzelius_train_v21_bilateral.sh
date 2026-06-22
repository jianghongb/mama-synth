#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J mamasynth_v21
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v21: Trained on unilateral crops (bilateral split + chest wall removal).
# Based on v20 config but uses data_split_v4_unilateral as dataroot.
# At inference time, use berzelius_infer_bilateral.sh (split→predict→stitch).

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
pip show scipy > /dev/null 2>&1 || pip install scipy

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

cd $PROJ/mama-synth

# ============================================================
# Step 1: Generate bilateral split (reuses berzelius_prep logic)
# ============================================================
UNILAT_DIR=$PROJ/data_split_v4_unilateral/train
TRAIN_INPUT=$PROJ/data_split_v4/train/mha/input
TRAIN_GT=$PROJ/data_split_v4/train/mha/ground_truth
TRAIN_MASK=$PROJ/data_split_v4/train/mha/mask
TRAIN_BREAST=$PROJ/data_split_v4/train/mha/breast_mask_2d

# Generate breast masks if missing
if [ ! -d "$TRAIN_BREAST" ] || [ $(ls "$TRAIN_BREAST"/*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Generating 2D breast masks with Dataset920 ==="
    mkdir -p $TRAIN_BREAST
    python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage as ndi

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$TRAIN_INPUT')
output_dir = Path('$TRAIN_BREAST')
model_dir = '$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d'

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

files = sorted(input_dir.glob('*.mha'))
existing = {f.name for f in output_dir.glob('*.mha')}
todo = [f for f in files if f.name not in existing]
print(f'Total: {len(files)}, Done: {len(existing)}, Todo: {len(todo)}')

props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
         'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}

for f in tqdm(todo, desc='breast masks'):
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
    input_arr = arr[np.newaxis, np.newaxis, :, :]
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    mask = (pred > 0).astype(np.uint8)
    labeled, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labeled, range(1, n+1))
        thresh = sizes.max() * 0.1
        mask = np.zeros_like(mask)
        for i, s in enumerate(sizes):
            if s >= thresh:
                mask[labeled == (i+1)] = 1
    out = sitk.GetImageFromArray(mask.astype(np.int16))
    out.CopyInformation(img)
    sitk.WriteImage(out, str(output_dir / f.name))
print(f'Complete: {len(list(output_dir.glob(\"*.mha\")))} masks')
"
fi

# Split bilateral → unilateral with chest cut
if [ ! -d "$UNILAT_DIR/mha/input" ] || [ $(ls "$UNILAT_DIR/mha/input"/*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Splitting bilateral → unilateral (with chest cut) ==="
    python src/preprocessing/split_bilateral.py \
        --input_dir $TRAIN_INPUT \
        --gt_dir $TRAIN_GT \
        --mask_dir $TRAIN_MASK \
        --breast_mask_dir $TRAIN_BREAST \
        --output_dir $UNILAT_DIR \
        --target_size 512 \
        --pad_ratio 0.03 \
        --chest_cut
else
    echo "=== Unilateral data exists ($UNILAT_DIR), skipping split ==="
fi

CROP_COUNT=$(ls "$UNILAT_DIR/mha/input"/*.mha 2>/dev/null | wc -l)
echo "=== Unilateral training crops: $CROP_COUNT ==="

# ============================================================
# Step 2: Train pix2pixHD on unilateral crops
# ============================================================
echo "=== Starting v21 training on unilateral crops ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v21 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $UNILAT_DIR \
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
  --breast_mask_dir $UNILAT_DIR/mha/breast_mask \
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v21/latest_net_G.pth"
echo "For inference on bilateral test data: sbatch src/preprocessing/berzelius_infer_bilateral.sh 21"
