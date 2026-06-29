#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J v28_ucgan
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v28: v22 (n_blocks=12) + UC-GAN uncertainty.
# Generator outputs (mu, log_var). Discriminator only judges
# high-confidence regions (confidence = exp(-log_var)).
# This forces G to be accurate where it's confident, and
# honest about where it's uncertain.

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

MASK_OUTPUT=$PROJ/data_split_v4/train/mha/breast_mask_930

TRAIN_INPUT=$PROJ/data_split_v4/train/mha/input

if [ ! -d "$MASK_OUTPUT" ] || [ $(ls "$MASK_OUTPUT"/*.mha 2>/dev/null | wc -l) -lt 100 ]; then
    echo "=== Generating breast masks with Dataset930 ==="
    mkdir -p $MASK_OUTPUT

    python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/weights/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$TRAIN_INPUT')
output_dir = Path('$MASK_OUTPUT')

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$PROJ/weights/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d',
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
    # Dataset930 is pseudo-3D: expects (C, D=1, H, W) and 3D spacing
    if input_arr.ndim == 2:
        input_arr = input_arr[np.newaxis, np.newaxis]  # (1, 1, H, W)
    elif input_arr.ndim == 3:
        input_arr = input_arr[np.newaxis]  # (1, D, H, W) or (1, 1, H, W)
    sp = list(img.GetSpacing())
    spacing_3d = [1.0, sp[0], sp[1]] if len(sp) == 2 else [sp[2], sp[0], sp[1]]
    props = {
        'sitk_stuff': {
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection(),
        },
        'spacing': spacing_3d,
    }
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    mask = (pred.squeeze() > 0).astype(np.float32)
    if arr.ndim == 3 and mask.ndim == 2:
        mask = mask[np.newaxis]
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(out_path))

n_out = len([f for f in output_dir.iterdir() if f.suffix == '.mha'])
print(f'Done. {n_out} masks generated.')
"
    echo "=== Mask generation complete ==="
else
    echo "=== Masks already exist ($MASK_OUTPUT), skipping ==="
fi

# Verify masks exist before training
MASK_COUNT=$(ls "$MASK_OUTPUT"/*.mha 2>/dev/null | wc -l)
if [ "$MASK_COUNT" -lt 100 ]; then
    echo "ERROR: Only $MASK_COUNT masks in $MASK_OUTPUT. Aborting."
    echo "Check: ls $PROJ/weights/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
    exit 1
fi
echo "=== Verified: $MASK_COUNT breast masks ready ==="

echo "=== Starting v28 training (v22 + UC-GAN uncertainty) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v28 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $PROJ/data_split_v4/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --uncertainty \
  --intensity_aug \
  --resize_or_crop resize \
  --loadSize 512 \
  --fineSize 512 \
  --n_downsample_global 4 \
  --ngf 64 \
  --n_blocks_global 12 \
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
  --save_epoch_freq 5 \
  --print_freq 100 \
  --gpu_ids 0

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v28/latest_net_G.pth"
