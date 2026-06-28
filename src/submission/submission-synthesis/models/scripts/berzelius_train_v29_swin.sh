#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J v29_swin
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# v29: v22 (n_blocks=12) + Swin Transformer bottleneck.
# Inserts 2 Swin blocks (W-MSA + SW-MSA, window_size=8) in the middle
# of the 12 ResBlocks. Captures long-range anatomical dependencies
# (e.g. bilateral breast symmetry) at 32×32 feature resolution.

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
os.environ['nnUNet_results'] = '$PROJ/distill_breastdivider/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$TRAIN_INPUT')
output_dir = Path('$MASK_OUTPUT')

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$PROJ/distill_breastdivider/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d/fold_0',
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

print(f'Done. {len(list(output_dir.glob("*.mha")))} masks generated.')
"
    echo "=== Mask generation complete ==="
else
    echo "=== Masks already exist ($MASK_OUTPUT), skipping ==="
fi

echo "=== Starting v29 training (v22 + Swin bottleneck) ==="
cd $PROJ/mama-synth/src/submission/submission-synthesis/models

python train.py \
  --name mamasynth_v29 \
  --model pix2pixHD \
  --dataset_mode mha \
  --dataroot $PROJ/data_split_v4/train \
  --checkpoints_dir $PROJ/checkpoints \
  --label_nc 0 \
  --input_nc 1 \
  --output_nc 1 \
  --no_instance \
  --residual_mode \
  --swin_bottleneck \
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

echo "Done! Weights: $PROJ/checkpoints/mamasynth_v29/latest_net_G.pth"
