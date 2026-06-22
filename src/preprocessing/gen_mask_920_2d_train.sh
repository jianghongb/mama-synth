#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J gen_mask_2d
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_2d_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_2d_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Generate 2D breast masks for all training data using distilled Dataset920 model.
# Output: $PROJ/data_split_v4/train/mha/breast_mask_2d/*.mha

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

TRAIN_INPUT=$PROJ/data_split_v4/train/mha/input
MASK_OUTPUT=$PROJ/data_split_v4/train/mha/breast_mask_2d
mkdir -p $MASK_OUTPUT

echo "Input: $TRAIN_INPUT"
echo "Output: $MASK_OUTPUT"
echo "Model: $PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d"

# Verify model exists
if [ ! -f "$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth" ]; then
    echo "ERROR: Model not found. Listing available paths:"
    find $PROJ -path "*Dataset920*" -type f -name "*.pth" 2>/dev/null
    find $PROJ -path "*Dataset920*" -type f -name "dataset.json" 2>/dev/null
    exit 1
fi

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
model_dir = '$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d'

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')
print('Model loaded successfully')

files = sorted(input_dir.glob('*.mha'))
existing = set(f.name for f in output_dir.glob('*.mha'))
todo = [f for f in files if f.name not in existing]
print(f'Total: {len(files)}, Already done: {len(existing)}, Todo: {len(todo)}')

for f in tqdm(todo):
    try:
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        was_3d = False
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
            was_3d = True
        if arr.ndim != 2:
            print(f'SKIP {f.name}: shape={arr.shape} (not 2D)')
            continue
        input_arr = arr[np.newaxis, np.newaxis]
        props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
                 'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}
        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
        mask = (pred > 0).astype(np.float32).squeeze()
        if was_3d:
            mask = mask[np.newaxis]
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(img)
        sitk.WriteImage(mask_img, str(output_dir / f.name))
    except Exception as e:
        print(f'ERROR {f.name}: {e}')
        continue

print(f'Done. {len(list(output_dir.glob(\"*.mha\")))} masks in {output_dir}')
"

echo "=== Complete ==="
ls $MASK_OUTPUT | wc -l
echo "mask files generated"
