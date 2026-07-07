#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00
#SBATCH -J tumor_seg_infer
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_infer_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 2: Generate predicted tumor masks using trained Dataset940 model.
# Supports parallel execution: pass SPLIT=train or SPLIT=test
#
# Usage:
#   sbatch --export=SPLIT=train berzelius_tumor_seg_infer.sh
#   sbatch --export=SPLIT=test berzelius_tumor_seg_infer.sh
#   (or without SPLIT to process both sequentially)
#
# Supports resume: skips already-generated masks.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
export TORCHDYNAMO_DISABLE=1

MODEL_DIR=$nnUNet_results/Dataset940_TumorSegT1w/nnUNetTrainer__nnUNetPlans__2d

# Determine which splits to process
SPLIT=${SPLIT:-all}
echo "=== Tumor Seg Inference (split=$SPLIT) ==="
echo "Model: $MODEL_DIR/fold_0"
echo ""

python -c "
import os, sys
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

device = torch.device('cuda')
predictor = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False,
)
predictor.initialize_from_trained_model_folder(
    '$MODEL_DIR', use_folds=(0,), checkpoint_name='checkpoint_final.pth',
)
print('Model loaded.')

props = {
    'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
                   'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)},
    'spacing': [1., 1., 1.],
}

def predict_and_save(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.mha'))
    todo = [f for f in files if not (output_dir / f.name).exists()]
    print(f'  {input_dir.parent.parent.name}: {len(files)} total, {len(files)-len(todo)} done, {len(todo)} to process')

    for f in tqdm(todo, desc=input_dir.parent.parent.name):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()

        input_arr = arr[np.newaxis, np.newaxis, :, :]
        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
        while pred.ndim > 2:
            pred = pred[0]

        tumor_mask = (pred > 0).astype(np.int16)
        if sitk.GetArrayFromImage(img).ndim == 3:
            tumor_mask = tumor_mask[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(tumor_mask)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(output_dir / f.name))

    print(f'  Done: {len(files)} total masks in {output_dir}')

split = '$SPLIT'
if split in ('train', 'all'):
    print('=== Train set ===')
    predict_and_save(
        '$PROJ/data_multislice_v3/train/mha/input',
        '$PROJ/data_multislice_v3/train/mha/predicted_tumor'
    )

if split in ('test', 'all'):
    print('')
    print('=== Test set ===')
    predict_and_save(
        '$PROJ/data_multislice_v3/test/mha/input',
        '$PROJ/data_multislice_v3/test/mha/predicted_tumor'
    )

print('')
print('Done!')
"

echo ""
echo "=== Complete (split=$SPLIT) ==="
echo "Train masks: $(ls $PROJ/data_multislice_v3/train/mha/predicted_tumor/ 2>/dev/null | wc -l)"
echo "Test masks:  $(ls $PROJ/data_multislice_v3/test/mha/predicted_tumor/ 2>/dev/null | wc -l)"
