#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -J tumor_seg_infer
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_infer_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 2: Generate predicted tumor masks using trained Dataset940 model.
# Runs inference on BOTH train and test sets.
# Output: predicted_tumor/ directory in each split.
#
# The predicted masks will be used as input channel for the conditional GAN (v32).
# Important: GAN trains with PREDICTED masks (not GT), matching inference conditions.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
export TORCHDYNAMO_DISABLE=1

MODEL_DIR=$nnUNet_results/Dataset940_TumorSegT1w/nnUNetTrainer__nnUNetPlans__2d
DATASET_ID=940

echo "=== Generating predicted tumor masks ==="
echo "Model: $MODEL_DIR/fold_0"
echo ""

# We need to convert MHA → NIfTI for nnUNet predict, then convert results back to MHA.
# Use a temp dir for NIfTI conversion, then convert predictions back.

python -c "
import os, sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess, tempfile, shutil

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

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
    '''Run tumor seg on all MHA files in input_dir, save binary masks to output_dir.'''
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.mha'))
    print(f'  Processing {len(files)} files → {output_dir}')

    done = 0
    for f in tqdm(files):
        out_path = output_dir / f.name
        if out_path.exists():
            done += 1
            continue

        # Read MHA
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()

        # Predict
        input_arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
        while pred.ndim > 2:
            pred = pred[0]

        # Binary tumor mask
        tumor_mask = (pred > 0).astype(np.int16)

        # Save as MHA with same metadata
        if sitk.GetArrayFromImage(img).ndim == 3:
            tumor_mask = tumor_mask[np.newaxis, ...]
        out_img = sitk.GetImageFromArray(tumor_mask)
        out_img.CopyInformation(img)
        sitk.WriteImage(out_img, str(out_path))
        done += 1

    print(f'  Done: {done} masks saved')

# Process train set
print('=== Train set ===')
predict_and_save(
    '$PROJ/data_multislice_v3/train/mha/input',
    '$PROJ/data_multislice_v3/train/mha/predicted_tumor'
)

# Process test set
print('')
print('=== Test set ===')
predict_and_save(
    '$PROJ/data_multislice_v3/test/mha/input',
    '$PROJ/data_multislice_v3/test/mha/predicted_tumor'
)

print('')
print('All done!')
"

echo ""
echo "=== Complete ==="
echo "Train masks: $(ls $PROJ/data_multislice_v3/train/mha/predicted_tumor/ 2>/dev/null | wc -l) files"
echo "Test masks:  $(ls $PROJ/data_multislice_v3/test/mha/predicted_tumor/ 2>/dev/null | wc -l) files"
echo ""
echo "Next: Train conditional GAN (v32) with 3-channel input"
echo "  sbatch berzelius_train_v32_conditional.sh"
