#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J lab_mask
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/lab_mask_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/lab_mask_%j.err
#
# Generate breast masks for LA-Breast data using Dataset930 nnUNet.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

INPUT_DIR=$PROJ/data_multislice_v2/train/mha/input
OUTPUT_DIR=$PROJ/data_multislice_v2/train/mha/breast_mask
MODEL_DIR=$PROJ/weights/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d

echo "=== Generating breast masks for LAB files ==="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$INPUT_DIR')
output_dir = Path('$OUTPUT_DIR')
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda')
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$MODEL_DIR', use_folds=(0,), checkpoint_name='checkpoint_final.pth')
print('Model loaded')

# Only process LAB files without existing masks
mha_files = sorted(f for f in input_dir.glob('LAB_*.mha') if not (output_dir / f.name).exists())
print(f'Generating masks for {len(mha_files)} files...')

for i, mha in enumerate(tqdm(mha_files)):
    img = sitk.ReadImage(str(mha))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    input_arr = arr[np.newaxis, np.newaxis] if arr.ndim == 2 else arr[np.newaxis]

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

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(output_dir / mha.name))

print(f'Done. {len(mha_files)} masks generated.')
"

echo "=== Mask generation complete ==="
echo "Total breast masks:"
ls $OUTPUT_DIR/LAB_*.mha 2>/dev/null | wc -l
