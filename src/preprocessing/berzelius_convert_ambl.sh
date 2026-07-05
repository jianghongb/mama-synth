#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -J convert_ambl
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/convert_ambl_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/convert_ambl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Convert Advanced-MRI-Breast-Lesions DICOM → MHA (data_multislice_v3 format).
# Then generate breast masks with Dataset930.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth
git pull origin dev

DICOM_DIR=$PROJ/Advanced-MRI-Breast-Lesions
OUTPUT=$PROJ/data_multislice_v3/train

echo "=== Step 1: Convert DICOM → MHA ==="
python src/preprocessing/convert_ambl_dicom.py \
    --dicom_dir $DICOM_DIR \
    --output_dir $OUTPUT \
    --global_stats src/preprocessing/training_pre_stats.json \
    --extra_slices 2 \
    --workers 16

echo ""
echo "AMBL files generated:"
ls $OUTPUT/mha/input/AMBL_*.mha 2>/dev/null | wc -l

echo ""
echo "=== Step 2: Generate breast masks for AMBL files ==="
export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

MODEL_DIR = '$PROJ/weights/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d'
device = torch.device('cuda')
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(MODEL_DIR, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

input_dir = Path('$OUTPUT/mha/input')
output_dir = Path('$OUTPUT/mha/breast_mask')
output_dir.mkdir(parents=True, exist_ok=True)

mha_files = sorted(f for f in input_dir.glob('AMBL_*.mha') if not (output_dir / f.name).exists())
print(f'Generating breast masks for {len(mha_files)} AMBL files...')

for mha in tqdm(mha_files):
    img = sitk.ReadImage(str(mha))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    input_arr = arr[np.newaxis, np.newaxis] if arr.ndim == 2 else arr[np.newaxis]
    sp = list(img.GetSpacing())
    spacing_3d = [1.0, sp[0], sp[1]] if len(sp) == 2 else [sp[2], sp[0], sp[1]]
    props = {
        'sitk_stuff': {'spacing': img.GetSpacing(), 'origin': img.GetOrigin(), 'direction': img.GetDirection()},
        'spacing': spacing_3d,
    }
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    mask = (pred.squeeze() > 0).astype(np.float32)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(output_dir / mha.name))

print('Done.')
"

echo ""
echo "=== Summary ==="
echo "AMBL input files: $(ls $OUTPUT/mha/input/AMBL_*.mha 2>/dev/null | wc -l)"
echo "AMBL breast masks: $(ls $OUTPUT/mha/breast_mask/AMBL_*.mha 2>/dev/null | wc -l)"
echo "Total train input: $(ls $OUTPUT/mha/input/*.mha 2>/dev/null | wc -l)"
