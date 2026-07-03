#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH -t 1:00:00
#SBATCH -J prep_msv3
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/preprocess_v3_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/preprocess_v3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Preprocess data_multislice_v3: peak slice ± 2, global peak phase GT.
# Then split into train/test and generate breast masks.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth
git pull origin dev

OUTPUT=$PROJ/data_multislice_v3

echo "=== Step 1: Preprocess (peak ± 2 slices, global peak GT) ==="
python src/preprocessing/preprocess_multislice_v3.py \
    --image_dir $PROJ/images \
    --seg_dir $PROJ/segmentations/automatic \
    --output_dir $OUTPUT \
    --global_stats src/preprocessing/training_pre_stats.json \
    --extra_slices 2 \
    --min_mask_area 50 \
    --skip_ambiguous_shapes \
    --exclude_list src/preprocessing/motion_cases.txt \
    --workers 32

echo ""
echo "Total slices:"
ls $OUTPUT/mha/input/*.mha 2>/dev/null | wc -l

echo ""
echo "=== Step 2: Generate train/test split CSV (5% test per source) ==="
python src/preprocessing/gen_split_csv_v2.py \
    --data_dir $OUTPUT \
    --test_ratio 0.05 \
    --seed 42

echo ""
echo "=== Step 3: Split into train/test directories ==="
python src/preprocessing/split_data_multislice_v2.py \
    --data_dir $OUTPUT \
    --splits_csv $OUTPUT/train_test_split.csv

echo ""
echo "=== Step 4: Generate breast masks (Dataset930) ==="
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
print('Breast seg model loaded')

for split in ['train', 'test']:
    input_dir = Path('$OUTPUT') / split / 'mha' / 'input'
    output_dir = Path('$OUTPUT') / split / 'mha' / 'breast_mask'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mha_files = sorted(f for f in input_dir.glob('*.mha') if not (output_dir / f.name).exists())
    print(f'{split}: generating {len(mha_files)} masks...')
    
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

print('Breast masks done.')
"

echo ""
echo "=== Final Summary ==="
echo "Train input: $(ls $OUTPUT/train/mha/input/*.mha 2>/dev/null | wc -l)"
echo "Train mask:  $(ls $OUTPUT/train/mha/breast_mask/*.mha 2>/dev/null | wc -l)"
echo "Test input:  $(ls $OUTPUT/test/mha/input/*.mha 2>/dev/null | wc -l)"
echo "Test mask:   $(ls $OUTPUT/test/mha/breast_mask/*.mha 2>/dev/null | wc -l)"
echo "=== Done ==="
