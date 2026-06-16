#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J gen_mask_932
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_932_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_932_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Job 1 of 方案A拆分: 只跑 Dataset932 3D breast mask 生成 (纯GPU推理)
# GPU 持续满载, 不会被 kill

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures

export nnUNet_raw="/tmp/nnUNet_raw"
export nnUNet_preprocessed="/tmp/nnUNet_preprocessed"
export nnUNet_results="/tmp/nnUNet_results"
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

IMAGE_DIR=$PROJ/images
OUTPUT_DIR=$PROJ/breast_masks_3d_932
MODEL_DIR=$PROJ/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres

mkdir -p $OUTPUT_DIR

# Pure GPU inference: generate 3D breast masks for each patient
python -c "
import os, sys, torch, numpy as np, nibabel as nib
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault('nnUNet_raw', '/tmp/nnUNet_raw')
os.environ.setdefault('nnUNet_preprocessed', '/tmp/nnUNet_preprocessed')
os.environ.setdefault('nnUNet_results', '/tmp/nnUNet_results')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

image_dir = Path('$IMAGE_DIR')
output_dir = Path('$OUTPUT_DIR')
model_dir = '$MODEL_DIR'

# Init predictor (GPU)
device = torch.device('cuda')
predictor = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device,
    verbose=False, allow_tqdm=True)
predictor.initialize_from_trained_model_folder(
    model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')
print('Model loaded on GPU')

patients = sorted([d for d in os.listdir(str(image_dir)) if (image_dir / d).is_dir()])
print(f'Processing {len(patients)} patients...')

for pid in tqdm(patients):
    out_path = output_dir / f'{pid}.nii.gz'
    if out_path.exists():
        continue

    pat_dir = image_dir / pid
    phase_files = sorted(pat_dir.glob('*.nii.gz'))
    if len(phase_files) < 2:
        continue

    # Load phases
    p0_img = nib.load(str(phase_files[0]))
    p0 = p0_img.get_fdata(dtype=np.float32)
    p1 = nib.load(str(phase_files[1])).get_fdata(dtype=np.float32)
    d_early = p1 - p0

    # d_late: last phase - p1
    if len(phase_files) >= 5:
        p4 = nib.load(str(phase_files[-1])).get_fdata(dtype=np.float32)
    else:
        p4 = nib.load(str(phase_files[-1])).get_fdata(dtype=np.float32)
    d_late = p4 - p1

    # Stack: (4, D, H, W)
    input_arr = np.stack([p0, p1, d_early, d_late])

    props = {
        'sitk_stuff': {
            'spacing': tuple(p0_img.header.get_zooms()[:3]),
            'origin': (0, 0, 0),
            'direction': (1, 0, 0, 0, 1, 0, 0, 0, 1),
        },
        'spacing': list(p0_img.header.get_zooms()[:3]),
    }

    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)

    # breast mask: label 1+2+3
    breast_mask = np.isin(pred, [1, 2, 3]).astype(np.uint8)
    nib.save(nib.Nifti1Image(breast_mask, p0_img.affine), str(out_path))

print(f'Done. Masks saved to {output_dir}')
"

echo "=== Breast mask generation complete ==="
echo "Output: $OUTPUT_DIR"
ls $OUTPUT_DIR | wc -l
echo "files generated"
