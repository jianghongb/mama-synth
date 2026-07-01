#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J distill_infer
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/distill_infer_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/distill_infer_%j.err
#
# Run Dataset932 inference on batch N (0-5), ~250 cases per batch
# Usage: sbatch distill_batch_infer.sh <batch_index>

BATCH=${1:-0}
BATCH_SIZE=252
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

python -c "
import os, numpy as np, nibabel as nib, torch
from pathlib import Path
from tqdm import tqdm

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

model_dir = '$PROJ/exp4x_for_maia/Dataset932/nnUNetTrainer__nnUNetPlans__3d_fullres'
images_dir = Path('$PROJ/images')
output_dir = Path('$PROJ/breast_seg_932_predictions')
output_dir.mkdir(exist_ok=True)

device = torch.device('cuda')
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

cases = sorted([d for d in images_dir.iterdir() if d.is_dir()])
batch = $BATCH
batch_size = $BATCH_SIZE
start = batch * batch_size
end = min(start + batch_size, len(cases))
batch_cases = cases[start:end]
print(f'Batch {batch}: cases {start}-{end-1} ({len(batch_cases)} cases)')

for case_dir in tqdm(batch_cases):
    name = case_dir.name
    out_path = output_dir / f'{name}.nii.gz'
    if out_path.exists():
        continue
    p0_path = case_dir / f'{name}_0000.nii.gz'
    p1_path = case_dir / f'{name}_0001.nii.gz'
    if not p0_path.exists() or not p1_path.exists():
        continue
    p0_nii = nib.load(str(p0_path))
    p0 = p0_nii.get_fdata(dtype=np.float32)
    p1 = nib.load(str(p1_path)).get_fdata(dtype=np.float32)
    d_early = p1 - p0
    d_late = d_early
    input_4ch = np.stack([p0, p1, d_early, d_late])
    props = {
        'sitk_stuff': {'spacing': tuple(p0_nii.header.get_zooms()[:3]),
                       'origin': (0.,0.,0.),
                       'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)},
        'spacing': list(p0_nii.header.get_zooms()[:3]),
    }
    pred = predictor.predict_single_npy_array(input_4ch, props, None, None, False)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), p0_nii.affine, p0_nii.header), str(out_path))

print(f'Batch {batch} done.')
"
