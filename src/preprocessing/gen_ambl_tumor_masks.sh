#!/bin/bash
#
# Generate enhancement-based tumor masks for AMBL files in data_multislice_v3.
# Run on login node (no GPU needed, ~1-2 min).
#
# Usage: bash src/preprocessing/gen_ambl_tumor_masks.sh

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

python -c "
import numpy as np, SimpleITK as sitk
from pathlib import Path

input_dir = Path('$PROJ/data_multislice_v3/train/mha/input')
gt_dir = Path('$PROJ/data_multislice_v3/train/mha/ground_truth')
mask_dir = Path('$PROJ/data_multislice_v3/train/mha/mask')

ambl_files = sorted(input_dir.glob('AMBL_*.mha'))
existing = sum(1 for f in ambl_files if (mask_dir / f.name).exists())
to_process = [f for f in ambl_files if not (mask_dir / f.name).exists()]
print(f'AMBL files: {len(ambl_files)} total, {existing} already have masks, {len(to_process)} to generate')

for i, f in enumerate(to_process):
    inp = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).squeeze()
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_dir / f.name))).squeeze()
    enh = gt - inp
    pos = enh[enh > 0]
    if pos.size > 100:
        thresh = float(np.percentile(pos, 90))
        mask = (enh > thresh).astype(np.int16)
    else:
        mask = np.zeros_like(inp, dtype=np.int16)
    img = sitk.ReadImage(str(f))
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    sitk.WriteImage(mask_img, str(mask_dir / f.name))
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{len(to_process)}')

print(f'Done. Generated {len(to_process)} masks.')
print(f'Total AMBL masks: {sum(1 for f in ambl_files if (mask_dir / f.name).exists())}')
"
