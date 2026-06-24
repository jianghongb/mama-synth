#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J breastdiv_distill
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/breastdiv_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/breastdiv_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# BreastDivider distillation pipeline:
# 1. Run BreastDivider 3D on all pre-contrast volumes (DUKE + ISPY2 + Yunnan)
# 2. Extract 2D slice masks using report.csv slice indices
# 3. Output: 2D breast masks ready for nnUNet 2D student training

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

cd $PROJ/mama-synth

# Output directories
MASK_3D_DIR=$PROJ/breastdivider_masks_3d
MASK_2D_DIR=$PROJ/data_split_v4/train/mha/breast_mask_bd
mkdir -p $MASK_3D_DIR $MASK_2D_DIR

python -c "
import os, sys, torch, numpy as np, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import csv

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'

# ============================================================
# Step 1: Collect all 3D volumes + their slice indices
# ============================================================

# Read report.csv to get slice indices
report_files = [
    '$PROJ/data_split_v4/train/report.csv',
    '$PROJ/data_split_v4/test/report.csv',
    '$PROJ/yunnan_v2/report.csv',
]
slice_map = {}  # patient_id -> selected_slice
for rpath in report_files:
    if not Path(rpath).exists():
        print(f'Warning: {rpath} not found, skipping')
        continue
    with open(rpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['patient_id']
            # Handle different column names
            if 'selected_slice' in row:
                slice_map[pid] = int(row['selected_slice'])
            elif 'slice_idx' in row:
                slice_map[pid] = int(row['slice_idx'])
print(f'Loaded slice indices for {len(slice_map)} patients')
print(f'Loaded slice indices for {len(slice_map)} patients')

# Collect 3D volume paths
volumes = {}  # patient_id -> path to pre-contrast 3D volume

# All datasets use same structure: images/<PID>/<PID>_0000.nii.gz
images_dir = Path('$PROJ/images')
for pdir in sorted(images_dir.iterdir()):
    if not pdir.is_dir():
        continue
    pid = pdir.name
    pre = pdir / f'{pid}_0000.nii.gz'
    if pre.exists():
        volumes[pid] = pre

# Filter: only process cases we have slice indices for
to_process = {pid: path for pid, path in volumes.items() if pid in slice_map}
print(f'Found {len(volumes)} volumes, {len(to_process)} have slice indices')

# Also include cases without slice index (we'll extract middle slice)
for pid, path in volumes.items():
    if pid not in to_process:
        to_process[pid] = path

print(f'Total to process: {len(to_process)}')

# ============================================================
# Step 2: Run BreastDivider on all volumes
# ============================================================

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

device = torch.device('cuda')
predictor = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(
    '$PROJ/BreastDividerModel',
    use_folds=('all',), checkpoint_name='checkpoint_final.pth')

mask_2d_dir = Path('$MASK_2D_DIR')
mask_3d_dir = Path('$MASK_3D_DIR')

processed = 0
skipped = 0

for pid, vol_path in tqdm(to_process.items(), desc='BreastDivider'):
    out_2d = mask_2d_dir / f'{pid}.mha'
    if out_2d.exists():
        skipped += 1
        continue

    try:
        # Load 3D volume
        img = sitk.ReadImage(str(vol_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)

        # Run BreastDivider
        input_arr = arr[np.newaxis]  # (1, D, H, W)
        props = {
            'sitk_stuff': {
                'spacing': img.GetSpacing(),
                'origin': img.GetOrigin(),
                'direction': img.GetDirection(),
            },
            'spacing': list(img.GetSpacing()),
        }
        pred_3d = predictor.predict_single_npy_array(input_arr, props, None, None, False)
        # pred_3d: (D, H, W) with labels 0, 1, 2

        # Get slice index
        if pid in slice_map:
            slice_idx = slice_map[pid]
        else:
            # Fallback: middle slice
            slice_idx = pred_3d.shape[0] // 2

        # Clamp slice index
        slice_idx = min(slice_idx, pred_3d.shape[0] - 1)

        # Extract 2D mask: left(1) + right(2) → binary breast mask
        mask_2d = (pred_3d[slice_idx] > 0).astype(np.float32)

        # Save as .mha matching the 2D input format
        # Need to get the 2D slice spatial info from the preprocessed data
        input_2d_path = Path('$PROJ/data_split_v4/train/mha/input') / f'{pid}.mha'
        if input_2d_path.exists():
            ref_img = sitk.ReadImage(str(input_2d_path))
            ref_arr = sitk.GetArrayFromImage(ref_img)
            # Resize mask to match 2D input dimensions
            from scipy.ndimage import zoom
            target_shape = ref_arr.squeeze().shape
            if mask_2d.shape != target_shape:
                zoom_factors = (target_shape[0] / mask_2d.shape[0],
                               target_shape[1] / mask_2d.shape[1])
                mask_2d = zoom(mask_2d, zoom_factors, order=0)  # nearest for mask
            # Match ndim
            if ref_arr.ndim == 3:
                mask_2d = mask_2d[np.newaxis]
            out_img = sitk.GetImageFromArray(mask_2d.astype(np.float32))
            out_img.CopyInformation(ref_img)
        else:
            # No 2D reference, save raw
            out_img = sitk.GetImageFromArray(mask_2d[np.newaxis].astype(np.float32))

        sitk.WriteImage(out_img, str(out_2d))
        processed += 1

    except Exception as e:
        print(f'ERROR {pid}: {e}')
        continue

print(f'Done! Processed: {processed}, Skipped (existing): {skipped}')
print(f'Output: {mask_2d_dir} ({len(list(mask_2d_dir.glob(\"*.mha\")))} files)')
"

echo ""
echo "=== BreastDivider distillation complete ==="
echo "2D masks: $MASK_2D_DIR"
echo "Count: $(ls $MASK_2D_DIR/*.mha 2>/dev/null | wc -l)"
