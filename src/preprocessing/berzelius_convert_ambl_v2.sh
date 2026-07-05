#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 4:00:00
#SBATCH -J convert_ambl_v2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/convert_ambl_v2_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/convert_ambl_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Convert AMBL DICOM → data_multislice_v2 format.
# Per-phase peak slice, GT = each phase itself.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth
git pull origin dev

DICOM_DIR=$PROJ/Advanced-MRI-Breast-Lesions
OUTPUT=$PROJ/data_multislice_v2/train

echo "=== Converting AMBL to data_multislice_v2 format ==="

python -c "
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dicom_dir = Path('$DICOM_DIR')
output_dir = Path('$OUTPUT')
mha_input_dir = output_dir / 'mha' / 'input'
mha_gt_dir = output_dir / 'mha' / 'ground_truth'
mha_mask_dir = output_dir / 'mha' / 'mask'

# Global stats
norm_mean, norm_std = 104.86, 215.86

def zscore(arr, mean, std):
    return ((arr - mean) / std).astype(np.float32)

def load_dicom_series(series_dir):
    dcm_files = sorted(series_dir.glob('*.dcm'))
    if not dcm_files:
        return None
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        loc = float(getattr(ds, 'SliceLocation', 0) or getattr(ds, 'InstanceNumber', 0))
        slices.append((loc, ds.pixel_array.astype(np.float32)))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=0)

def load_multiphase(series_dir):
    dcm_files = sorted(series_dir.glob('*.dcm'))
    if not dcm_files:
        return {}
    phase_slices = {}
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        tp = int(getattr(ds, 'TemporalPositionIdentifier', 1))
        loc = float(getattr(ds, 'SliceLocation', 0) or getattr(ds, 'InstanceNumber', 0))
        if tp not in phase_slices:
            phase_slices[tp] = []
        phase_slices[tp].append((loc, ds.pixel_array.astype(np.float32)))
    phases = {}
    for tp, sl in phase_slices.items():
        sl.sort(key=lambda x: x[0])
        phases[tp] = np.stack([s[1] for s in sl], axis=0)
    return phases

def process_patient(args):
    pid, mask_dir, multi_dir = args
    try:
        pre_vol = load_dicom_series(mask_dir)
        if pre_vol is None:
            return pid, 0
        phases = load_multiphase(multi_dir)
        if not phases:
            return pid, 0

        n_slices = pre_vol.shape[0]
        n_extracted = 0

        # For each phase, find its peak slice (highest mean intensity)
        for phase_num, phase_vol in phases.items():
            if phase_vol.shape[0] != n_slices:
                continue

            # Find peak slice for this phase (subtraction-based)
            enhancement = phase_vol - pre_vol
            slice_means = enhancement.mean(axis=(1, 2))
            peak_idx = int(np.argmax(slice_means))

            pre_2d = pre_vol[peak_idx]
            gt_2d = phase_vol[peak_idx]

            # Enhancement-based mask
            enh = gt_2d - pre_2d
            pos = enh[enh > 0]
            if pos.size > 100:
                thresh = float(np.percentile(pos, 90))
                mask_2d = (enh > thresh).astype(np.int16)
            else:
                mask_2d = np.zeros_like(pre_2d, dtype=np.int16)

            pre_norm = zscore(pre_2d, norm_mean, norm_std)
            gt_norm = zscore(gt_2d, norm_mean, norm_std)

            fname = f'AMBL_{pid}_p{phase_num}'

            if (mha_input_dir / f'{fname}.mha').exists():
                n_extracted += 1
                continue

            sitk.WriteImage(sitk.GetImageFromArray(pre_norm), str(mha_input_dir / f'{fname}.mha'))
            sitk.WriteImage(sitk.GetImageFromArray(gt_norm), str(mha_gt_dir / f'{fname}.mha'))
            sitk.WriteImage(sitk.GetImageFromArray(mask_2d), str(mha_mask_dir / f'{fname}.mha'))
            n_extracted += 1

        return pid, n_extracted
    except Exception as e:
        return pid, -1

# Scan DICOM dirs
logger.info('Scanning DICOM directories...')
series_dirs = [d for d in dicom_dir.iterdir() if d.is_dir()]

patient_data = {}
for sd in series_dirs:
    dcm_files = list(sd.glob('*.dcm'))
    if not dcm_files:
        continue
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        pid = str(getattr(ds, 'PatientID', 'unknown')).replace('-', '_')
        desc = str(getattr(ds, 'SeriesDescription', ''))
        if pid not in patient_data:
            patient_data[pid] = {}
        if 'MASK' in desc.upper() and 'VIBRANT' in desc.upper():
            patient_data[pid]['mask_dir'] = sd
        elif 'MULTIPHASE' in desc.upper():
            patient_data[pid]['multi_dir'] = sd
    except:
        continue

valid = [(pid, d['mask_dir'], d['multi_dir']) for pid, d in patient_data.items()
         if 'mask_dir' in d and 'multi_dir' in d]
logger.info(f'Valid patients: {len(valid)}')

# Process in parallel
total = 0
errors = 0
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_patient, args) for args in valid]
    for i, future in enumerate(as_completed(futures), 1):
        pid, n = future.result()
        if n < 0:
            errors += 1
        else:
            total += n
        if i % 50 == 0:
            logger.info(f'  Progress: {i}/{len(valid)}, {total} slices, {errors} errors')

logger.info(f'Done. {total} slices from {len(valid)} patients → {output_dir}')
"

echo ""
echo "=== Generate breast masks for new AMBL files ==="
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
echo "AMBL v2 input: $(ls $OUTPUT/mha/input/AMBL_*.mha 2>/dev/null | wc -l)"
echo "AMBL v2 breast mask: $(ls $OUTPUT/mha/breast_mask/AMBL_*.mha 2>/dev/null | wc -l)"
echo "Total v2 train: $(ls $OUTPUT/mha/input/*.mha 2>/dev/null | wc -l)"
