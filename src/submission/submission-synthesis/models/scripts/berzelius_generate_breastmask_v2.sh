#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2:00:00
#SBATCH -J breastmask_v2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/breastmask_v2_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/breastmask_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Generate breast masks for data_multislice_v2 using Dataset920 (2D nnUNet).
# Batch GPU inference + multi-threaded postprocessing.
#
# Input:  data_multislice_v2/mha/input/*.mha
# Output: data_multislice_v2/mha/breast_mask/*.mha

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

INPUT_DIR=$PROJ/data_multislice_v2/mha/input
OUTPUT_DIR=$PROJ/data_multislice_v2/mha/breast_mask
MODEL_DIR=$PROJ/weights/nnUNet_results/Dataset930_BreastDivider2D/nnUNetTrainer__nnUNetPlans__2d

mkdir -p $OUTPUT_DIR

echo "=== Generating breast masks for data_multislice_v2 ==="
echo "Input:  $INPUT_DIR ($(ls $INPUT_DIR | wc -l) files)"
echo "Output: $OUTPUT_DIR"
echo "Model:  Dataset930 BreastDivider 2D"
echo ""

python -c "
import os, sys
import numpy as np
import SimpleITK as sitk
import torch
import cv2
from scipy import ndimage as ndi
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ.setdefault('nnUNet_raw', '$PROJ/nnUNet_raw')
os.environ.setdefault('nnUNet_preprocessed', '$PROJ/nnUNet_preprocessed')
os.environ.setdefault('nnUNet_results', '$PROJ/nnUNet_results')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_DIR = Path('$INPUT_DIR')
OUTPUT_DIR = Path('$OUTPUT_DIR')
MODEL_DIR = '$MODEL_DIR'
N_POSTPROCESS_WORKERS = 8
BATCH_SIZE = 32  # batch GPU inference

# ─── Load model ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

predictor = nnUNetPredictor(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False,
)
predictor.initialize_from_trained_model_folder(
    MODEL_DIR, use_folds=(0,), checkpoint_name='checkpoint_final.pth',
)
print('Model loaded.')

props = {
    'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
                   'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)},
    'spacing': [1., 1., 1.],
}

# ─── Gather files to process ─────────────────────────────────
all_files = sorted(INPUT_DIR.glob('*.mha'))
todo = [f for f in all_files if not (OUTPUT_DIR / f.name).exists()]
print(f'Total: {len(all_files)}, already done: {len(all_files)-len(todo)}, to process: {len(todo)}')

if not todo:
    print('Nothing to do!')
    sys.exit(0)

# ─── Postprocessing function ─────────────────────────────────
def postprocess_mask(pred_2d):
    '''Morphological cleanup of raw nnUNet prediction → binary breast mask.'''
    breast_mask = (pred_2d > 0).astype(np.uint8)
    if breast_mask.sum() == 0:
        return breast_mask

    # Drop small components (<10% of largest)
    labeled, n_comp = ndi.label(breast_mask)
    if n_comp > 1:
        sizes = ndi.sum(breast_mask, labeled, range(1, n_comp + 1))
        max_size = sizes.max()
        breast_mask = np.zeros_like(breast_mask, dtype=np.uint8)
        for idx, s in enumerate(sizes):
            if s >= max_size * 0.1:
                breast_mask[labeled == (idx + 1)] = 1

    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)

    # Fill holes
    h, w = breast_mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    inv = breast_mask.copy()
    cv2.floodFill(inv, flood, (0, 0), 1)
    breast_mask = (breast_mask | (1 - inv)).astype(np.uint8)

    return breast_mask


def save_mask(fname, mask_2d, ref_img):
    '''Save mask as MHA with same metadata as reference image.'''
    arr = ref_img.GetArrayFromImage if hasattr(ref_img, 'GetArrayFromImage') else None
    out_arr = mask_2d.astype(np.int16)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    if ref_arr.ndim == 3:
        out_arr = out_arr[np.newaxis, ...]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, str(OUTPUT_DIR / fname))

# ─── Process in batches ──────────────────────────────────────
print(f'Processing {len(todo)} files in batches of {BATCH_SIZE}...')

n_done = 0
for batch_start in range(0, len(todo), BATCH_SIZE):
    batch_files = todo[batch_start:batch_start + BATCH_SIZE]

    # Load batch
    batch_data = []
    batch_refs = []
    for f in batch_files:
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
        batch_data.append(arr)
        batch_refs.append(img)

    # GPU inference one by one (nnUNet doesn't support true batch for different sizes)
    batch_preds = []
    for arr in batch_data:
        input_arr = arr[np.newaxis, np.newaxis, :, :]
        pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
        while pred.ndim > 2:
            pred = pred[0]
        batch_preds.append(pred)

    # Parallel postprocessing + saving
    def process_one(idx):
        mask = postprocess_mask(batch_preds[idx])
        save_mask(batch_files[idx].name, mask, batch_refs[idx])

    with ThreadPoolExecutor(max_workers=N_POSTPROCESS_WORKERS) as executor:
        futures = [executor.submit(process_one, i) for i in range(len(batch_files))]
        for fut in as_completed(futures):
            fut.result()  # raise exceptions if any

    n_done += len(batch_files)
    if n_done % 500 == 0 or n_done == len(todo):
        print(f'  Progress: {n_done}/{len(todo)} ({n_done/len(todo)*100:.1f}%)')

print(f'Done! {n_done} breast masks saved to {OUTPUT_DIR}')
"

echo ""
echo "=== Complete ==="
echo "Output: $OUTPUT_DIR ($(ls $OUTPUT_DIR | wc -l) files)"
