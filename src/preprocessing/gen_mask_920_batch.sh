#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J gen_mask_920
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_920_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/gen_mask_920_%j.err
#
# Generate breast masks with Dataset920 in batches
# Usage: sbatch gen_mask_920_batch.sh <batch_index>
# ~2528 cases, ~8 cases/sec → ~5min total, but split for safety

BATCH=${1:-0}
BATCH_SIZE=1300
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show opencv-python-headless > /dev/null 2>&1 || pip install opencv-python-headless

FILTERED=$PROJ/data_split_v4_axial/train/mha
BREAST_MASK_DIR=$PROJ/data_split_v4_axial/train/mha/breast_mask_920
mkdir -p $BREAST_MASK_DIR

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

python -c "
import os, sys, numpy as np, SimpleITK as sitk, torch, cv2
from pathlib import Path
from scipy import ndimage as ndi

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
sys.path.insert(0, '$PROJ/mama-synth/src/submission/submission-synthesis/models')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

model_dir = '$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d'
input_dir = Path('$FILTERED/input')
output_dir = Path('$BREAST_MASK_DIR')

device = torch.device('cuda')
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=device, verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_best.pth')

files = sorted(input_dir.glob('*.mha'))
batch = $BATCH
batch_size = $BATCH_SIZE
start = batch * batch_size
end = min(start + batch_size, len(files))
batch_files = files[start:end]
print(f'Batch {batch}: {start}-{end-1} ({len(batch_files)} files)')

props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
         'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}

from tqdm import tqdm
for f in tqdm(batch_files):
    out_path = output_dir / f.name
    if out_path.exists():
        continue
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
    input_arr = arr[np.newaxis, np.newaxis, :, :]
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    breast_mask = (pred > 0).astype(np.uint8)

    # Post-processing
    labeled, n = ndi.label(breast_mask)
    if n > 1:
        sizes = ndi.sum(breast_mask, labeled, range(1, n+1))
        max_size = sizes.max()
        breast_mask = np.zeros_like(breast_mask, dtype=np.uint8)
        for idx, s in enumerate(sizes):
            if s >= max_size * 0.1:
                breast_mask[labeled == (idx+1)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
    h, w = breast_mask.shape
    flood = np.zeros((h+2, w+2), np.uint8)
    inv = breast_mask.copy()
    cv2.floodFill(inv, flood, (0,0), 1)
    breast_mask = (breast_mask | (1 - inv)).astype(np.uint8)
    labeled2, n2 = ndi.label(breast_mask)
    if n2 > 1:
        sizes2 = ndi.sum(breast_mask, labeled2, range(1, n2+1))
        top2_idx = np.argsort(sizes2)[-2:] + 1
        centroids = ndi.center_of_mass(breast_mask, labeled2, top2_idx)
        c1, c2 = centroids[0], centroids[1]
        dy, dx = abs(c1[0]-c2[0]), abs(c1[1]-c2[1])
        if dx > dy:
            mask_top2 = np.isin(labeled2, top2_idx).astype(np.uint8)
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
            temp = mask_top2.copy()
            for _ in range(30):
                temp = cv2.dilate(temp, dk)
                if ndi.label(temp)[1] <= 1: break
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
            temp = cv2.erode(temp, ek)
            breast_mask = ((breast_mask > 0) | (temp > 0)).astype(np.uint8)
        else:
            largest = top2_idx[np.argmax([sizes2[i-1] for i in top2_idx])]
            breast_mask = (labeled2 == largest).astype(np.uint8)

    out_arr = breast_mask.astype(np.int16)
    if sitk.GetArrayFromImage(img).ndim == 3:
        out_arr = out_arr[np.newaxis,...]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, str(out_path))

print(f'Batch {batch} done. Total masks: {len(list(output_dir.glob(\"*.mha\")))}')
"
