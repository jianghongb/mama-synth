#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J prep_bilateral_split
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/prep_bilat_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/prep_bilat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Prepare data_split_v4 bilateral→unilateral splits.
#
# Pipeline:
#   1. Ensure breast masks exist (Dataset920 2D, reuse if available)
#   2. Run split_bilateral.py on training data → doubles training set
#   3. Run split_bilateral.py on test data (for unilateral model eval)
#
# Output:
#   $PROJ/data_split_v4_unilateral/train/mha/{input,ground_truth,mask,breast_mask}/
#   $PROJ/data_split_v4_unilateral/test/mha/{input,ground_truth,mask,breast_mask}/
#
# For inference (bilateral→split→model→stitch), use stitch_bilateral.py

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

pip show nnunetv2 > /dev/null 2>&1 || pip install nnunetv2 dynamic-network-architectures
pip show scipy > /dev/null 2>&1 || pip install scipy

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

cd $PROJ/mama-synth

# ============================================================
# Config
# ============================================================
TRAIN_INPUT=$PROJ/data_split_v4/train/mha/input
TRAIN_GT=$PROJ/data_split_v4/train/mha/ground_truth
TRAIN_MASK=$PROJ/data_split_v4/train/mha/mask
TRAIN_BREAST=$PROJ/data_split_v4/train/mha/breast_mask_2d

TEST_INPUT=$PROJ/data_split_v2/test/mha/input
TEST_GT=$PROJ/data_split_v2/test/mha/ground_truth
TEST_MASK=$PROJ/data_split_v2/test/mha/mask
TEST_BREAST=$PROJ/data_split_v2/test/mha/breast_mask_2d

OUTPUT_TRAIN=$PROJ/data_split_v4_unilateral/train
OUTPUT_TEST=$PROJ/data_split_v4_unilateral/test

TARGET_SIZE=512

# ============================================================
# Step 1: Generate breast masks if missing (Dataset920 2D)
# ============================================================
generate_masks() {
    local INPUT_DIR=$1
    local OUTPUT_DIR=$2
    local LABEL=$3

    if [ -d "$OUTPUT_DIR" ] && [ $(ls "$OUTPUT_DIR"/*.mha 2>/dev/null | wc -l) -ge $(ls "$INPUT_DIR"/*.mha 2>/dev/null | wc -l) ]; then
        echo "=== [$LABEL] Breast masks already complete ($OUTPUT_DIR) ==="
        return
    fi

    echo "=== [$LABEL] Generating breast masks → $OUTPUT_DIR ==="
    mkdir -p $OUTPUT_DIR

    python -c "
import os, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage as ndi

os.environ['nnUNet_raw'] = '$PROJ/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '$PROJ/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '$PROJ/nnUNet_results'
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

input_dir = Path('$INPUT_DIR')
output_dir = Path('$OUTPUT_DIR')
model_dir = '$PROJ/nnUNet_results/Dataset920_BreastSeg2D/nnUNetTrainer__nnUNetPlans__2d'

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
    perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, allow_tqdm=False)
predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

files = sorted(input_dir.glob('*.mha'))
existing = {f.name for f in output_dir.glob('*.mha')}
todo = [f for f in files if f.name not in existing]
print(f'[$LABEL] Total: {len(files)}, Done: {len(existing)}, Todo: {len(todo)}')

props = {'sitk_stuff': {'spacing': (1.,1.,1.), 'origin': (0.,0.,0.),
         'direction': (1.,0.,0.,0.,1.,0.,0.,0.,1.)}, 'spacing': [1.,1.,1.]}

for f in tqdm(todo, desc='$LABEL masks'):
    img = sitk.ReadImage(str(f))
    arr = sitk.GetArrayFromImage(img).astype(np.float32).squeeze()
    if arr.ndim != 2:
        continue
    input_arr = arr[np.newaxis, np.newaxis, :, :]
    pred = predictor.predict_single_npy_array(input_arr, props, None, None, False)
    while pred.ndim > 2:
        pred = pred[0]
    mask = (pred > 0).astype(np.uint8)
    # Keep components >= 10% of largest
    labeled, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labeled, range(1, n+1))
        thresh = sizes.max() * 0.1
        mask = np.zeros_like(mask)
        for i, s in enumerate(sizes):
            if s >= thresh:
                mask[labeled == (i+1)] = 1
    out = sitk.GetImageFromArray(mask.astype(np.int16))
    out.CopyInformation(img)
    sitk.WriteImage(out, str(output_dir / f.name))

print(f'[$LABEL] Complete: {len(list(output_dir.glob(\"*.mha\")))} masks')
"
}

generate_masks "$TRAIN_INPUT" "$TRAIN_BREAST" "train"
generate_masks "$TEST_INPUT" "$TEST_BREAST" "test"

# ============================================================
# Step 2: Split bilateral → unilateral (training)
# ============================================================
echo "=== Splitting training data (bilateral → unilateral) ==="
python src/preprocessing/split_bilateral.py \
    --input_dir $TRAIN_INPUT \
    --gt_dir $TRAIN_GT \
    --mask_dir $TRAIN_MASK \
    --breast_mask_dir $TRAIN_BREAST \
    --output_dir $OUTPUT_TRAIN \
    --target_size $TARGET_SIZE \
    --pad_ratio 0.03 \
    --chest_cut

TRAIN_COUNT=$(ls $OUTPUT_TRAIN/mha/input/*.mha 2>/dev/null | wc -l)
echo "=== Training: $TRAIN_COUNT unilateral crops ==="

# ============================================================
# Step 3: Split bilateral → unilateral (test)
# ============================================================
echo "=== Splitting test data (bilateral → unilateral) ==="
python src/preprocessing/split_bilateral.py \
    --input_dir $TEST_INPUT \
    --gt_dir $TEST_GT \
    --mask_dir $TEST_MASK \
    --breast_mask_dir $TEST_BREAST \
    --output_dir $OUTPUT_TEST \
    --target_size $TARGET_SIZE \
    --pad_ratio 0.03 \
    --chest_cut

TEST_COUNT=$(ls $OUTPUT_TEST/mha/input/*.mha 2>/dev/null | wc -l)
echo "=== Test: $TEST_COUNT unilateral crops ==="

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  data_split_v4_unilateral preparation done"
echo "=========================================="
echo "  Train: $OUTPUT_TRAIN ($TRAIN_COUNT cases)"
echo "  Test:  $OUTPUT_TEST ($TEST_COUNT cases)"
echo ""
echo "  Expected: ~5622 train (2811×2), ~300 test (150×2)"
echo ""
echo "  To train:"
echo "    --dataroot $OUTPUT_TRAIN"
echo "    --breast_mask_dir $OUTPUT_TRAIN/mha/breast_mask"
echo ""
echo "  For inference on bilateral test data, use stitch_bilateral.py:"
echo "    from stitch_bilateral import BilateralSplitter"
echo "    splitter = BilateralSplitter(target_size=$TARGET_SIZE)"
echo "=========================================="
