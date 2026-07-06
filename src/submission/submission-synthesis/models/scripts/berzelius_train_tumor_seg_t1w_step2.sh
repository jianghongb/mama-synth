#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -J tumor_seg_train
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_train_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/tumor_seg_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Step 2 (GPU): nnUNet plan + preprocess + train
# Run AFTER step1_prep.sh has completed data conversion.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $XDG_CACHE_HOME

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

export nnUNet_raw=$PROJ/nnUNet_raw
export nnUNet_preprocessed=$PROJ/nnUNet_preprocessed
export nnUNet_results=$PROJ/nnUNet_results

# Maximize GPU utilization
export nnUNet_n_proc_DA=16          # data augmentation workers
export nnUNet_def_n_proc=16         # preprocessing workers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation

DATASET_ID=940
DATASET_NAME="Dataset${DATASET_ID}_TumorSegT1w"

# Verify data exists
NUM_IMAGES=$(ls $nnUNet_raw/$DATASET_NAME/imagesTr/*.nii.gz 2>/dev/null | wc -l)
if [ "$NUM_IMAGES" -lt 100 ]; then
    echo "ERROR: Only $NUM_IMAGES images found. Run step1_prep.sh first!"
    exit 1
fi
echo "=== Dataset940_TumorSegT1w: $NUM_IMAGES training images ==="

echo ""
echo "=== Plan + Preprocess ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 2d --verify_dataset_integrity

echo ""
echo "=== Training fold 0 (2D) ==="
nnUNetv2_train $DATASET_ID 2d 0 --npz

echo ""
echo "=== Complete ==="
echo "Model: $nnUNet_results/$DATASET_NAME/nnUNetTrainer__nnUNetPlans__2d/fold_0/"
echo ""
echo "Next: generate predicted tumor masks for GAN training"
echo "  nnUNetv2_predict -d $DATASET_ID -c 2d -f 0 -i <input_nifti> -o <output>"
