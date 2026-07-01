#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH -t 1:00:00
#SBATCH -J preprocess_multislice_v2
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/preprocess_v2_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/preprocess_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Preprocess images → data_multislice_v2
# Extract peak-performance slice from each DCE phase (parallel, 32 workers)
# Excludes 160 motion artifact cases.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth
git pull origin dev

echo "=== Starting multislice_v2 preprocessing ==="
echo "Patients: $PROJ/images"
echo "Segmentations: $PROJ/segmentations/automatic"
echo "Output: $PROJ/data_multislice_v2"
echo "Workers: 32"
echo ""

python src/preprocessing/preprocess_multislice_v2.py \
    --image_dir $PROJ/images \
    --seg_dir $PROJ/segmentations/automatic \
    --output_dir $PROJ/data_multislice_v2 \
    --global_stats src/preprocessing/training_pre_stats.json \
    --skip_ambiguous_shapes \
    --exclude_list src/preprocessing/motion_cases.txt \
    --workers 32

echo ""
echo "=== Done! ==="
echo "Output files:"
ls $PROJ/data_multislice_v2/mha/input/ | wc -l
