#!/bin/bash
#SBATCH -A berzelius-2025-422
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_download
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_download_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/DSB_for_chanllenge/distill_download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
set -e
PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
WORK=$PROJ/distill_breastdivider
mkdir -p $WORK

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan
pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub

# HF_TOKEN should be set via 'hf auth login' on the compute node before running
huggingface-cli login

echo "=== Downloading BreastDividerDataset (batch1 only, ~220GB) ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Bubenpo/BreastDividerDataset', repo_type='dataset', local_dir='$WORK/BreastDividerDataset', max_workers=16, allow_patterns=['imagesTr_batch1/*', 'labelsTr_batch1/*', 'dataset.json', 'breastdivider_id_mapping.csv'])
"
echo "=== Download complete ==="
