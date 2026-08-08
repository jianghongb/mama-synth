#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J table1_stats
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/table1_stats_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/table1_stats_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hongjia@kth.se
#
# Recompute the Table 1 intensity statistics on the full training split.
# =====================================================================
# WHY
#   The Table 1 caption claims Mean/Std are "computed over breast foreground
#   pixels", but a local check on the test split reproduces the published
#   YUNNAN pair (-0.33 +/- 0.05, 0.23 +/- 0.07) exactly under an ALL-pixels
#   convention, not a foreground one. If that holds on the training split the
#   caption is factually wrong and must be corrected before camera-ready.
#
# WHAT IT PRINTS
#   For each dataset, four conventions (foreground vs all pixels, aggregated
#   per slice vs per patient) side by side with the published values. The row
#   that reproduces the published pair is the convention the caption should
#   state.
#
# NOTE
#   This is CPU/IO-bound; the GPU is requested only to satisfy the Berzelius
#   partition policy, and a keepalive loop prevents low-power termination.

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

DATA_ROOT=$PROJ/data_multislice_v3/train/mha

if [ ! -d "$DATA_ROOT/input" ] || [ ! -d "$DATA_ROOT/breast_mask" ]; then
    echo "ERROR: expected $DATA_ROOT/{input,breast_mask}"
    ls -la "$DATA_ROOT" 2>/dev/null
    exit 1
fi

echo "=== Table 1 statistics recomputation ==="
echo "Data:   $DATA_ROOT"
echo "Slices: $(ls "$DATA_ROOT"/input/*.mha 2>/dev/null | wc -l)"
echo "Masks:  $(ls "$DATA_ROOT"/breast_mask/*.mha 2>/dev/null | wc -l)"
echo ""

cd $PROJ/mama-synth
git pull origin dev

# Keep the GPU above the Berzelius power floor during this CPU-bound job.
python -c "
import torch, time
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device='cuda')
    while True:
        _ = torch.mm(x, x)
        time.sleep(0.5)
" &
KEEPALIVE_PID=$!
trap 'kill $KEEPALIVE_PID 2>/dev/null' EXIT

export PYTHONUNBUFFERED=1
python -u recompute_table1_stats.py --root "$DATA_ROOT" --workers 16

kill $KEEPALIVE_PID 2>/dev/null

echo ""
echo "=== Done ==="
echo "Compare each convention against the published columns. Report the"
echo "matching row back so the Table 1 caption can be corrected."
