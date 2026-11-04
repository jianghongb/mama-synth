#!/bin/bash
#SBATCH -A berzelius-2026-192
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -J chest_mask
#SBATCH -o /proj/berzbiomedicalimagingkth/users/x_honji/chest_mask_%j.log
#SBATCH -e /proj/berzbiomedicalimagingkth/users/x_honji/chest_mask_%j.err
#
# Generate the chest-region loss masks for v36.
# ============================================================================
# The bare login-node `python` has no numpy, so make_chest_mask.py cannot be
# called directly; it needs the same conda environment the training scripts use.
# This wrapper activates it and passes arguments through.
#
# Works either way:
#   bash scripts/berzelius_make_chest_mask.sh                 # login node, ~minutes
#   sbatch scripts/berzelius_make_chest_mask.sh               # as a short job
#   bash scripts/berzelius_make_chest_mask.sh --rim 24        # override the band
#
# The work is CPU-only and I/O bound (reads two masks per slice, writes one), so
# the login node is adequate for a one-off run. The SBATCH header exists so it
# can be queued instead if the login node is busy.
#
# INODE COST, WORTH KNOWING BEFORE RUNNING
#   This writes 9,504 new files. The project sits at INODE_SOFT_EXCEEDED with
#   roughly 112,205 files left before the 3,000,000 hard limit locks it for every
#   user, so this consumes about 8.5 % of the remaining headroom. That is
#   affordable but not free. The alternative is deriving chest = foreground AND
#   NOT breast inside the dataloader so nothing is written, which costs a code
#   change in mha_perimage_norm_dataset.py and a new flag instead of inodes.
#   Writing the directory was chosen because it keeps v36 a single-variable
#   change against row (5): only --breast_mask_dir differs, and the dataloader
#   that produced every previous result stays untouched.

set -e

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji

export CONDA_PKGS_DIRS=$PROJ/.conda/pkgs
export PIP_CACHE_DIR=$PROJ/.pip_cache
export XDG_CACHE_HOME=$PROJ/.cache

module load Miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda activate $PROJ/envs/gan

cd $PROJ/mama-synth

# scipy.ndimage supplies the dilation and erosion. envs/gan runs training, so it
# has numpy, but scipy has not been needed by it before.
python - <<'PY' || exit 1
# importlib.util is a submodule and is not bound by `import importlib` alone,
# so it has to be imported explicitly.
import importlib.util
import sys
missing = [m for m in ("numpy", "scipy", "SimpleITK") if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: envs/gan is missing {missing}.")
    print("Install with:  pip install " + " ".join(missing))
    sys.exit(1)
import numpy, scipy, SimpleITK
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  "
      f"SimpleITK {SimpleITK.__version__}")
PY

D=$PROJ/data_multislice_v3/train/mha
for d in foreground breast_mask; do
    n=$(ls -1 $D/$d/*.mha 2>/dev/null | wc -l)
    echo "  $d: $n"
    if [ "$n" -lt 9000 ]; then
        echo "ERROR: $D/$d has only $n files, expected 9504."
        exit 1
    fi
done
echo ""

python scripts/make_chest_mask.py \
    --root $PROJ/data_multislice_v3 \
    --split train \
    "$@"

echo ""
echo "Check the reported areas before training: 'loss mask' must stay clearly"
echo "below 'foreground'. If they are close, the band has swallowed the breast"
echo "interior and --rim is too large, which would put generator B in competition"
echo "with generator A over the region A already handles."
echo ""
echo "Then:  sbatch src/submission/submission-synthesis/models/scripts/berzelius_train_v36_chest.sh"
