#!/bin/bash
# Build ensemble Docker for GC submission (v20 kfold, 5 models)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Preparing ensemble weights ==="
mkdir -p weights

# Copy/download 5 fold weights (rename to fold_N_net_G.pth)
for i in 0 1 2 3 4; do
    src="weights_kfold_v20/fold_${i}/latest_net_G.pth"
    dst="weights/fold_${i}_net_G.pth"
    if [ -f "$src" ]; then
        cp -f "$src" "$dst"
        echo "  fold_${i}: OK ($(du -h $dst | cut -f1))"
    else
        echo "  fold_${i}: MISSING — download from Berzelius:"
        echo "    rsync -avP x_honji@berzelius.nsc.liu.se:/proj/berzbiomedicalimagingkth/users/x_honji/checkpoints/kfold_v20_f${i}/latest_net_G.pth $src"
    fi
done

# Verify
n=$(ls weights/fold_*_net_G.pth 2>/dev/null | wc -l)
if [ "$n" -lt 3 ]; then
    echo "ERROR: Need at least 3 fold weights. Only found $n."
    exit 1
fi

touch models/__init__.py

echo ""
echo "=== Building Docker (ensemble) ==="
docker build -f Dockerfile.ensemble -t mamasynth_ensemble .
echo "Done. Image: mamasynth_ensemble"
