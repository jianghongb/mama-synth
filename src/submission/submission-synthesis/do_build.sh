#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Stage weights into weights/ directory
echo "=== Staging weights ==="
mkdir -p weights

# GAN weights (use v16 for v18 submission)
GAN_SRC="${GAN_WEIGHTS:-weights_v16/latest_net_G.pth}"
if [ ! -f "$GAN_SRC" ]; then
    echo "ERROR: GAN weights not found: $GAN_SRC"
    echo "Set GAN_WEIGHTS=/path/to/latest_net_G.pth or place in weights_v16/"
    exit 1
fi
cp -f "$GAN_SRC" weights/latest_net_G.pth
echo "  GAN: $GAN_SRC → weights/latest_net_G.pth ($(du -h weights/latest_net_G.pth | cut -f1))"

# Refiner weights (use v18 if available, fallback to v17)
REFINER_SRC="${REFINER_WEIGHTS:-weights_v18/refiner_latest.pth}"
if [ ! -f "$REFINER_SRC" ]; then
    REFINER_SRC="weights_v17/refiner_latest.pth"
fi
if [ -f "$REFINER_SRC" ]; then
    cp -f "$REFINER_SRC" weights/refiner_latest.pth
    echo "  Refiner: $REFINER_SRC → weights/refiner_latest.pth ($(du -h weights/refiner_latest.pth | cut -f1))"
else
    echo "  WARNING: No refiner weights found. Will run GAN-only mode."
    rm -f weights/refiner_latest.pth
fi

# Ensure models/__init__.py exists
touch models/__init__.py

echo ""
echo "=== Building Docker image ==="
docker build -t mamasynth .
echo ""
echo "Done. Image: mamasynth ($(docker image inspect mamasynth --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo '?'))"
