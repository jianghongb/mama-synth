#!/bin/bash
# Build v32 conditional GAN Docker image for Grand Challenge submission
set -e

cd "$(dirname "$0")"

# Ensure weights are in place
if [ ! -f weights/latest_net_G.pth ]; then
    echo "ERROR: weights/latest_net_G.pth not found"
    echo "Copy v32 conditional weights:"
    echo "  cp weights/latest_net_G-v32_conditional.pth weights/latest_net_G.pth"
    exit 1
fi

if [ ! -d weights/breast_seg ]; then
    echo "ERROR: weights/breast_seg/ not found"
    echo "Need Dataset930 BreastDivider 2D weights"
    exit 1
fi

if [ ! -d weights/tumor_seg ]; then
    echo "ERROR: weights/tumor_seg/ not found"
    echo "Need Dataset940 T1w Tumor Seg weights"
    exit 1
fi

echo "=== Building v32 conditional GAN Docker image ==="
echo "Weights:"
echo "  GAN:       $(ls -lh weights/latest_net_G.pth | awk '{print $5}')"
echo "  Breast seg: weights/breast_seg/"
echo "  Tumor seg:  weights/tumor_seg/"

docker build -f Dockerfile.v32 -t mama-synth-v32 .

echo ""
echo "=== Build complete ==="
echo "Image: mama-synth-v32"
docker images mama-synth-v32
