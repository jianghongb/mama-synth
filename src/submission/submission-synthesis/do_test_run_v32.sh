#!/bin/bash
# Test v32 Docker container locally
set -e

cd "$(dirname "$0")"

# Use test input or specify custom
INPUT_DIR="${1:-$(pwd)/test/input}"
OUTPUT_DIR="${2:-$(pwd)/test/output_v32}"

mkdir -p "$OUTPUT_DIR"

echo "=== Testing v32 Docker container ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# Run with GPU if available, otherwise CPU
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "GPU: enabled"
else
    GPU_FLAG=""
    echo "GPU: not available, using CPU"
fi

docker run --rm \
    $GPU_FLAG \
    -v "$INPUT_DIR":/input \
    -v "$OUTPUT_DIR":/output \
    mama-synth-v32

echo ""
echo "=== Test complete ==="
echo "Output files:"
find "$OUTPUT_DIR" -name "*.mha" -exec ls -lh {} \;
