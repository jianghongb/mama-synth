#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_DIR="${1:-test/input}"
OUTPUT_DIR="${2:-test/output_docker}"

mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

echo "=== Running inference ==="
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"

if [ "${USE_GPU:-1}" = "0" ]; then
    GPU_FLAG=""
    echo "  Device: CPU"
else
    GPU_FLAG="--gpus all"
    echo "  Device: GPU"
fi

docker run --rm \
    $GPU_FLAG \
    -v "$(realpath $INPUT_DIR)":/input:ro \
    -v "$(realpath $OUTPUT_DIR)":/output \
    mamasynth

echo ""
echo "=== Output ==="
ls -la "$OUTPUT_DIR/images/synthetic-contrast-dce-mri-slice-breast/" 2>/dev/null || echo "No output found!"
