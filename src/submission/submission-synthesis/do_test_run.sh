#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

bash "$SCRIPT_DIR/do_build.sh"

rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

INPUT_DIR="$SCRIPT_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo "ERROR: No .mha in $INPUT_DIR"
    exit 1
fi

USE_GPU="${USE_GPU:-1}"
if [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpus device=0"
else
    GPU_FLAG=""
fi

docker run --rm --network=none $GPU_FLAG \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    mama-synth-synthesis

echo "=== Output ==="
find "$SCRIPT_DIR/test/output" -type f
