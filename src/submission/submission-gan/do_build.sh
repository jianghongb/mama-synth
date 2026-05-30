#!/usr/bin/env bash
# Build the Pix2PixHD GAN baseline algorithm container.
#
# This script:
#   1. Stages the medigan model weights from MODEL_WEIGHTS_DIR into the
#      Docker build context (models/00023/).
#   2. Runs docker build.
#
# The model weights are NOT committed to git (they are listed in .gitignore).
# Set MODEL_WEIGHTS_DIR to point to your local copy of the medigan 00023 model:
#
#   MODEL_WEIGHTS_DIR=/path/to/00023 ./do_build.sh
#
# Default path matches the reference location on the challenge admin's machine.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-/Users/richardosuala/Desktop/baseline GAN/00023}"

if [ ! -f "$MODEL_WEIGHTS_DIR/30_net_G.pth" ]; then
    echo ""
    echo "ERROR: Pix2PixHD weights not found at:"
    echo "  $MODEL_WEIGHTS_DIR/30_net_G.pth"
    echo ""
    echo "Set MODEL_WEIGHTS_DIR to the directory that contains '30_net_G.pth'."
    echo "  export MODEL_WEIGHTS_DIR=/path/to/00023"
    echo ""
    exit 1
fi

STAGING_DIR="$SCRIPT_DIR/models/00023"

echo "Staging model from : $MODEL_WEIGHTS_DIR"
echo "Staging target     : $STAGING_DIR"

# Clean and recreate staging directory
#rm -rf "$STAGING_DIR"
#mkdir -p "$STAGING_DIR"

# Copy model files, excluding:
#   input/         — large sample PNG images not needed in the container
#   __pycache__/   — byte-code caches (platform-specific, rebuilt in image)
#   *.pyc / *.pyo  — same reason
#   .DS_Store      — macOS metadata
#   MMG_env/       — any virtual environment that may exist in the model dir
#   test.sh        — test script only useful locally
rsync -a \
    --exclude='input/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='MMG_env/' \
    --exclude='test.sh' \
    "$MODEL_WEIGHTS_DIR/" "$STAGING_DIR/"

# Ensure the git-tracked __init__.py is not overwritten by the weights dir
git checkout -- "$STAGING_DIR/__init__.py" 2>/dev/null || true

echo ""
echo "Staged files:"
find "$STAGING_DIR" -type f | sort | grep -v __pycache__

echo ""
echo "Building Docker image: mama-synth-gan-baseline"
# --no-cache is intentionally omitted: Docker layer caching makes incremental
# rebuilds fast (only changed layers re-run).  Pass DOCKER_NO_CACHE=1 to force
# a fully clean build when you want to guarantee freshness:
#   DOCKER_NO_CACHE=1 ./do_build.sh
docker build ${DOCKER_NO_CACHE:+--no-cache} -t mama-synth-gan-baseline "$SCRIPT_DIR"
echo ""
echo "Build complete."
