#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Verify weights exist
if [ ! -f weights/latest_net_G.pth ]; then
    echo "ERROR: weights/latest_net_G.pth not found"
    exit 1
fi

touch models/__init__.py

echo "Building Docker image..."
docker build -t mamasynth .
echo "Done."
