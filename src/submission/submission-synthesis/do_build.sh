#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
docker build -t mama-synth-synthesis "$SCRIPT_DIR"
echo "Build complete: mama-synth-synthesis"
