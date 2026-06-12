#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
VERSION="${1:-v1.0.0}"
OUT_FILE="$SCRIPT_DIR/mama-synth-synthesis-${VERSION}.tar.gz"

bash "$SCRIPT_DIR/do_build.sh"
docker save mama-synth-synthesis | gzip > "$OUT_FILE"
echo "Saved: $OUT_FILE ($(ls -lh "$OUT_FILE" | awk '{print $5}'))"
echo "Upload to GC → Algorithm → Container Management."
