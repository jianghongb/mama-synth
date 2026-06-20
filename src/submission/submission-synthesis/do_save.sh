#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT="${1:-mamasynth.tar.gz}"

echo "=== Exporting Docker image ==="
docker save mamasynth | gzip > "$OUTPUT"
echo "Saved: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo ""
echo "Upload this file to Grand Challenge."
