#!/bin/bash
# Export v32 Docker image as .tar.gz for Grand Challenge upload
set -e

cd "$(dirname "$0")"

OUTPUT_FILE="mama-synth-v32.tar.gz"

echo "=== Exporting Docker image ==="
docker save mama-synth-v32 | gzip > "$OUTPUT_FILE"

echo "Saved: $OUTPUT_FILE ($(ls -lh $OUTPUT_FILE | awk '{print $5}'))"
echo ""
echo "Upload this file to Grand Challenge."
