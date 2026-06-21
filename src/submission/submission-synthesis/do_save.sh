#!/bin/bash
set -e

VERSION="${1:?Usage: ./do_save.sh <version> (e.g. v4.0.0)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT="mama-synth-synthesis_${VERSION}.tar.gz"

echo "Exporting Docker image as ${OUTPUT}..."
docker save mamasynth | gzip > "$OUTPUT"
echo "Saved: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
