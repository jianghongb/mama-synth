#!/bin/bash
# ============================================================
# Upload Yunnan P0 files + BreastDivider weights to Berzelius
# Yunnan files renamed to match images/ structure:
#   images/YUNNAN_001/YUNNAN_001_0000.nii.gz
# ============================================================
set -e

REMOTE="x_honji@berzelius.nsc.liu.se"
PROJ="/proj/berzbiomedicalimagingkth/users/x_honji"
YUNNAN_SRC="/Users/ehogjig/Downloads/8068383"

echo "=== Step 1: Upload BreastDivider model (~100 MB) ==="
rsync -avP /Users/ehogjig/git/kth/BreastDividerModel/ \
  ${REMOTE}:${PROJ}/BreastDividerModel/

echo ""
echo "=== Step 2: Stage Yunnan P0 files into images/ structure ==="
STAGING="/tmp/yunnan_upload"
rm -rf "$STAGING"

for dir in "$YUNNAN_SRC"/*/; do
    id=$(basename "$dir")
    [[ ! -d "$dir" ]] && continue
    P0="$dir/P0.nii.gz"
    [ -f "$P0" ] || continue
    padded=$(printf "%03d" "$id")
    dest="$STAGING/YUNNAN_${padded}"
    mkdir -p "$dest"
    ln -s "$P0" "$dest/YUNNAN_${padded}_0000.nii.gz"
done

COUNT=$(ls "$STAGING" | wc -l | tr -d ' ')
echo "  Staged $COUNT cases"

echo ""
echo "=== Step 3: Upload to Berzelius images/ (~10 GB) ==="
rsync -avPL "$STAGING/" ${REMOTE}:${PROJ}/images/

echo ""
echo "=== Done! Yunnan now at: ${PROJ}/images/YUNNAN_001/ ... YUNNAN_100/ ==="
echo "Next: sbatch berzelius_breastdivider_distill.sh"
