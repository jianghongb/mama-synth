#!/bin/bash
# ============================================================
# Upload Yunnan P0 files + BreastDivider weights to Berzelius
# Yunnan structure: 8068383/N/N/P0.nii.gz (or 8068383/1/P0.nii.gz for case 1)
# Output structure: images/YUNNAN_NNN/YUNNAN_NNN_0000.nii.gz
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

count=0
for id in $(seq 1 100); do
    # Try both: N/P0.nii.gz and N/N/P0.nii.gz
    P0=""
    if [ -f "$YUNNAN_SRC/$id/P0.nii.gz" ]; then
        P0="$YUNNAN_SRC/$id/P0.nii.gz"
    elif [ -f "$YUNNAN_SRC/$id/$id/P0.nii.gz" ]; then
        P0="$YUNNAN_SRC/$id/$id/P0.nii.gz"
    fi

    if [ -n "$P0" ]; then
        padded=$(printf "%03d" "$id")
        dest="$STAGING/YUNNAN_${padded}"
        mkdir -p "$dest"
        ln -s "$P0" "$dest/YUNNAN_${padded}_0000.nii.gz"
        count=$((count+1))
    fi
done

echo "  Staged $count cases"

echo ""
echo "=== Step 3: Upload to Berzelius images/ (~10 GB) ==="
rsync -avPL "$STAGING/" ${REMOTE}:${PROJ}/images/

echo ""
echo "=== Done! Yunnan at: ${PROJ}/images/YUNNAN_001/ ... YUNNAN_100/ ==="
echo "Next: sbatch berzelius_breastdivider_distill.sh"
