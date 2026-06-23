#!/bin/bash
#
# clean_kfold_labreast.sh — Remove LABREAST cases from kfold train/test dirs.
# LABREAST has ellipse-approximated masks (73% Dice=0), not suitable for segmentation eval.
#
# Usage (on Berzelius):
#   bash clean_kfold_labreast.sh [--dry-run]

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
KFOLD_DIR=$PROJ/kfold
DRY_RUN=false

if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN (no files will be removed) ==="
fi

TOTAL=0
for fold in 0 1 2 3 4; do
    for split in train test; do
        for sub in input ground_truth mask breast_mask; do
            DIR=$KFOLD_DIR/fold_${fold}/${split}/mha/${sub}
            if [ ! -d "$DIR" ]; then
                continue
            fi
            COUNT=$(ls "$DIR"/LABREAST*.mha 2>/dev/null | wc -l)
            if [ "$COUNT" -gt 0 ]; then
                echo "fold_${fold}/${split}/mha/${sub}: $COUNT LABREAST files"
                TOTAL=$((TOTAL + COUNT))
                if [ "$DRY_RUN" = false ]; then
                    rm -f "$DIR"/LABREAST*.mha
                fi
            fi
        done
    done
done

if [ "$DRY_RUN" = true ]; then
    echo "=== Would remove $TOTAL files total. Run without --dry-run to delete. ==="
else
    echo "=== Removed $TOTAL LABREAST files from kfold directories ==="
fi
