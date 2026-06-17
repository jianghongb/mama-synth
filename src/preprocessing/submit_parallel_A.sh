#!/bin/bash
# Submit parallel mask_and_preprocess jobs (方案A, Dataset932)
# Each job processes 80 patients, GPU stays busy → won't be killed
#
# Usage: bash src/preprocessing/submit_parallel_A.sh

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
SCRIPT=$PROJ/mama-synth/src/preprocessing/run_mask_preprocess_A.sh
TOTAL=1507
BATCH=80

for START in $(seq 0 $BATCH $((TOTAL-1))); do
    END=$((START + BATCH))
    
    sbatch --export=ALL,BATCH_START=$START,BATCH_END=$END \
           --job-name="mask_${START}_${END}" \
           $SCRIPT
    
    echo "Submitted batch [$START:$END]"
done
