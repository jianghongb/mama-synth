#!/bin/bash
# Submit all distillation jobs in sequence (each <1h)
# Usage: bash src/preprocessing/distill_submit_all.sh

PROJ=/proj/berzbiomedicalimagingkth/users/x_honji
SCRIPT_DIR=$PROJ/mama-synth/src/preprocessing

# Step 1: 932 inference in 6 batches (~250 cases each, ~1h per batch)
for i in 0 1 2 3 4 5; do
    JID=$(sbatch --parsable $SCRIPT_DIR/distill_batch_infer.sh $i)
    echo "Submitted batch $i: job $JID"
    if [ $i -lt 5 ]; then
        # Chain next job to start after this one
        sleep 1
    fi
done

echo ""
echo "After all inference batches complete, run:"
echo "  sbatch $SCRIPT_DIR/distill_batch_extract.sh"
echo "Then:"
echo "  sbatch $SCRIPT_DIR/distill_step3_train.sh"
