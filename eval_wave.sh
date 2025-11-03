#!/bin/bash
# Check if checkpoint ID and eval step are provided as arguments
if [ $# -eq 2 ]; then
    CHECKPOINT_ID=$1
    EVAL_STEP=$2
elif [ $# -eq 1 ]; then
    CHECKPOINT_ID=$1
    # Default eval step if only checkpoint ID is provided
    EVAL_STEP=100000
else
    # Default checkpoint ID and eval step if none provided
    CHECKPOINT_ID=6u72bd95
    EVAL_STEP=100000
fi

OVERRIDES="eval.checkpoint_dir=pinns/wave/square/checkpoints/${CHECKPOINT_ID},eval.step=${EVAL_STEP}"

PYTHONPATH=. python -m manifold_pinns.pipeline.cli pinn wave square \
    --mode eval \
    --override "${OVERRIDES}"
