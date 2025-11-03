#!/bin/bash

if [ $# -ge 1 ]; then
    DATASET=$1
    if [ $# -ge 2 ]; then
        CHECKPOINT=$2
        if [ $# -ge 3 ]; then
            CHART=$3
        else
            CHART=1
        fi
    else
        CHECKPOINT=None
        CHART=1
    fi
else
    DATASET=coil
    CHECKPOINT=None
    CHART=1
fi

OVERRIDES="dataset.charts_path=./datasets/${DATASET}/charts_${CHART},training.batches_path=./pinns/eikonal/${DATASET}/data/charts_${CHART}/"

if [ "$CHECKPOINT" != "None" ]; then
    OVERRIDES="${OVERRIDES},eval.checkpoint_dir=./pinns/eikonal/${DATASET}/checkpoints/${CHECKPOINT}"
fi

PYTHONPATH=. python -m manifold_pinns.pipeline.cli pinn eikonal ${DATASET} \
    --mode eval \
    --override "${OVERRIDES}"
