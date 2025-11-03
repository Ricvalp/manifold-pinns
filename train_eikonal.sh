#!/bin/bash

if [ $# -ge 1 ]; then
    DATASET=$1
    if [ $# -ge 2 ]; then
        MODE=$2
        if [ $# -ge 3 ]; then
            CHART=$3
        else
            CHART=1
        fi
    else
        MODE=train
        CHART=1
    fi
else
    DATASET=coil
    MODE=train
    CHART=1
fi


OVERRIDES="dataset.charts_path=./datasets/${DATASET}/charts_${CHART},training.batches_path=./pinns/eikonal/${DATASET}/data/charts_${CHART}/"

PYTHONPATH=. python -m manifold_pinns.pipeline.cli pinn eikonal ${DATASET} \
    --mode ${MODE} \
    --override "${OVERRIDES}"
