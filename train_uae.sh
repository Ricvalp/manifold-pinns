#!/bin/bash

if [ $# -eq 1 ]; then
    DATASET=$1
else
    DATASET=bunny
fi


PYTHONPATH=. python -m manifold_pinns.pipeline.cli uae ${DATASET}
