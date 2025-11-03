#!/bin/bash

if [ $# -eq 1 ]; then
    MESH=$1
else
    MESH=coil
fi

PYTHONPATH=. python charts/make_charts.py \
 --config=charts/config/make_charts_${MESH}.py \
