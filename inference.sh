#!/bin/bash

proj_dir="./dataset"
datas="stomata_all"

models="model"

# Inference
for d in $datas; do
    for m in $models; do
        echo "******** inference $m on $d ***********"
        # Run inference
        python seg/segment/predict-abrc-new.py \
            --device cpu \
            --weights ./deploy/$m.pt \
            --conf 0.3 \
            --img-size 640 \
            --line-thickness 2 \
            --source $proj_dir/$d/images/val \
            --project ./runs/predict-seg \
            --save-txt
    done
done
