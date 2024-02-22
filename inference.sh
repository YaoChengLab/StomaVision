#!/bin/bash

# data
#datas="2023-ABRC-cauliflower-102 2023-ABRC-maize-30 2023-ABRC-lily-31 2023-Leafnet-F6-new-52 2023-Leafnet-F6-old-95 2023-Leafnet-F6-147 2023-all-337"

proj_dir="/mnt/linux/abrc/abrc/dataset/stomaVDP"
datas="2023-all-new-337"

# weight
#models="2023-ABRC-cauliflower-102-hyp-med-32 2023-ABRC-lily-31-hyp-med-32 2023-ABRC-maize-30-hyp-med-32 2023-all-337-hyp-med-32 2023-Leafnet-F6-147-hyp-med-32 2023-Leafnet-F6-new-52-hyp-med-32 2023-Leafnet-F6-old-95-hyp-low-32"

models="2023-all-new-337-hyp-high"

# Inference
for d in $datas; do
    for m in $models; do
        echo "******** inference $m on $d ***********"
        # Run inference
        python seg/segment/predict-abrc-new.py \
            --device cpu \
            --weights /mnt/linux/abrc/abrc/models/$m.pt \
            --conf 0.3 \
            --img-size 640 \
            --line-thickness 2 \
            --source $proj_dir/$d/images/val \
            --project ./runs/predict-seg \
            --save-txt
    done
done
