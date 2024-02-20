#!/bin/bash

# data
# datas="2023-ABRC-cauliflower-102 2023-ABRC-maize-30 2023-ABRC-lily-31 2023-Leafnet-F6-new-52 2023-Leafnet-F6-old-95 2023-Leafnet-F6-147 2023-all-337"

datas="2023-all-new-337"

# weight
#models="2023-ABRC-cauliflower-102-hyp-med-32 2023-ABRC-lily-31-hyp-med-32 2023-ABRC-maize-30-hyp-med-32 2023-all-337-hyp-med-32 2023-Leafnet-F6-147-hyp-med-32 2023-Leafnet-F6-new-52-hyp-med-32 2023-Leafnet-F6-old-95-hyp-low-32"

models="2023-all-new-337-hyp-high"

# Validation
for d in $datas; do
    for m in $models; do
        echo "******** val $m on $d ***********"
        # Run validation (evaluation)
        python -m torch.distributed.run \
            --nproc_per_node 2 \
            ../seg/segment/val.py \
            --workers 16 \
            --device 0,1 \
            --batch-size 16 \
            --data data/$d.yaml \
            --img 640 \
            --weights /mnt/linux/abrc/abrc/models/$m.pt \
            --conf 0.3 \
            --name $m-$d
    done
done
