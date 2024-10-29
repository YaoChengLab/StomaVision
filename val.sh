#!/bin/bash

# weight
models="single-label-1 multi-label-1"

# Validation
# for m in $models; do
#     python -m torch.distributed.run \
#         --nproc_per_node 2 \
#         seg/segment/val.py \
#         --workers 16 \
#         --device 0,1 \
#         --batch-size 16 \
#         --data data/stomavision.yaml \
#         --img 640 \
#         --weights /mnt/local/abrc/weights/$m.pt \
#         --conf 0.3 \
#         --task test \
#         --name $m
# done

python seg/segment/val.py \
    --workers 8 \
    --device 0 \
    --batch-size 8 \
    --data data/stomavision.yaml \
    --img 640 \
    --weights /mnt/local/abrc/weights/og.pt \
    --conf 0.3 \
    --task test \
    --name og


# python seg/segment/val.py \
#     --workers 8 \
#     --device 0 \
#     --batch-size 8 \
#     --data data/stomavision-multilabel.yaml \
#     --img 640 \
#     --weights /mnt/local/abrc/weights/multi-label-3.pt \
#     --conf 0.3 \
#     --task test \
#     --name multi-label-3
