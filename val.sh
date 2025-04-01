#!/bin/bash

# for multilabel validation
# python seg/segment/val.py \
#     --workers 8 \
#     --device 1 \
#     --batch-size 8 \
#     --data data/stomavision-multilabel.yaml \
#     --img 640 \
#     --weights /home/heiru/projects/instill/StomaVision/seg/runs/train-seg/multilabel-2024-11-04_17:32:44/weights/best.pt \
#     --task test \
#     --name multilabel

python seg/segment/val_ensemble.py \
    --workers 16 \
    --device 1 \
    --batch-size 8 \
    --data data/stomavision-multilabel.yaml \
    --img 640 \
    --weights_stage1 /home/heiru/projects/instill/StomaVision/seg/runs/train-seg/outerline-best/weights/best.pt \
    --weights_stage2 /home/heiru/projects/instill/StomaVision/seg/runs/train-seg/cropped-best/weights/best.pt \
    --task test \
    --name emsemble
