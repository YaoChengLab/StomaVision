# Finetune p5 models with multiple GPUs
python -m torch.distributed.run \
    --nproc_per_node 2 \
    seg/segment/train.py \
    --workers 32 \
    --device 0,1 \
    --batch-size 16 \
    --cfg cfg/training/yolov7-seg.yaml \
    --data data/2023-all-new-337.yaml \
    --img 640 \
    --weights 'model/yolov7x-seg.pt' \
    --name yolov7-abrc-2023-all-new-337 \
    --hyp hyp/hyp.scratch.abrc-high.yaml

# Finetune p5 models with single GPU
#python ../seg/segment/train.py \
#--workers 8 \
#--device 2 \
#--batch-size 16 \
#--data data/stomata200-mix.yaml \
#--img 640 \
#--cfg cfg/training/yolov7-seg.yaml \
#--weights 'model/yolov7-seg.pt' \
#--name yolov7-abrc-stomata200-mix \
#--hyp hyp/hyp.scratch.abrc.yaml
