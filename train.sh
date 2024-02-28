# Finetune p5 models with multiple GPUs

data=example

#python -m torch.distributed.run \
python seg/segment/train.py \
    --workers 6 \
    --device cpu \
    --batch-size 8 \
    --cfg cfg/training/yolov7-seg.yaml \
    --data data/$data.yaml \
    --img 640 \
    --weights 'model/yolov7-seg.pt' \
    --name yolov7-abrc-$data \
    --hyp hyp/hyp.scratch.abrc.yaml

# Finetune p5 models with single GPU
#python seg/segment/train.py \
#--workers 8 \
#--device 2 \
#--batch-size 16 \
#--data data/stomata200-mix.yaml \
#--img 640 \
#--cfg cfg/training/yolov7-seg.yaml \
#--weights 'model/yolov7-seg.pt' \
#--name yolov7-abrc-stomata200-mix \
#--hyp hyp/hyp.scratch.abrc.yaml
