# Finetune p5 models with multiple GPUs

batch_size=8

time=$(date +%F_%T)

# python seg/segment/train.py \
#     --workers 6 \
#     --device cpu \
#     --batch-size $batch_size \
#     --cfg cfg/training/yolov7-seg.yaml \
#     --data data/$data.yaml \
#     --img 640 \
#     --weights 'model/yolov7-seg.pt' \
#     --name yolov7-abrc-$data \
#     --hyp hyp/hyp.scratch.abrc.yaml

# sample command for distributed training with 2 gpu
    # -m torch.distributed.run \
    # --nproc_per_node 2 \
    # --sync-bn \
# python seg/segment/train.py \
#     --workers 8 \
#     --device 0 \
#     --batch-size $batch_size \
#     --cfg cfg/training/yolov7-seg.yaml \
#     --data data/stomavision.yaml \
#     --img 640 \
#     --weights 'model/yolov7-seg.pt' \
#     --name single-label-$time \
#     --hyp hyp/hyp.scratch.abrc.yaml \
#     --epochs 300

python seg/segment/train.py \
    --workers 8 \
    --device 1 \
    --batch-size $batch_size \
    --cfg cfg/training/yolov7-seg-multilabel.yaml \
    --data data/stomavision-multilabel.yaml \
    --img 640 \
    --weights 'model/yolov7-seg.pt' \
    --name multi-label-$time \
    --hyp hyp/hyp.scratch.abrc.yaml \
    --epochs 300

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
