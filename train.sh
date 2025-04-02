# Finetune p5 models with multiple GPUs

batch_size=4

time=$(date +%F_%T)

# sample command for distributed training with 2 gpu
# python -m torch.distributed.run \
#     --nproc_per_node 2 \
#     --sync-bn \
#     --workers 6 \
#     --device 0,1 \
#     --batch-size $batch_size \
#     --cfg cfg/training/yolov7-seg.yaml \
#     --data data/$data.yaml \
#     --img 640 \
#     --weights 'model/yolov7-seg.pt' \
#     --name yolov7-abrc-$data \
#     --hyp hyp/hyp.scratch.abrc.yaml


# train stage 1 model
# python seg/segment/train.py \
#     --workers 8 \
#     --device 0 \
#     --batch-size $batch_size \
#     --cfg cfg/training/yolov7-seg.yaml \
#     --data data/stomavision-outerline.yaml \
#     --img 640 \
#     --weights 'model/yolov7-seg.pt' \
#     --name outerline-$time \
#     --hyp hyp/hyp.scratch.abrc.yaml \
#     --epochs 100

# train stage 2 model
python seg/segment/train.py \
    --workers 8 \
    --device 1 \
    --batch-size $batch_size \
    --cfg cfg/training/yolov7-seg.yaml \
    --data data/stomavision-cropped.yaml \
    --img 640 \
    --weights 'model/yolov7-seg.pt' \
    --name cropped-$time \
    --hyp hyp/hyp.scratch.abrc-cropped.yaml \
    --epochs 100
