# Finetune p5 models with multiple GPUs
# train from yolov7x-seg.pt
HYPS="high med low"
data=example
batch_size=16
for h in $HYPS; do
    python -m torch.distributed.run \
        --nproc_per_node 4 \
        --master_port 9527 \
        seg/segment/train.py \
        --workers 8 \
        --device 0,1,2,3 \
        --batch-size $batch_size \
        --cfg cfg/training/yolov7-seg.yaml \
        --data data/$data.yaml \
        --img 640 \
        --weights 'model/yolov7x-seg.pt' \
        --name yolov7-abrc-$data-x-hyp-$h-$batch_size \
        --hyp hyp/hyp.scratch.abrc-$h.yaml \
        --epochs 1000
done

batch_size=32
for h in $HYPS; do
    python -m torch.distributed.run \
        --nproc_per_node 4 \
        --master_port 9527 \
        seg/segment/train.py \
        --workers 8 \
        --device 0,1,2,3 \
        --batch-size $batch_size \
        --cfg cfg/training/yolov7-seg.yaml \
        --data data/$data.yaml \
        --img 640 \
        --weights 'model/yolov7x-seg.pt' \
        --name yolov7-abrc-$data-x-hyp-$h-$batch_size \
        --hyp hyp/hyp.scratch.abrc-$h.yaml \
        --epochs 1000
done

batch_size=48
for h in $HYPS; do
    python -m torch.distributed.run \
        --nproc_per_node 4 \
        --master_port 9527 \
        seg/segment/train.py \
        --workers 8 \
        --device 0,1,2,3 \
        --batch-size $batch_size \
        --cfg cfg/training/yolov7-seg.yaml \
        --data data/$data.yaml \
        --img 640 \
        --weights 'model/yolov7x-seg.pt' \
        --name yolov7-abrc-$data-x-hyp-$h-$batch_size \
        --hyp hyp/hyp.scratch.abrc-$h.yaml \
        --epochs 1000
done
