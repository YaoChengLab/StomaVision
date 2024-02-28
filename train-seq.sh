# train from yolov7-seg.pt

HYPS="low med high"

DATA="example"

batch_size=24

for d in $DATA; do
    for h in $HYPS; do
        python -m torch.distributed.run \
            --nproc_per_node ４ \
            --master_port 9527 \
            seg/segment/train.py \
            --workers 8 \
            --device 0,1,2,3 \
            --batch-size $batch_size \
            --cfg cfg/training/yolov7-seg.yaml \
            --data data/$d.yaml \
            --img 640 \
            --weights 'model/yolov7-seg.pt' \
            --name $d-hyp-$h-$batch_size \
            --hyp hyp/hyp.scratch.abrc-$h.yaml \
            --epochs 500
    done
done
