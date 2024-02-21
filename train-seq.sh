# train from yolov7-seg.pt

HYPS="low med high"

#DATA="2023-ABRC-maize-30 2023-ABRC-lily-31 2023-ABRC-wheat-25 2023-Leafnet-F6-new-52 2023-Leafnet-F6-old-95 2023-Leafnet-F6-147 2023-all-337 2023-SAI-arabidopsis 2023-SAI-barley"

DATA="2023-ABRC-cauliflower-102.yaml"
#DATA="2023-SAI-arabidopsis-42"
#DATA="2023-ABRC-cauliflower-102"

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
