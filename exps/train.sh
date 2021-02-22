python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --coco_path $COCO_PATH \
    --epochs 50 \
    --lr_drop 40 \
    --output_dir /home/wangshijie/ckpt/detr/coco/cgdetr_w_baseline3 \
    --lr 1e-4 \
    --lr_backbone 1e-5
