#!/bin/bash

WORKDIR=${1:-./exp13/tiny-llm}
mkdir -p $WORKDIR

export PYTHONPATH=`pwd`:$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=8 --master_port=12345 \
    yololm/train/train.py \
    --model_name_or_path arnir0/Tiny-LLM  \
    --vision_tower ../runs/detect/yolov8l-worldv2-vp/weights/best.pt \
    --image_scale 800 \
    --in_channels 512 \
    --fp16 True \
    --output_dir $WORKDIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length 2048 \
    --report_to "none" \
    --ddp_find_unused_parameters False \
    --seed 0 \
    | tee $WORKDIR/train.log