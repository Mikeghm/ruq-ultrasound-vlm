#!/bin/bash
export PYTHONPATH=/workspaces/M3D-Original:$PYTHONPATH

# run "accelerate config" first!


accelerate launch LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path /workspace/M3D-Original/LaMed/pretrained_model/Phi-3-mini-4k-instruct \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /workspace/M3D-Original/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter /workspace/M3D_Test/M3D-LaMed-Phi-3-4B/mm_projector.bin \
    --bf16 False \
    --fp16 True \
    --output_dir ./LaMed/output/VQA-LaMed-Phi3-4B-finetune-0000 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard
