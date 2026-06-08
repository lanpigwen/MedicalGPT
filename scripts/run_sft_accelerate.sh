#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
SWANLAB_PROJECT="qwen-medical" \
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29501 \
    training/supervised_finetuning.py \
    --model_name_or_path ./model/Qwen2.5-3B\
    --train_file_dir ./data/dxw/sft/train \
    --validation_file_dir ./data/dxw/sft/val \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_eval_samples 500 \
    --model_max_length 512 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 30 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 4 \
    --output_dir output/sft-qwen2.5-high \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to swanlab \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --flash_attn True
