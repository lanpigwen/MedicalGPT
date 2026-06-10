#!/bin/bash
SWANLAB_PROJECT="qwen-medical-grpo" \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 training/grpo_training.py \
    --model_name_or_path model/Qwen2.5-3B-sft-2000 \
    --train_file_dir data/dxw/grpo/train \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 13 \
    --output_dir outputs-grpo-qwen-v1 \
    --dtype bfloat16 \
    --bf16 True \
    --report_to swanlab \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm True \
    --vllm_max_model_len 4096 \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules all \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 512

echo "训练完成!"
