#!/bin/bash

subtask="c_all_ref---r_0.5_path2chosen"

configs=(
"1.0|sigmoid|1.0"
"1.0|sigmoid sft|1.0 0.2"
"1.0|sigmoid sft|1.0 5.0"
"1.0|sigmoid dpop|1.0 0.2"
"1.0|sigmoid dpop|1.0 5.0"
)

learning_rate=2e-5
warmup_steps=100
for config in "${configs[@]}"; do

    IFS='|' read -r dpo_beta loss_type_str loss_weight_str <<< "$config"

    read -ra loss_type <<< "$loss_type_str"
    read -ra loss_weights <<< "$loss_weight_str"

    loss_type_name=$(IFS=_; echo "${loss_type[*]}")
    loss_weights_name=$(IFS=_; echo "${loss_weights[*]}")
    param_config="beta${dpo_beta}_${loss_type_name}_${loss_weights_name}_${learning_rate}"
    name="${subtask}+${param_config}"

    echo "Running ${name}"

    SWANLAB_PROJECT="qwen-medical-dpo" \
    CUDA_VISIBLE_DEVICES=0,1 python3 training/dpo_training.py \
        --model_name_or_path model/Qwen2.5-3B-sft-2000 \
        --train_file_dir data/dxw/dpo/${subtask}/train \
        --validation_file_dir data/dxw/dpo/${subtask}/val \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --per_device_eval_batch_size 2 \
        --do_train \
        --do_eval \
        --use_peft True \
        --max_eval_samples 100 \
        --eval_steps 10 \
        --save_steps 50 \
        --max_source_length 1024 \
        --max_target_length 512 \
        --output_dir "output/${subtask}/${param_config}" \
        --target_modules all \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --torch_dtype bfloat16 \
        --bf16 True \
        --fp16 False \
        --report_to swanlab \
        --remove_unused_columns False \
        --gradient_checkpointing True \
        --tool_format default \
        --cache_dir ./cache \
        --dpo_beta "${dpo_beta}" \
        --run_name "${name}" \
        --loss_type "${loss_type[@]}" \
        --loss_weights "${loss_weights[@]}" \
        --learning_rate "${learning_rate}" \
        --warmup_steps "${warmup_steps}"
done