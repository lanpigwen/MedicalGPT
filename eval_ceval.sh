#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
models=(
    "Qwen2.5-3B" \
    # "Qwen2.5-3B-sft" \
    "Qwen2.5-3B-sft-500" \
    "Qwen2.5-3B-sft-1000" \
    "Qwen2.5-3B-sft-2000" \
    "Qwen2.5-3B-sft-3000" \
    "Qwen2.5-3B-sft-4000" \
)
num_fewshot=5

# 自动创建结果文件夹
mkdir -p ./eval_results

for model in "${models[@]}"; do
    echo "Evaluating model: $model with ${num_fewshot}-shot setting..."
    python -m lm_eval \
    --model hf \
    --model_args pretrained=./model/${model} \
    --tasks ceval-valid_basic_medicine,ceval-valid_clinical_medicine,ceval-valid_physician,ceval-valid_veterinary_medicine \
    --num_fewshot ${num_fewshot} \
    --batch_size auto \
    --device cuda \
    --output_path ./eval_results/${model}_${num_fewshot}shot_ceval_medical_result.json
done