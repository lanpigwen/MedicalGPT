#!/bin/bash

set -e




checks=(
output/dpo-qwen2.5-2+1_0.1_sigmoid_1.0_5e-4
output/dpo-qwen2.5-2+1_0.1_sigmoid_dpop_1.0_0.2_5e-4
output/dpo-qwen2.5-2+1_0.1_sigmoid_dpop_1.0_5_5e-4
output/dpo-qwen2.5-2+1_0.1_sigmoid_sft_1.0_0.2_5e-4
)

BASE_MODEL="model/Qwen2.5-3B-sft-2000"
for check in "${checks[@]}"; do
  LORA_DIR="${check}"
  OUTPUT_DIR="model/Qwen2.5-3B-${check}"
  python merge_peft.py \
    --base_model ${BASE_MODEL} \
    --lora_dir ${LORA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --dtype float16
done
