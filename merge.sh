#!/bin/bash

set -e

checks=(
  500
  1000
  2000
  3000
  4000
)

BASE_MODEL="model/Qwen2.5-3B"
for check in "${checks[@]}"; do
  LORA_DIR="output/sft-qwen2.5-high/checkpoint-${check}"
  OUTPUT_DIR="model/Qwen2.5-3B-sft-${check}"
  python merge_peft.py \
    --base_model ${BASE_MODEL} \
    --lora_dir ${LORA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --dtype float16
done
