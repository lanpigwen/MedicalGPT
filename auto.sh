#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
set -e

BASE_MODEL="model/Qwen2.5-3B-sft-2000"
LORA_ROOTS=(
  "output/c_all_ref---r_0.5_path2chosen"
  "output/c_all_ref---r_all_path2afterpath1"
)
for LORA_ROOT in "${LORA_ROOTS[@]}"; do
    lora_dirs=()
    echo "===== Find LoRA Dirs ====="

    for dir in "$LORA_ROOT"/*; do
        if [ -d "$dir" ] && [ -f "$dir/adapter_config.json" ]; then
            echo "Found: $dir"
            lora_dirs+=("$dir")
        fi
    done

    echo
    echo "===== Evaluate ====="

    python eval_tools/eval_base_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_dirs "${lora_dirs[@]}" \
    --test_file data/dxw/sft/test/test_sharegpt.jsonl \
    --output_file ${LORA_ROOT}.jsonl \
    --max_new_tokens 512 \
    --max_lora_rank 64 \
    --max_loras 8

    echo
    echo "===== Metrics ====="

    python eval_tools/calc_metrics.py \
        --input_file ${LORA_ROOT}.jsonl \
        --use_bertscore

done



