# step=no
# lora_base_dir="output/sft-qwen2.5-high"

# lora_dirs=(
#   ${lora_base_dir}/checkpoint-500
#   ${lora_base_dir}/checkpoint-1000
#   ${lora_base_dir}/checkpoint-2000
#   ${lora_base_dir}/checkpoint-3000
#   ${lora_base_dir}/checkpoint-4000
# )

# python eval_tools/eval_base_lora.py \
#   --base_model model/Qwen2.5-3B \
#   --lora_dirs "${lora_dirs[@]}" \
#   --test_file data/dxw/sft/test/test_sharegpt.jsonl \
#   --output_file multi_lora_results-high.jsonl \
#   --max_new_tokens 512 \
#   --max_lora_rank 64 \
#   --max_loras 8

# python eval_tools/calc_ppl.py \
#     --base_model model/Qwen2.5-3B \
#     --sft_model model/Qwen2.5-3B-sft-18500 \
#     --test_file data/dxw/sft/test/test_sharegpt.jsonl

# python eval_tools/calc_metric.py \
#   --input_file sft_compare_results_merged_${step}.jsonl \
#   --use_bertscore


python eval_tools/calc_metrics.py \
  --input_file multi_lora_results-high.jsonl \
  --use_bertscore