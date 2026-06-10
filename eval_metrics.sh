export HF_ENDPOINT=https://hf-mirror.com

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
#     --sft_model model/Qwen2.5-3B-sft-2000 \
#     --test_file data/dxw/sft/test/test_sharegpt.jsonl

# python eval_tools/calc_metric.py \
#   --input_file sft_compare_results_merged_${step}.jsonl \
#   --use_bertscore


# python eval_tools/calc_metrics.py \
#   --input_file multi_lora_results-high.jsonl \
#   --use_bertscore

# # merged
# python eval_tools/eval_multi_merged.py \
#   --base_model model/Qwen2.5-3B \
#   --model_dirs \
#     model/Qwen2.5-3B-output/dpo-qwen2.5-2+1_0.1_sigmoid_1.0_5e-4 \
#     model/Qwen2.5-3B-output/dpo-qwen2.5-2+1_0.1_sigmoid_dpop_1.0_0.2_5e-4 \
#     model/Qwen2.5-3B-output/dpo-qwen2.5-2+1_0.1_sigmoid_dpop_1.0_5_5e-4 \
#     model/Qwen2.5-3B-output/dpo-qwen2.5-2+1_0.1_sigmoid_sft_1.0_0.2_5e-4 \
#   --test_file data/dxw/sft/test/test_sharegpt.jsonl \
#   --output_file multi_merged_results-high-dpo-new-1+2.jsonl \
#   --max_new_tokens 512 \
#   --dtype bfloat16 \
#   --tensor_parallel_size 2 \
#   --max_model_len 4096


# python eval_tools/calc_metrics.py \
#   --input_file multi_merged_results-high-dpo-new-1+2.jsonl \
#   --use_bertscore