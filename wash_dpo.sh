# 04.2) 用 stop-ngrams 再跑 MinHash dedup（question 版）
# python data_tools/wash/04_minhash_dedup_question.py \
#   --input_jsonl data/dxw/step05b_filter_by_quality_score/kept_10.jsonl \
#   --output_dir data/dxw/step07_minhash_q_grpo \
#   --threshold 0.5 \
#   --stop_ngrams_json data/dxw/step04_stop_ngrams/stop_ngrams.ng3.json


# 05) 用 exact match + MinHash dedup 来做 deleak（question 版）
# python data_tools/wash/06_deleak.py \
#   --train_jsonl data/dxw/step07_minhash_q_grpo/kept.jsonl \
#   --eval_jsonl data/dxw/sft/test/test_sharegpt.jsonl \
#   --output_dir data/dxw/step07_deleak_train_grpo \
#   --use_exact \
#   --use_minhash \
#   --ngram 3 \
#   --num_perm 64 \
#   --threshold 0.5

# python data_tools/dpo/make_preference_data.py \
#   --backend api \
#   --api_base_url http://localhost:4141/v1 \
#   --api_key EMPTY \
#   --api_model gpt-4o \
#   --input_file data/dxw/step07_deleak_train_grpo/train_10.jsonl \
#   --output_file data/dxw/step08_preference_grpo/path1_ref_chosen.jsonl \
#   --use_reference_as_chosen \
#   --api_concurrency 60 \
#   --chunk_size 100


# python data_tools/dpo/make_preference_data.py \
#   --backend vllm \
#   --model_path ./model/Qwen2.5-3B \
#   --input_file data/dxw/step07_deleak_train_grpo/train_10.jsonl \
#   --output_file data/dxw/step08_preference_grpo/path1_ref_chosen.jsonl \
#   --use_reference_as_chosen \
#   --chunk_size 1000 \
#   --tensor_parallel_size 2 \
#   --chosen_temperature 0.2 \
#   --rejected_temperature 0.4

# python data_tools/dpo/make_preference_data_useRef_rate.py \
#   --model_path ./model/Qwen2.5-3B-sft-2000 \
#   --input_file data/dxw/step07_deleak_train_grpo/train_10.jsonl \
#   --output_file data/dxw/step08_preference_grpo/path2_ref_chosen.jsonl \
#   --num_candidates 5 \
#   --temperature 0.7 \
#   --top_p 0.95 \
#   --max_tokens 512 \
#   --rejected_strategy mixed \
#   --min_score_gap 0.03 \
#   --tensor_parallel_size 2 \
#   --max_model_len 4096


# python data_tools/dpo/make_preference_data_useRef_rate.py \
#   --model_path ./model/Qwen2.5-3B-sft-2000 \
#   --input_file data/dxw/step07_deleak_train_grpo/train_10.jsonl \
#   --output_file data/dxw/step08_preference_grpo/path2_ref_chosen.jsonl \
#   --num_candidates 5 \
#   --temperature 0.7 \
#   --top_p 0.95 \
#   --max_tokens 512 \
#   --rejected_strategy mixed \
#   --min_score_gap 0.03 \
#   --tensor_parallel_size 2 \
#   --max_model_len 4096


# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/dxw/dpo/train/train.jsonl \
#   --output_file data/dxw/dpo/train/train_sharegpt.jsonl

# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/dxw/dpo/test/test.jsonl \
#   --output_file data/dxw/dpo/test/test_sharegpt.jsonl

# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/dxw/dpo/val/val.jsonl \
#   --output_file data/dxw/dpo/val/val_sharegpt.jsonl


# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/medical/reward/train/train.jsonl \
#   --output_file data/medical/reward/train/train_sharegpt.jsonl

# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/medical/reward/test/test.jsonl \
#   --output_file data/medical/reward/test/test_sharegpt.jsonl

# python data_tools/dpo/convert_reward_format.py \
#   --input_file data/medical/reward/val/val.jsonl \
#   --output_file data/medical/reward/val/val_sharegpt.jsonl



# python data_tools/dpo/make_preference_data.py \
#   --backend vllm \
#   --model_path ./model/Qwen2.5-3B \
#   --input_file data/dxw/step08_preference_grpo/path2_ref_chosen_cleaned.jsonl \
#   --output_file data/dxw/step08_preference_grpo/path2Afterpath1.jsonl \
#   --use_reference_as_chosen \
#   --chunk_size 1000 \
#   --tensor_parallel_size 2 \
#   --chosen_temperature 0.2 \
#   --rejected_temperature 0.4


# python data_tools/dpo/len.py --input data/dxw/step08_preference_grpo/path2Afterpath1.jsonl



python data_tools/dpo/convert_reward_format.py \
  --input_file data/dxw/step08_preference_grpo/path2Afterpath1_cleaned.jsonl \
  --output_file data/dxw/step08_preference_grpo/path2Afterpath1_cleaned_sharegpt.jsonl