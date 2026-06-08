# # !/bin/bash
# # 01 统计原始数据的基本信息
# python data_tools/wash/01_raw_stats.py

# # 02 对原始数据进行去广告等清洗
# python data_tools/wash/02_prefilter.py \
#     --input_jsonl data/dxw/step01_raw/kept.jsonl

# # 03 精确去重（question 版）
# python data_tools/wash/03_exact_dedup.py \
#     --input_jsonl data/dxw/step02_prefilter/kept.jsonl

# # 04.1) 统计 stop-ngrams（question 版）
# python data_tools/wash/04a_build_stop_ngrams.py \
#   --input_jsonl data/dxw/step03_exact_dedup/kept.jsonl \
#   --output_json data/dxw/step04_stop_ngrams/stop_ngrams.ng3.json \
#   --ngram 3 \
#   --min_df 200 \
#   --min_df_ratio 0.02

# # 04.2) 用 stop-ngrams 再跑 MinHash dedup（question 版）
# python data_tools/wash/04_minhash_dedup_question.py \
#   --input_jsonl data/dxw/step03_exact_dedup/kept.jsonl \
#   --output_dir data/dxw/step04_minhash_q \
#   --threshold 0.85 \
#   --stop_ngrams_json data/dxw/step04_stop_ngrams/stop_ngrams.ng3.json


# 05.a ) 用 LLM 进行领域过滤（question 版）
# python data_tools/wash/05a_llm_qa_quality_score.py \
#   --input_jsonl data/dxw/step04_minhash_q/kept.jsonl \
#   --output_jsonl data/dxw/step05_qa_quality_score/result.jsonl \
#   --base_url http://localhost:8000 \
#   --model Qwen3-4B \
#   --workers 64 \
#   --max_tokens 8 \
#   --disable_think \
#   --question_only

# 05.a ) 用 LLM 进行领域过滤（question + answer版）
# python data_tools/wash/05a_llm_qa_quality_score.py \
#   --input_jsonl data/dxw/step04_minhash_q/kept.jsonl \
#   --output_jsonl data/dxw/step05_qa_quality_score/result.jsonl \
#   --base_url http://localhost:8000 \
#   --model Qwen3-4B \
#   --workers 64 \
#   --max_tokens 8 \
#   --disable_think

# filter_threshold=9
# # 05.b ) 根据 LLM 评分进行过滤（question 版）
# python data_tools/wash/05b_filter_by_quality_score.py \
#  --input_jsonl data/dxw/step05a_qa_quality_score/result.jsonl \
#  --output_dir data/dxw/step05b_filter_by_quality_score \
#  --threshold $filter_threshold



# # 06) 最终划分训练/验证/测试集，并进行最终的 deleak（question 版）
# python data_tools/wash/06_split_and_deleak.py \
#  --input_jsonl data/dxw/step05b_filter_by_quality_score/kept_$filter_threshold.jsonl \
#  --output_dir data/dxw/step06_split_$filter_threshold \
#  --seed 42 \
#  --val_target 1000 \
#  --test_target 1000 \
#  --eval_oversample 1.4 \
#  --dedup_eval_exact \
#  --deleak_exact \
#  --deleak_minhash \
#  --stop_ngrams_json data/dxw/step04_stop_ngrams/stop_ngrams.ng3.json \
#  --lsh_threshold 0.9


# # # 06 shareGPT
# splits=(train val test)
# for split in "${splits[@]}"; do
#   python data_tools/wash/06_jsonl_format.py \
#     --input_jsonl data/dxw/step06_split_$filter_threshold/${split}.jsonl \
#     --output_jsonl data/dxw/step06_split_$filter_threshold/${split}_sharegpt.jsonl \
#     --key_to_keep conversations
# done


# # # 06 shareGPT
# python data_tools/wash/06_jsonl_format.py \
#   --input_jsonl CMExam/cmexam_qa.jsonl \
#   --output_jsonl CMExam/cmexam_qa_sharegpt.jsonl \
#   --key_to_keep conversations