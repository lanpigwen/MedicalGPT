python rewrite_cmexam_to_qa_vllm.py \
  --model_path /home/apulis-dev/userdata/LLM/Qwen/Qwen3-4B \
  --input_file CMExam/train.jsonl \
  --output_file CMExam/cmexam_qa_sharegpt.jsonl \
  --max_new_tokens 512 \
  --limit 12000