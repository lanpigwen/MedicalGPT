export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen2.5-3B \
  --local-dir ./model/Qwen2.5-3B \
  --local-dir-use-symlinks False