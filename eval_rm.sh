# python eval_tools/eval_rm.py \
#   --base_model model/Qwen2.5-3B-sft-2000 \
#   --reward_model output/rm/c_all_ref---r_0.5_path2chosen \
#   --test_file data/dxw/dpo/c_all_ref---r_0.5_path2chosen/test/test_2_+_1_ref_0.5rejected_is_path2_chosen.jsonl \
#   --use_lora

# python eval_tools/eval_rm.py \
#   --base_model model/Qwen2.5-3B-sft-2000 \
#   --reward_model output/rm/c_all_ref---r_all_path2afterpath1 \
#   --test_file data/dxw/dpo/c_all_ref---r_0.5_path2chosen/test/test_2_+_1_ref_0.5rejected_is_path2_chosen.jsonl \
#   --use_lora


python eval_tools/eval_rm.py \
  --base_model model/Qwen2.5-3B-sft-2000 \
  --reward_model output/rm/c_all_ref---r_0.5_path2chosen \
  --test_file data/dxw/dpo/c_all_ref---r_all_path2afterpath1/test/test_2_+_1_ref.jsonl \
  --use_lora

python eval_tools/eval_rm.py \
  --base_model model/Qwen2.5-3B-sft-2000 \
  --reward_model output/rm/c_all_ref---r_all_path2afterpath1 \
  --test_file data/dxw/dpo/c_all_ref---r_all_path2afterpath1/test/test_2_+_1_ref.jsonl \
  --use_lora