# reward model 训练暂不支持 torchrun 多卡训练
subtask="c_all_ref---r_0.5_path2chosen"
SWANLAB_PROJECT="qwen-medical-rm" \
CUDA_VISIBLE_DEVICES=0,1 python3 training/reward_modeling.py \
    --model_name_or_path model/Qwen2.5-3B-sft-2000 \
    --train_file_dir data/dxw/dpo/${subtask}/train \
    --validation_file_dir data/dxw/dpo/${subtask}/val \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --use_peft True \
    --seed 42 \
    --max_eval_samples 20 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 1 \
    --eval_steps 5 \
    --eval_strategy steps \
    --save_steps 40 \
    --save_strategy steps \
    --save_total_limit 30 \
    --max_target_length 256 \
    --output_dir output/rm/${subtask} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to swanlab \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True 



subtask="c_all_ref---r_all_path2afterpath1"
SWANLAB_PROJECT="qwen-medical-rm" \
CUDA_VISIBLE_DEVICES=0,1 python3 training/reward_modeling.py \
    --model_name_or_path model/Qwen2.5-3B-sft-2000 \
    --train_file_dir data/dxw/dpo/${subtask}/train \
    --validation_file_dir data/dxw/dpo/${subtask}/val \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --use_peft True \
    --seed 42 \
    --max_eval_samples 20 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --weight_decay 0.001 \
    --logging_strategy steps \
    --logging_steps 1 \
    --eval_steps 5 \
    --eval_strategy steps \
    --save_steps 40 \
    --save_strategy steps \
    --save_total_limit 30 \
    --max_target_length 256 \
    --output_dir output/rm/${subtask} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to swanlab \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --gradient_checkpointing True 
