import os
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading base model from: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA from: {args.lora_dir}")
    model = PeftModel.from_pretrained(
        model,
        args.lora_dir,
        torch_dtype=torch_dtype,
    )

    print("Merging LoRA...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    print("Saving tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    print("Checking tokenizer...")
    _ = AutoTokenizer.from_pretrained(
        args.output_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    print(f"Done. Merged model saved in: {args.output_dir}")


if __name__ == "__main__":
    main()