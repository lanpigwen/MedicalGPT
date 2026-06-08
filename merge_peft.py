import os
import json
import shutil
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TOKENIZER_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def copy_tokenizer_files(base_model, output_dir):
    print("Copying tokenizer files directly from base model...")

    for filename in TOKENIZER_FILES:
        src = os.path.join(base_model, filename)
        dst = os.path.join(output_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  copied {filename}")

    # 额外保险：如果 tokenizer_config.json 里有错误的 extra_special_tokens，删掉
    tok_cfg_path = os.path.join(output_dir, "tokenizer_config.json")

    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if isinstance(cfg.get("extra_special_tokens"), list):
            print("  removing invalid extra_special_tokens list")
            cfg.pop("extra_special_tokens", None)

        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


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

    copy_tokenizer_files(args.base_model, args.output_dir)

    print("Checking tokenizer...")
    _ = AutoTokenizer.from_pretrained(
        args.output_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    print(f"Done. Merged model saved in: {args.output_dir}")


if __name__ == "__main__":
    main()