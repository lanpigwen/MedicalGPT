import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import numpy as np
import matplotlib.pyplot as plt

def build_text(tokenizer, conversations, answer):
    messages = []

    for msg in conversations:
        role = msg["from"]
        value = msg["value"]

        if role in ["human", "user"]:
            messages.append({"role": "user", "content": value})
        elif role in ["gpt", "assistant"]:
            messages.append({"role": "assistant", "content": value})
        elif role == "system":
            messages.append({"role": "system", "content": value})

    messages.append({"role": "assistant", "content": answer})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@torch.no_grad()
def get_reward(model, tokenizer, text, device, max_length):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    reward = model(**inputs).logits.squeeze()
    return reward.float().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--reward_model", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2560)
    parser.add_argument("--use_lora", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if args.use_lora:
        model = PeftModel.from_pretrained(model, args.reward_model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model,
            num_labels=1,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model.to(device)
    model.eval()

    total = 0
    correct = 0
    margins = []
    chosen_rewards = []
    rejected_rewards = []

    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)

            chosen_text = build_text(tokenizer, item["conversations"], item["chosen"])
            rejected_text = build_text(tokenizer, item["conversations"], item["rejected"])
            

            r_chosen = get_reward(model, tokenizer, chosen_text, device, args.max_length)
            r_rejected = get_reward(model, tokenizer, rejected_text, device, args.max_length)

            margin = r_chosen - r_rejected

            total += 1
            correct += int(margin > 0)
            margins.append(margin)
            chosen_rewards.append(r_chosen)
            rejected_rewards.append(r_rejected)

    pairwise_acc = correct / total

    chosen_np = np.array(chosen_rewards)
    rejected_np = np.array(rejected_rewards)
    margins_np = chosen_np - rejected_np

    mean_margin = margins_np.mean()
    mean_chosen_reward = chosen_np.mean()
    mean_rejected_reward = rejected_np.mean()

    print("=" * 50)
    print(f"num_samples          : {total}")
    print(f"pairwise_acc         : {pairwise_acc:.4f}")
    print(f"mean_margin          : {mean_margin:.4f}")
    print(f"mean_chosen_reward   : {mean_chosen_reward:.4f}")
    print(f"mean_rejected_reward : {mean_rejected_reward:.4f}")
    print("=" * 50)

    def print_dist(name, arr):
        print(f"\n[{name}]")
        print(f"mean   : {arr.mean():.4f}")
        print(f"std    : {arr.std():.4f}")
        print(f"min    : {arr.min():.4f}")
        print(f"p1     : {np.percentile(arr, 1):.4f}")
        print(f"p5     : {np.percentile(arr, 5):.4f}")
        print(f"p25    : {np.percentile(arr, 25):.4f}")
        print(f"median : {np.percentile(arr, 50):.4f}")
        print(f"p75    : {np.percentile(arr, 75):.4f}")
        print(f"p95    : {np.percentile(arr, 95):.4f}")
        print(f"p99    : {np.percentile(arr, 99):.4f}")
        print(f"max    : {arr.max():.4f}")

    # print_dist("Chosen Reward", chosen_np)
    # print_dist("Rejected Reward", rejected_np)
    # print_dist("Margin", margins_np)

    # --------------------------------------------------
    # chosen reward histogram
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(chosen_np, bins=50)
    plt.title("Chosen Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("chosen_reward_distribution.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # rejected reward histogram
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(rejected_np, bins=50)
    plt.title("Rejected Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rejected_reward_distribution.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # chosen vs rejected overlay
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.hist(
        chosen_np,
        bins=50,
        alpha=0.5,
        label="Chosen",
        density=True,
    )

    plt.hist(
        rejected_np,
        bins=50,
        alpha=0.5,
        label="Rejected",
        density=True,
    )

    plt.title("Reward Distribution Comparison")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_distribution_compare.png", dpi=300)
    plt.close()

    print("\nSaved figures:")
    print("chosen_reward_distribution.png")
    print("rejected_reward_distribution.png")
    print("reward_distribution_compare.png")

if __name__ == "__main__":
    main()