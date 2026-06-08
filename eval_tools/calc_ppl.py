import json
import math
import argparse

import torch
from tqdm import tqdm
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def build_sample(item):
    convs = item["conversations"]

    question = None
    answer = None

    for i in range(len(convs) - 1):
        if convs[i]["from"] in ["human", "user"] and \
           convs[i + 1]["from"] in ["gpt", "assistant"]:

            question = convs[i]["value"]
            answer = convs[i + 1]["value"]
            break

    return question, answer


@torch.no_grad()
def compute_ppl(model, tokenizer, samples):

    losses = []

    for question, answer in tqdm(samples):

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        full_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).input_ids.to(model.device)

        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )

        user_ids = tokenizer(
            user_text,
            return_tensors="pt"
        ).input_ids.to(model.device)

        labels = full_ids.clone()

        answer_start = user_ids.shape[1]

        labels[:, :answer_start] = -100

        outputs = model(
            input_ids=full_ids,
            labels=labels
        )

        losses.append(outputs.loss.item())

    mean_loss = sum(losses) / len(losses)

    return math.exp(mean_loss)


def evaluate(model_path, samples):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    ppl = compute_ppl(
        model,
        tokenizer,
        samples
    )

    del model
    torch.cuda.empty_cache()

    return ppl


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True
    )

    parser.add_argument(
        "--sft_model",
        required=True
    )

    parser.add_argument(
        "--test_file",
        required=True
    )

    args = parser.parse_args()

    data = load_jsonl(args.test_file)

    samples = []

    for item in data:
        q, a = build_sample(item)

        if q and a:
            samples.append((q, a))

    print(f"Loaded {len(samples)} samples")

    base_ppl = evaluate(
        args.base_model,
        samples
    )

    sft_ppl = evaluate(
        args.sft_model,
        samples
    )

    table = PrettyTable()

    table.field_names = [
        "Model",
        "PPL"
    ]

    table.add_row([
        "Base",
        f"{base_ppl:.4f}"
    ])

    table.add_row([
        "SFT",
        f"{sft_ppl:.4f}"
    ])

    print(table)


if __name__ == "__main__":
    main()