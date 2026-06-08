import os
import json
import argparse
from collections import defaultdict

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_data(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def extract_sharegpt(item):
    convs = item["conversations"]

    question = None
    reference = ""

    for i, msg in enumerate(convs):
        role = msg.get("from")
        value = msg.get("value", "").strip()

        if role in ["human", "user"]:
            question = value

            if i + 1 < len(convs):
                next_msg = convs[i + 1]
                if next_msg.get("from") in ["gpt", "assistant"]:
                    reference = next_msg.get("value", "").strip()
            break

    return question, reference


def build_prompt(tokenizer, question):
    messages = [
        {"role": "user", "content": question}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def get_lora_name(lora_path):
    return os.path.basename(os.path.normpath(lora_path))


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_llm(base_model, max_lora_rank=64, max_loras=8):
    return LLM(
        model=base_model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=max_loras,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="multi_lora_onepass_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--max_loras", type=int, default=8)

    args = parser.parse_args()

    data = load_data(args.test_file)

    questions = []
    references = []

    for item in data:
        q, ref = extract_sharegpt(item)
        if q is not None:
            questions.append(q)
            references.append(ref)

    print(f"Loaded {len(questions)} samples.")

    lora_infos = []
    for idx, lora_dir in enumerate(args.lora_dirs, start=1):
        lora_name = get_lora_name(lora_dir)
        lora_infos.append({
            "name": lora_name,
            "id": idx,
            "path": lora_dir,
        })

    print("LoRA adapters:")
    for info in lora_infos:
        print(f"  id={info['id']} name={info['name']} path={info['path']}")

    llm = build_llm(
        base_model=args.base_model,
        max_lora_rank=args.max_lora_rank,
        max_loras=max(args.max_loras, len(lora_infos))
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.1,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    all_prompts = []
    all_lora_requests = []
    all_meta = []

    for sample_idx, q in enumerate(questions):
        prompt = build_prompt(tokenizer, q)

        # base request
        all_prompts.append(prompt)
        all_lora_requests.append(None)
        all_meta.append({
            "sample_idx": sample_idx,
            "type": "base",
            "lora_name": None,
        })

        # lora requests
        for info in lora_infos:
            all_prompts.append(prompt)
            all_lora_requests.append(
                LoRARequest(
                    info["name"],
                    info["id"],
                    info["path"]
                )
            )
            all_meta.append({
                "sample_idx": sample_idx,
                "type": "lora",
                "lora_name": info["name"],
            })

    print(f"Total requests: {len(all_prompts)}")
    print("Running mixed base + multi-LoRA generation...")

    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=all_lora_requests,
    )

    base_answers = {}
    lora_answers = defaultdict(dict)

    for meta, out in zip(all_meta, outputs):
        sample_idx = meta["sample_idx"]
        text = out.outputs[0].text.strip()

        if meta["type"] == "base":
            base_answers[sample_idx] = text
        else:
            lora_name = meta["lora_name"]
            lora_answers[sample_idx][lora_name] = text

    results = []

    for i, (q, ref) in enumerate(zip(questions, references)):
        results.append({
            "question": q,
            "reference": ref,
            "base_answer": base_answers.get(i, ""),
            "sft_results": lora_answers.get(i, {})
        })

    save_jsonl(results, args.output_file)

    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()