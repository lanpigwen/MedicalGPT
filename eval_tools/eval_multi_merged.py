import os
import sys
import json
import argparse
import tempfile
import subprocess

from vllm import LLM, SamplingParams


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def get_model_name(model_path):
    return os.path.basename(os.path.normpath(model_path))


def run_worker(args):
    data = load_jsonl(args.test_file)

    questions = []
    references = []

    for item in data:
        q, ref = extract_sharegpt(item)
        if q is not None:
            questions.append(q)
            references.append(ref)

    print(f"Loaded {len(questions)} samples.")
    print(f"Running model: {args.model_name}")
    print(f"Model path: {args.model_path}")

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    tokenizer = llm.get_tokenizer()

    prompts = [
        build_prompt(tokenizer, q)
        for q in questions
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    outputs = llm.generate(
        prompts,
        sampling_params
    )

    answers = [
        out.outputs[0].text.strip()
        for out in outputs
    ]

    worker_results = []

    for q, ref, ans in zip(questions, references, answers):
        worker_results.append({
            "question": q,
            "reference": ref,
            "answer": ans
        })

    save_jsonl(worker_results, args.worker_output)

    print(f"Worker result saved to {args.worker_output}")


def run_one_model_subprocess(
    script_path,
    model_path,
    model_name,
    test_file,
    worker_output,
    max_new_tokens,
    dtype,
    tensor_parallel_size,
    gpu_memory_utilization,
    max_model_len,
    repetition_penalty,
):
    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--model_path", model_path,
        "--model_name", model_name,
        "--test_file", test_file,
        "--worker_output", worker_output,
        "--max_new_tokens", str(max_new_tokens),
        "--dtype", dtype,
        "--tensor_parallel_size", str(tensor_parallel_size),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
        "--max_model_len", str(max_model_len),
        "--repetition_penalty", str(repetition_penalty),
    ]

    print("\n" + "=" * 80)
    print(f"Launching worker for {model_name}")
    print("=" * 80)

    subprocess.run(cmd, check=True)


def load_worker_answers(path):
    data = load_jsonl(path)
    return [item["answer"] for item in data]


def main():
    parser = argparse.ArgumentParser()

    # 主进程参数：不要在 argparse 这里 required=True
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--model_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="multi_merged_results.jsonl")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    # worker 参数
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--worker_output", type=str, default=None)

    args = parser.parse_args()

    # worker 模式：只检查 worker 需要的参数
    if args.worker:
        if args.model_path is None:
            raise ValueError("--model_path is required in worker mode")
        if args.model_name is None:
            raise ValueError("--model_name is required in worker mode")
        if args.worker_output is None:
            raise ValueError("--worker_output is required in worker mode")

        run_worker(args)
        return

    # 主进程模式：再检查主进程参数
    if args.base_model is None:
        raise ValueError("--base_model is required")
    if args.model_dirs is None or len(args.model_dirs) == 0:
        raise ValueError("--model_dirs is required")

    data = load_jsonl(args.test_file)

    questions = []
    references = []

    for item in data:
        q, ref = extract_sharegpt(item)
        if q is not None:
            questions.append(q)
            references.append(ref)

    print(f"Loaded {len(questions)} valid samples.")

    script_path = os.path.abspath(__file__)

    all_model_paths = [args.base_model] + args.model_dirs
    all_model_names = ["base"] + [
        get_model_name(p)
        for p in args.model_dirs
    ]

    model_answers = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for model_name, model_path in zip(all_model_names, all_model_paths):
            worker_output = os.path.join(
                tmpdir,
                f"{model_name}.jsonl"
            )

            run_one_model_subprocess(
                script_path=script_path,
                model_path=model_path,
                model_name=model_name,
                test_file=args.test_file,
                worker_output=worker_output,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                repetition_penalty=args.repetition_penalty,
            )

            model_answers[model_name] = load_worker_answers(worker_output)

    results = []

    for i, (q, ref) in enumerate(zip(questions, references)):
        sft_results = {}

        for model_name in all_model_names:
            if model_name == "base":
                continue

            sft_results[model_name] = model_answers[model_name][i]

        results.append({
            "question": q,
            "reference": ref,
            "base_answer": model_answers["base"][i],
            "sft_results": sft_results
        })

    save_jsonl(results, args.output_file)

    print(f"\nSaved final results to {args.output_file}")


if __name__ == "__main__":
    main()