#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import random

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_prompt_hash(prompt):
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def load_done_keys(output_file):
    done = set()

    if not output_file or not Path(output_file).exists():
        return done

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                key = obj.get("id") or obj.get("meta", {}).get("prompt_hash")
                if key:
                    done.add(str(key))
            except Exception:
                continue

    return done


def extract_sample(item):
    """
    支持：
    1. ShareGPT:
       {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}

    2. 普通格式:
       {"question":"...", "reference":"..."}
       {"Question":"...", "Answer":"..."}

    3. DPO格式:
       {"question":"...", "chosen":"...", "rejected":"..."}
       提取 question 和 chosen
    """

    if "conversations" in item:
        convs = item.get("conversations") or []
        question = None
        reference = ""

        for i, msg in enumerate(convs):
            role = msg.get("from")
            value = str(msg.get("value", "")).strip()

            if role in ["human", "user"]:
                question = value

                if i + 1 < len(convs):
                    nxt = convs[i + 1]
                    if nxt.get("from") in ["gpt", "assistant"]:
                        reference = str(nxt.get("value", "")).strip()
                break

        return question, reference

    question = (
        item.get("question")
        or item.get("prompt")
        or item.get("Question")
        or ""
    )

    # 优先使用 chosen
    reference = (
        item.get("chosen")
        or item.get("response_chosen")
        or item.get("reference")
        or item.get("answer")
        or item.get("Answer")
        or item.get("output")
        or ""
    )

    return str(question).strip(), str(reference).strip()
def clean_think(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("\n\n", "\n")
    text = text.replace("rejected:  \n", "")
    text = text.replace("rejected:", "")

    return text.strip()


def build_chosen_prompt(question, reference=""):
    ref_part = ""
    if reference:
        ref_part = f"""
参考答案：
{reference}

请优先参考上述参考答案，但不要机械复制。
"""

    return f"""/no_think

你是一名严谨的医学问答专家。请回答下面的医学问题，用于构造医学大模型偏好数据。

要求：
1. 回答必须医学准确、安全、清晰。
2. 不要编造没有依据的诊断或治疗。
3. 回答应简洁完整，适合作为高质量 chosen answer。
4. 不要输出思考过程。
5. 不要输出 <think>。
6. 只输出最终回答，不要输出标题、编号或额外解释。

问题：
{question}

{ref_part}
"""


def build_rejected_prompt(question, chosen):
    difficulty = random.choices(
        population=[1, 2, 3, 4],
        weights=[0.25, 0.25, 0.25, 0.25],
        k=1
    )[0]

    error_level_desc = {
        1: """等级1（严重错误）
    - 制造一个或多个严重医学错误
    - 可涉及核心诊断、核心病因、核心治疗方向或关键用药
    - 回答质量应明显低于正确回答
    - 但不要胡言乱语""",

        2: """等级2（局部错误）
    - 整体回答基本合理
    - 仅包含一个明确医学错误
    - 可在诊断、病因、治疗、用药、检查建议、风险提示或预后判断中选择一种改错
    - 除该错误外，其余内容尽量保持正确""",

        3: """等级3（信息缺失）
    - 不要制造明显医学错误
    - 整体方向保持正确
    - 通过遗漏关键信息降低回答质量
    - 可遗漏检查建议、风险提示、治疗细节或重要结论
    - 回答应明显不如正确回答，但仍然合理""",

        4: """等级4（困难负样本）
    - 不要制造明显医学错误
    - 不要遗漏核心信息
    - 回答与正确回答应非常接近
    - 可降低专业性、准确性、完整性或信息密度
    - 可使用更泛化、更模糊的表述
    - 可存在少量冗余或重点不突出
    - 质量仅略差于正确回答"""
    }
    
    length = random.choices(
        population=[0, 1, 2],
        weights=[0.5, 0.25, 0.25],
        k=1
    )[0]    
    
    length_desc = ["长度与正确回答基本一致", "长度可以明显短于正确回答", "长度可以明显长于正确回答"][length]

    return f"""/no_think
你是一个医学偏好数据构造助手。

任务：
根据给定问题和正确回答，生成一个质量差一些的 rejected 回答，用于训练奖励模型和偏好优化模型。

错误等级：
{difficulty}

等级要求：
{error_level_desc[difficulty]}

生成要求：

1. 根据当前错误等级生成 rejected。
2. 不要解释或分析。
3. {length_desc}。
4. 只输出 最后的rejected的回答。

问题：
{question}

正确回答：
{chosen}
"""


def is_bad_text(text):
    if not text:
        return True

    bad_patterns = [
        "<think>",
        "</think>",
        "无法回答",
        "不能提供",
        "我是AI",
        "作为AI",
        "这是错误回答",
        "单点错误",
        "故意错误",
        "故意",
    ]

    return any(x in text for x in bad_patterns)


def too_similar(a, b):
    a = a.strip()
    b = b.strip()

    if a == b:
        return True

    if not a or not b:
        return True

    overlap = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    return overlap > 0.98


def build_chat_prompt(tokenizer, content):
    messages = [{"role": "user", "content": content}]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_vllm(model_path, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9):
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )


def batch_generate_vllm(
    llm,
    tokenizer,
    prompts,
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
):
    from vllm import SamplingParams

    chat_prompts = [
        build_chat_prompt(tokenizer, p)
        for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    outputs = llm.generate(chat_prompts, sampling_params)

    results = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        results.append(clean_think(text))

    return results


def build_openai_client(base_url, api_key):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def call_openai_once(
    client,
    model,
    prompt,
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
    retries=3,
    sleep_seconds=2.0,
):
    last_err = None

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            text = resp.choices[0].message.content or ""
            return clean_think(text)

        except Exception as e:
            last_err = e
            time.sleep(sleep_seconds * (attempt + 1))

    print(f"[WARN] API call failed after {retries} retries: {last_err}")
    return ""


def batch_generate_api(
    client,
    prompts,
    model="default",
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
    concurrency=4,
    retries=3,
):
    results = [None] * len(prompts)

    if concurrency <= 1:
        for i, prompt in enumerate(tqdm(prompts, desc="API generation")):
            results[i] = call_openai_once(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                retries=retries,
            )
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {}

        for i, prompt in enumerate(prompts):
            fut = executor.submit(
                call_openai_once,
                client,
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                retries,
            )
            future_to_idx[fut] = i

        for fut in tqdm(as_completed(future_to_idx), total=len(prompts), desc="API generation"):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[WARN] future failed at idx={idx}: {e}")
                results[idx] = ""

    return results


def generate_texts(
    backend,
    prompts,
    args,
    llm=None,
    tokenizer=None,
    client=None,
    max_tokens=512,
    temperature=0.2,
):
    if not prompts:
        return []

    if backend == "vllm":
        return batch_generate_vllm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=args.top_p,
        )

    if backend == "api":
        return batch_generate_api(
            client=client,
            prompts=prompts,
            model=args.api_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=args.top_p,
            concurrency=args.api_concurrency,
            retries=args.api_retries,
        )

    raise ValueError(f"Unknown backend: {backend}")


def build_result_item(sample, chosen, rejected, args):
    chosen = clean_think(chosen)
    rejected = clean_think(rejected)

    if is_bad_text(chosen) or is_bad_text(rejected) or too_similar(chosen, rejected):
        return None

    prompt_hash = get_prompt_hash(sample["question"])

    item = {
        "id": prompt_hash,
        "prompt": sample["question"],
        "chosen": chosen,
        "rejected": rejected,
        "source": "path1_strong_model_single_error_edit",
        "meta": {
            "prompt_hash": prompt_hash,
            "reference": sample["reference"],
            "strategy": "strong_chosen_plus_single_point_error",
            "backend": args.backend,
        }
    }

    if args.save_raw:
        item["raw"] = sample["raw"]

    return item


def process_chunk(chunk, args, llm=None, tokenizer=None, client=None):
    """
    处理一个 chunk，并返回可保存的 preference pairs。
    """

    # 1. chosen
    if args.use_reference_as_chosen:
        chosen_answers = [s["reference"] or "" for s in chunk]

        missing_indices = [
            i for i, ans in enumerate(chosen_answers)
            if not ans.strip()
        ]

        if missing_indices:
            chosen_prompts = [
                build_chosen_prompt(chunk[i]["question"], chunk[i]["reference"])
                for i in missing_indices
            ]

            generated = generate_texts(
                backend=args.backend,
                prompts=chosen_prompts,
                args=args,
                llm=llm,
                tokenizer=tokenizer,
                client=client,
                max_tokens=args.chosen_max_tokens,
                temperature=args.chosen_temperature,
            )

            for idx, ans in zip(missing_indices, generated):
                chosen_answers[idx] = ans

    else:
        chosen_prompts = [
            build_chosen_prompt(s["question"], s["reference"])
            for s in chunk
        ]

        chosen_answers = generate_texts(
            backend=args.backend,
            prompts=chosen_prompts,
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            client=client,
            max_tokens=args.chosen_max_tokens,
            temperature=args.chosen_temperature,
        )

    # 2. rejected
    rejected_prompts = [
        build_rejected_prompt(s["question"], chosen)
        for s, chosen in zip(chunk, chosen_answers)
    ]

    rejected_answers = generate_texts(
        backend=args.backend,
        prompts=rejected_prompts,
        args=args,
        llm=llm,
        tokenizer=tokenizer,
        client=client,
        max_tokens=args.rejected_max_tokens,
        temperature=args.rejected_temperature,
    )

    # 3. build pairs
    valid_items = []
    failed = 0

    for s, chosen, rejected in zip(chunk, chosen_answers, rejected_answers):
        item = build_result_item(s, chosen, rejected, args)

        if item is None:
            failed += 1
            continue

        valid_items.append(item)

    return valid_items, failed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "api"])

    # vLLM args
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # API args
    parser.add_argument("--api_base_url", type=str, default="http://localhost:4141/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--api_model", type=str, default="default")
    parser.add_argument("--api_concurrency", type=int, default=4)
    parser.add_argument("--api_retries", type=int, default=3)

    # data args
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)

    # generation args
    parser.add_argument("--chosen_max_tokens", type=int, default=512)
    parser.add_argument("--rejected_max_tokens", type=int, default=512)
    parser.add_argument("--chosen_temperature", type=float, default=0.2)
    parser.add_argument("--rejected_temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--use_reference_as_chosen", action="store_true")
    parser.add_argument("--save_raw", action="store_true")

    # streaming / resume
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary_file", type=str, default=None)

    args = parser.parse_args()

    raw_data = load_jsonl(args.input_file)

    samples = []
    for item in raw_data:
        q, ref = extract_sample(item)

        if q:
            samples.append({
                "question": q,
                "reference": ref,
                "raw": item,
            })

    if args.limit is not None:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} valid samples.")

    if args.resume:
        done_keys = load_done_keys(args.output_file)
        before = len(samples)

        samples = [
            s for s in samples
            if get_prompt_hash(s["question"]) not in done_keys
        ]

        print(f"Resume enabled. Already done: {before - len(samples)}. Remaining: {len(samples)}")
    else:
        # 非 resume 模式下，清空旧输出，避免重复追加
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")

    llm = None
    tokenizer = None
    client = None

    if args.backend == "vllm":
        if not args.model_path:
            raise ValueError("--model_path is required when --backend vllm")

        print(f"Loading vLLM model from: {args.model_path}")

        llm = build_vllm(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = llm.get_tokenizer()

    elif args.backend == "api":
        print(f"Using OpenAI-compatible API: {args.api_base_url}")
        print(f"API model: {args.api_model}")

        client = build_openai_client(
            base_url=args.api_base_url,
            api_key=args.api_key,
        )

    total_valid = 0
    total_failed = 0
    total_seen = 0

    chunks = [
        samples[i:i + args.chunk_size]
        for i in range(0, len(samples), args.chunk_size)
    ]

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Chunks")):
        print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)}, size={len(chunk)}")

        valid_items, failed = process_chunk(
            chunk=chunk,
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            client=client,
        )

        append_jsonl(valid_items, args.output_file)

        total_valid += len(valid_items)
        total_failed += failed
        total_seen += len(chunk)

        print(
            f"Chunk done. "
            f"seen={total_seen}, "
            f"new_valid={len(valid_items)}, "
            f"new_failed={failed}, "
            f"total_valid={total_valid}, "
            f"total_failed={total_failed}"
        )

        if args.summary_file:
            save_json(
                {
                    "input_file": args.input_file,
                    "output_file": args.output_file,
                    "backend": args.backend,
                    "processed_in_this_run": total_seen,
                    "valid_in_this_run": total_valid,
                    "failed_in_this_run": total_failed,
                    "remaining_after_resume": len(samples),
                    "chunk_size": args.chunk_size,
                },
                args.summary_file,
            )

    print(f"\nSaved to: {args.output_file}")
    print(f"Valid pairs this run: {total_valid}")
    print(f"Filtered / failed this run: {total_failed}")


if __name__ == "__main__":
    main()