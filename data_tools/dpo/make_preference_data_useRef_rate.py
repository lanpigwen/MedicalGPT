#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path

import jieba
from tqdm import tqdm
from rouge_chinese import Rouge
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_sample(item):
    """
    支持：
    1. ShareGPT:
       {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
    2. 普通格式:
       {"question":"...", "reference":"..."}
    3. 其他常见格式:
       {"Question":"...", "Answer":"..."}
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

    reference = (
        item.get("reference")
        or item.get("answer")
        or item.get("Answer")
        or item.get("output")
        or ""
    )

    return str(question).strip(), str(reference).strip()


def clean_think(text):
    if text is None:
        return ""

    text = str(text)

    # 删除完整 think block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    # 删除残留标签
    text = text.replace("<think>", "").replace("</think>", "")

    return text.strip()


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_zh(text):
    text = normalize_space(text)
    return " ".join(jieba.cut(text))


def build_answer_prompt(question):
    return question



def build_chat_prompt(tokenizer, content):
    messages = [
        {"role": "user", "content": content}
    ]

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


def is_bad_candidate(text):
    text = clean_think(text)

    if not text:
        return True

    if len(text) < 5:
        return True

    bad_patterns = [
        "无法回答",
        "不能回答",
        "我是AI",
        "作为AI",
        "仅供娱乐",
        "不知道",
        "<think>",
        "</think>",
    ]

    return any(p in text for p in bad_patterns)


def repeat_4gram_rate(text):
    toks = list(jieba.cut(str(text)))

    if len(toks) < 8:
        return 0.0

    grams = [
        tuple(toks[i:i + 4])
        for i in range(len(toks) - 3)
    ]

    if not grams:
        return 0.0

    return 1.0 - len(set(grams)) / len(grams)


def length_penalty(candidate, reference):
    """
    防止模型靠特别长或特别短刷 ROUGE。
    返回 0~1，越接近 reference 长度越高。
    """

    c_len = max(1, len(candidate))
    r_len = max(1, len(reference))

    ratio = c_len / r_len

    if 0.5 <= ratio <= 1.8:
        return 1.0

    if ratio < 0.5:
        return max(0.0, ratio / 0.5)

    return max(0.0, 1.8 / ratio)


def rouge_score(candidate, reference):
    if not candidate or not reference:
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }

    rouge = Rouge()

    cand_tok = tokenize_zh(candidate)
    ref_tok = tokenize_zh(reference)

    try:
        score = rouge.get_scores(cand_tok, ref_tok)[0]

        return {
            "rouge-1": float(score["rouge-1"]["f"]),
            "rouge-2": float(score["rouge-2"]["f"]),
            "rouge-l": float(score["rouge-l"]["f"]),
        }

    except Exception:
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }


def score_candidate(candidate, reference):
    candidate = clean_think(candidate)
    reference = clean_think(reference)

    r = rouge_score(candidate, reference)
    lp = length_penalty(candidate, reference)
    rep = repeat_4gram_rate(candidate)

    final = (
        0.45 * r["rouge-l"]
        + 0.35 * r["rouge-1"]
        + 0.10 * r["rouge-2"]
        + 0.10 * lp
        - 0.10 * rep
    )

    return {
        "score": float(final),
        "rouge_1": r["rouge-1"],
        "rouge_2": r["rouge-2"],
        "rouge_l": r["rouge-l"],
        "length_penalty": float(lp),
        "repeat_4gram": float(rep),
        "char_len": len(candidate),
    }


def dedup_candidates(candidates):
    seen = set()
    out = []

    for c in candidates:
        c = clean_think(c).strip()

        if not c:
            continue

        key = re.sub(r"\s+", "", c)

        if key in seen:
            continue

        seen.add(key)
        out.append(c)

    return out


def select_pair(
    question,
    reference,
    candidates,
    min_score_gap=0.02,
    rejected_strategy="mixed",
    sample_idx=0,
):
    """
    rejected_strategy:
    - hard:
      选择分数低于 chosen，但尽量接近 chosen，且分差 >= min_score_gap 的候选
    - worst:
      选择分数最低的候选
    - mixed:
      一半 hard，一半 worst
    """

    candidates = dedup_candidates(candidates)
    candidates = [c for c in candidates if not is_bad_candidate(c)]

    if len(candidates) < 2:
        return None

    scored = []

    for i, c in enumerate(candidates):
        s = score_candidate(c, reference)
        scored.append({
            "idx": i,
            "answer": c,
            **s,
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    chosen_item = scored[0]
    worst_item = scored[-1]

    hard_item = None

    for item in scored[1:]:
        gap = chosen_item["score"] - item["score"]
        if gap >= min_score_gap:
            hard_item = item
            break

    if hard_item is None:
        hard_item = worst_item

    if rejected_strategy == "hard":
        rejected_item = hard_item
        actual_strategy = "hard"

    elif rejected_strategy == "worst":
        rejected_item = worst_item
        actual_strategy = "worst"

    elif rejected_strategy == "mixed":
        if sample_idx % 2 == 0:
            rejected_item = hard_item
            actual_strategy = "hard"
        else:
            rejected_item = worst_item
            actual_strategy = "worst"

    else:
        raise ValueError(f"Unknown rejected_strategy: {rejected_strategy}")

    score_gap = chosen_item["score"] - rejected_item["score"]

    if score_gap < min_score_gap:
        return None

    return {
        "prompt": question,
        "chosen": chosen_item["answer"],
        "rejected": rejected_item["answer"],
        "source": "path2_local_sft_pass5_ref_rank",
        "meta": {
            "reference": reference,
            "num_candidates_after_filter": len(scored),
            "rejected_strategy": actual_strategy,
            "score_gap": float(score_gap),
            "chosen_score": {
                k: chosen_item[k]
                for k in [
                    "score",
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                    "length_penalty",
                    "repeat_4gram",
                    "char_len",
                ]
            },
            "rejected_score": {
                k: rejected_item[k]
                for k in [
                    "score",
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                    "length_penalty",
                    "repeat_4gram",
                    "char_len",
                ]
            },
            "all_candidates": scored,
        }
    }


def generate_passk(
    llm,
    tokenizer,
    questions,
    num_candidates,
    max_tokens,
    temperature,
    top_p,
):
    prompts = [
        build_chat_prompt(tokenizer, build_answer_prompt(q))
        for q in questions
    ]

    sampling_params = SamplingParams(
        n=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    outputs = llm.generate(prompts, sampling_params)

    all_candidates = []

    for out in outputs:
        candidates = [
            clean_think(o.text.strip())
            for o in out.outputs
        ]
        all_candidates.append(candidates)

    return all_candidates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)

    parser.add_argument(
        "--rejected_strategy",
        type=str,
        default="mixed",
        choices=["hard", "worst", "mixed"],
    )

    parser.add_argument("--min_score_gap", type=float, default=0.03)

    parser.add_argument("--save_raw", action="store_true")

    args = parser.parse_args()

    raw_data = load_jsonl(args.input_file)

    samples = []

    for item in raw_data:
        q, ref = extract_sample(item)

        if not q:
            continue

        if not ref:
            continue

        samples.append({
            "question": q,
            "reference": ref,
            "raw": item,
        })

    if args.limit is not None:
        samples = samples[:args.limit]

    print(f"Loaded valid samples with reference: {len(samples)}")

    print("Loading vLLM model...")

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,

        # speed related
        enable_prefix_caching=True,
        disable_log_stats=True,
    )

    tokenizer = llm.get_tokenizer()

    questions = [s["question"] for s in samples]

    print(f"Generating pass{args.num_candidates} candidates...")
    print(f"max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print(f"max_model_len={args.max_model_len}")

    all_candidates = generate_passk(
        llm=llm,
        tokenizer=tokenizer,
        questions=questions,
        num_candidates=args.num_candidates,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Ranking candidates by reference similarity...")

    results = []
    failed = 0

    for sample_idx, (s, candidates) in enumerate(
        tqdm(
            zip(samples, all_candidates),
            total=len(samples),
            desc="Select chosen/rejected",
        )
    ):
        pair = select_pair(
            question=s["question"],
            reference=s["reference"],
            candidates=candidates,
            min_score_gap=args.min_score_gap,
            rejected_strategy=args.rejected_strategy,
            sample_idx=sample_idx,
        )

        if pair is None:
            failed += 1
            continue

        if args.save_raw:
            pair["raw"] = s["raw"]

        results.append(pair)

    save_jsonl(results, args.output_file)

    print(f"Saved to {args.output_file}")
    print(f"Valid preference pairs: {len(results)}")
    print(f"Filtered / failed: {failed}")


if __name__ == "__main__":
    main()