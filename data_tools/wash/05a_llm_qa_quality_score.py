#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 06: Score Chinese medical QA quality with an OpenAI-compatible GPT API.

Input:
- ShareGPT-ish JSONL:
  {"id":..., "conversations":[{"from":"human","value":...},{"from":"gpt","value":...}], "meta": {...}}

Output:
- result.jsonl by default: original sample plus `quality_score`
- failed.jsonl: malformed rows or API/parse failures
- summary.json: counters and score distribution

The model is instructed to output only one integer from 0 to 10.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


SYSTEM_PROMPT = (
    "你是一个严格的中文医疗问答语料质检员。你的任务是给用户问题和助手回答组成的问答对打质量分。"
    "高质量问答对应满足：属于医疗/健康领域；问题语义清楚、具体、不是乱码或广告；"
    "回答与问题匹配，内容通顺、有帮助、不过度空泛，不包含联系方式、推广、无关闲聊或明显危险建议。"
    "医疗/健康领域包括：疾病、症状、诊断、检查、用药、治疗、手术、护理、疫苗、检验指标、健康管理、生育、妇产、儿科、男科等。"
    "低质量包括：非医疗问题、问题残缺/乱码/模板痕迹严重、回答答非所问、回答为空、广告营销、违法违规内容、纯闲聊、情感鸡汤、财经、编程、法律、政治、娱乐、游戏、学习作业等。"
)


QUESTION_ONLY_SYSTEM_PROMPT = (
    "你是一个严格的中文医疗语料质检员。你的任务是给用户问题本身打质量分。"
    "高质量问题应满足：属于医疗/健康领域，语义清楚、具体、可回答，不是乱码、广告、纯标题或无意义模板。"
    "医疗/健康领域包括：疾病、症状、诊断、检查、用药、治疗、手术、护理、疫苗、检验指标、健康管理、生育、妇产、儿科、男科等。"
    "低质量包括：非医疗问题、残缺乱码、广告营销、纯闲聊、情感鸡汤、财经、编程、法律、政治、娱乐、游戏、学习作业等。"
)


INT_RE = re.compile(r"\b(?:10|[0-9])\b")


def maybe_total_lines(path: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["wc", "-l", path], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
        return int(out.strip().split()[0])
    except Exception:
        return None


def get_question_answer(obj: Dict[str, Any]) -> Tuple[str, str]:
    conv = obj.get("conversations") or []
    if len(conv) < 2:
        raise ValueError("bad_schema")
    q = conv[0].get("value")
    a = conv[1].get("value")
    return "" if q is None else str(q), "" if a is None else str(a)


def build_user_prompt(question: str, answer: str, question_only: bool = False) -> str:
    """Return a prompt that asks the model to output a single integer 0-10."""
    if question_only:
        return (
            "请判断下面这个【问题】的语料质量。\n"
            "只输出一个整数分数 0 到 10（不要输出任何其他字符）。\n"
            "10 表示非常高质量；0 表示非常低质量。\n\n"
            "评分参考：\n"
            "8-10：医疗/健康问题明确、自然、具体、可回答。\n"
            "4-7：大致是医疗问题，但表达不够清楚、过短、模板化或信息不足。\n"
            "0-3：非医疗、乱码、广告、无意义、不可回答或严重残缺。\n\n"
            f"【问题】\n{question}\n"
        )
    return (
        "请判断下面这个【问答对】的整体语料质量。\n"
        "只输出一个整数分数 0 到 10（不要输出任何其他字符）。\n"
        "10 表示非常高质量；0 表示非常低质量。\n\n"
        "评分参考：\n"
        "8-10：医疗/健康领域明确，问题清楚，回答相关、通顺、有帮助，无广告和明显危险建议。\n"
        "4-7：基本可用，但问题或回答存在轻微模板化、信息不足、泛泛而谈或相关性一般。\n"
        "0-3：非医疗、乱码、广告、答非所问、回答为空、严重不通顺或明显不适合训练。\n\n"
        f"【问题】\n{question}\n\n"
        f"【回答】\n{answer}\n"
    )


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def parse_score(text: str) -> int:
    text = text.strip()
    if text.isdigit() and 0 <= int(text) <= 10:
        return int(text)
    m = INT_RE.search(text)
    if not m:
        raise ValueError(f"cannot_parse_score: {text[:100]}")
    score = int(m.group(0))
    if not 0 <= score <= 10:
        raise ValueError(f"score_out_of_range: {score}")
    return score


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    disable_think: bool,
) -> str:
    url = normalize_base_url(base_url) + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    return str(obj["choices"][0]["message"]["content"])


def call_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
    disable_think: bool,
) -> Tuple[int, str]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            raw = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                disable_think=disable_think,
            )
            return parse_score(raw), raw
        except Exception as e:
            last_error = e
            if attempt >= retries:
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)) + random.random() * 0.2)
    raise RuntimeError(str(last_error))


def iter_rows(input_jsonl: str, max_docs: int) -> Iterable[Tuple[int, Dict[str, Any], str]]:
    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if max_docs > 0 and line_no > max_docs:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw), raw
            except Exception:
                yield line_no, {"__malformed_json__": True}, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", default="", help="Default: <output_dir>/result.jsonl.")
    ap.add_argument("--output_dir", default="", help="Default: parent dir of --output_jsonl, or data/dxw/step06_llm_qa_quality_score.")
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--score_field", default="quality_score")
    ap.add_argument("--raw_field", default="", help="Optional field name to store raw model text, e.g. quality_score_raw.")
    ap.add_argument("--question_only", action="store_true", help="Score only the question, ignoring the answer.")
    ap.add_argument("--disable_think", action="store_true", help="For Qwen3-like models: request no thinking and append /no_think.")
    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: stop after N input lines.")
    args = ap.parse_args()

    if args.output_jsonl:
        result_path = Path(args.output_jsonl)
        out_dir = Path(args.output_dir) if args.output_dir else result_path.parent
    else:
        out_dir = Path(args.output_dir) if args.output_dir else Path("data/dxw/step06_llm_qa_quality_score")
        result_path = out_dir / "result.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / "failed.jsonl"
    summary_path = out_dir / "summary.json"

    result_lock = Lock()
    failed_lock = Lock()
    score_counter: Counter[int] = Counter()
    counters = Counter()

    system_prompt = QUESTION_ONLY_SYSTEM_PROMPT if args.question_only else SYSTEM_PROMPT

    def handle(row: Tuple[int, Dict[str, Any], str]) -> Optional[Dict[str, Any]]:
        line_no, obj, raw = row
        if obj.get("__malformed_json__"):
            return {"failed": {"reason": "malformed_json", "line_no": line_no, "raw": raw[:500]}}
        try:
            question, answer = get_question_answer(obj)
            if not question or ((not args.question_only) and not answer):
                return {"failed": {"reason": "empty_question_or_answer", "line_no": line_no, "id": obj.get("id")}}
            user_prompt = build_user_prompt(question, answer, question_only=args.question_only)
            if args.disable_think:
                user_prompt = user_prompt.rstrip() + "\n/no_think\n"
            score, raw_score = call_with_retries(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                disable_think=args.disable_think,
            )
            obj[args.score_field] = score
            if args.raw_field:
                obj[args.raw_field] = raw_score.strip()
            return {"ok": obj, "score": score}
        except Exception as e:
            return {"failed": {"reason": "score_failed", "line_no": line_no, "id": obj.get("id"), "error": str(e)[:500]}}

    total_lines = maybe_total_lines(args.input_jsonl)
    progress_total = min(total_lines, args.max_docs) if (total_lines is not None and args.max_docs > 0) else total_lines
    rows = iter_rows(args.input_jsonl, args.max_docs)
    with result_path.open("w", encoding="utf-8") as fout, failed_path.open("w", encoding="utf-8") as ffail:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            pending = set()
            future_to_seq: Dict[Any, int] = {}
            completed: Dict[int, Optional[Dict[str, Any]]] = {}
            next_submit = 0
            next_write = 0
            row_iter = iter(rows)
            max_pending = max(1, args.workers) * 4
            pbar = tqdm(total=progress_total, desc="LLM QA quality score", unit="rows")

            def write_item(item: Optional[Dict[str, Any]]) -> None:
                if not item:
                    return
                if "ok" in item:
                    with result_lock:
                        fout.write(json.dumps(item["ok"], ensure_ascii=False) + "\n")
                    score_counter[int(item["score"])] += 1
                    counters["scored"] += 1
                else:
                    with failed_lock:
                        ffail.write(json.dumps(item["failed"], ensure_ascii=False) + "\n")
                    counters[str(item["failed"].get("reason", "failed"))] += 1

            def consume_done(done: Iterable[Any]) -> None:
                nonlocal next_write
                for fut in done:
                    seq = future_to_seq.pop(fut)
                    completed[seq] = fut.result()
                    pbar.update(1)
                while next_write in completed:
                    write_item(completed.pop(next_write))
                    next_write += 1

            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < max_pending:
                    try:
                        fut = ex.submit(handle, next(row_iter))
                        pending.add(fut)
                        future_to_seq[fut] = next_submit
                        next_submit += 1
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume_done(done)
            pbar.close()

    summary = {
        "input_jsonl": args.input_jsonl,
        "outputs": {"result": str(result_path), "failed": str(failed_path)},
        "params": {
            "base_url": args.base_url,
            "model": args.model,
            "workers": args.workers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "score_field": args.score_field,
            "question_only": args.question_only,
            "disable_think": args.disable_think,
            "max_docs": args.max_docs,
        },
        "counts": dict(counters),
        "score_distribution": {str(k): score_counter[k] for k in range(11)},
        "total_input_lines": total_lines,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scored": counters["scored"], "failed": sum(v for k, v in counters.items() if k != "scored"), "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
