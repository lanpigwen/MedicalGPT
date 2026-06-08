#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a data-driven stop-ngram list for MinHash dedup.

Motivation
----------
In Chinese medical questions, high-frequency "template" phrases (e.g. 怎么办/如何治疗/是什么)
can dominate character n-gram MinHash similarity, causing false positives:
"糖尿病怎么治疗" vs "低血糖怎么治疗".

This script estimates high document-frequency (DF) character n-grams and outputs a stop list.
You can then pass the result to:
  - 04_minhash_dedup_question.py --stop_ngrams_json stop_ngrams.json
  - 04_minhash_dedup_qa.py       --stop_ngrams_json stop_ngrams.json

Approach
--------
We only need the *most frequent* n-grams, not a full DF table.
So we use a streaming heavy-hitters algorithm (Misra-Gries) on per-document unique n-grams
to get candidates, then do a second pass to compute exact DF for those candidates.

Input JSONL schema (same as other steps):
  {"id":..., "conversations":[{"from":"human","value":...},{"from":"gpt","value":...}], "meta": {...}}

Output JSON:
  {
    "input_jsonl": "...",
    "ngram": 3,
    "doc_count": 123,
    "params": {...},
    "stop_ngrams": ["怎么治", ...],
    "candidates_stats": [{"ngram":"怎么治","df":100,"df_ratio":0.81}, ...]
  }
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[，,。.!！?？；;：:\-—_、\"'“”‘’()\[\]{}<>《》【】]")


def normalize_for_minhash(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s).lower()
    s = SPACE_RE.sub("", s)
    s = PUNCT_RE.sub("", s)
    return s


def char_ngrams(text: str, n: int) -> Iterable[str]:
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return (text[i : i + n] for i in range(0, len(text) - n + 1))


def maybe_total_lines(path: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["wc", "-l", path], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
        return int(out.strip().split()[0])
    except Exception:
        return None


def misra_gries_update(counters: Dict[str, int], items: Iterable[str], capacity: int) -> None:
    """
    Streaming heavy hitters (Misra-Gries).
    We update once per document using per-document unique n-grams to approximate DF-heavy hitters.
    """
    for it in items:
        if it in counters:
            counters[it] += 1
            continue
        if len(counters) < capacity:
            counters[it] = 1
            continue

        # Decrement all; remove zeros.
        to_del: List[str] = []
        for k in counters:
            counters[k] -= 1
            if counters[k] <= 0:
                to_del.append(k)
        for k in to_del:
            del counters[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: stop after N docs.")
    ap.add_argument("--candidates", type=int, default=20000, help="Target candidate set size (approx).")
    ap.add_argument("--capacity_mult", type=int, default=5, help="Misra-Gries capacity = candidates * mult.")
    ap.add_argument("--min_df", type=int, default=50, help="Minimum DF to include in stop list.")
    ap.add_argument(
        "--min_df_ratio",
        type=float,
        default=0.02,
        help="Minimum DF ratio to include in stop list (default 2%).",
    )
    ap.add_argument("--top_k", type=int, default=5000, help="Keep at most top-k by DF in output.")
    ap.add_argument(
        "--template_chars",
        default="怎么|如何|请问|是什么|怎么办|治疗|原因|症状|方法|能否|可以|是否|会不会|怎么做|怎样",
        help="Template hints (pipe-separated). Keep only n-grams matching any hint. Empty disables.",
    )
    args = ap.parse_args()

    total_lines = maybe_total_lines(args.input_jsonl)
    capacity = max(1000, int(args.candidates) * int(args.capacity_mult))

    # Pass 1: heavy hitters candidates.
    hh: Dict[str, int] = {}
    doc_count = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        it = tqdm(fin, total=total_lines, desc="Build stop-ngrams (pass1)", unit="lines")
        for i, line in enumerate(it, start=1):
            if args.max_docs > 0 and doc_count >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj["conversations"][0]["value"]
            except Exception:
                continue

            norm = normalize_for_minhash(q)
            if not norm:
                continue

            grams = set(char_ngrams(norm, args.ngram))
            if not grams:
                continue
            misra_gries_update(hh, grams, capacity=capacity)
            doc_count += 1

    candidates = set(hh.keys())

    # Pass 2: exact DF for candidates.
    df = Counter()
    doc_count_2 = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        it = tqdm(fin, total=total_lines, desc="Build stop-ngrams (pass2)", unit="lines")
        for i, line in enumerate(it, start=1):
            if args.max_docs > 0 and doc_count_2 >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj["conversations"][0]["value"]
            except Exception:
                continue

            norm = normalize_for_minhash(q)
            if not norm:
                continue

            grams = set(char_ngrams(norm, args.ngram))
            if not grams:
                continue

            for g in grams:
                if g in candidates:
                    df[g] += 1
            doc_count_2 += 1

    doc_count_final = doc_count_2 if doc_count_2 > 0 else doc_count
    if doc_count_final <= 0:
        raise SystemExit("No valid documents found; cannot build stop list.")

    rows: List[Tuple[str, int, float]] = []
    for g, c in df.items():
        ratio = c / float(doc_count_final)
        if c < args.min_df:
            continue
        if ratio < args.min_df_ratio:
            continue
        if args.template_chars:
            # Template-biased filter: keep only n-grams that contain at least one template hint.
            # This avoids accidentally dropping frequent medical entities.
            hints = [h for h in args.template_chars.split("|") if h]
            if hints and not any(h in g for h in hints):
                continue
        rows.append((g, c, ratio))

    rows.sort(key=lambda x: (-x[1], -x[2], x[0]))
    rows = rows[: max(0, int(args.top_k))]

    stop_ngrams = [g for (g, _, _) in rows]
    out = {
        "input_jsonl": args.input_jsonl,
        "ngram": args.ngram,
        "doc_count": doc_count_final,
        "params": {
            "max_docs": args.max_docs,
            "candidates": args.candidates,
            "capacity_mult": args.capacity_mult,
            "capacity": capacity,
            "min_df": args.min_df,
            "min_df_ratio": args.min_df_ratio,
            "top_k": args.top_k,
        },
        "stop_ngrams": stop_ngrams,
        "candidates_stats": [{"ngram": g, "df": c, "df_ratio": r} for (g, c, r) in rows],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stop_ngrams": len(stop_ngrams), "doc_count": doc_count_final, "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
