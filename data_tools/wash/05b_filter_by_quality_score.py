#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 07: Filter JSONL rows by an LLM quality score threshold.

Input:
- JSONL produced by data_tools/06_llm_qa_quality_score.py, or any JSONL with
  an integer score field such as `quality_score`.

Outputs (under --output_dir):
- kept.jsonl: rows with score >= threshold
- dropped.jsonl: rows with score < threshold or bad/missing score
- summary.json: counters and score distribution
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


def maybe_total_lines(path: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["wc", "-l", path], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
        return int(out.strip().split()[0])
    except Exception:
        return None


def parse_score(obj: Dict[str, Any], field: str) -> int:
    value = obj.get(field)
    if isinstance(value, bool):
        raise ValueError("bool_is_not_score")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        raise ValueError("missing_or_invalid_score")
    if not 0 <= score <= 10:
        raise ValueError("score_out_of_range")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=int, required=True, help="Keep rows with score >= threshold.")
    ap.add_argument("--score_field", default="quality_score")
    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: stop after N input lines.")
    args = ap.parse_args()

    if not 0 <= args.threshold <= 10:
        raise SystemExit("--threshold must be an integer from 0 to 10.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = out_dir / f"kept_{args.threshold}.jsonl"
    dropped_path = out_dir / f"dropped_{args.threshold}.jsonl"
    summary_path = out_dir / f"summary_{args.threshold}.json"

    counters = Counter()
    kept_scores: Counter[int] = Counter()
    dropped_scores: Counter[int] = Counter()

    total_lines = maybe_total_lines(args.input_jsonl)
    progress_total = min(total_lines, args.max_docs) if (total_lines is not None and args.max_docs > 0) else total_lines

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, kept_path.open("w", encoding="utf-8") as fk, dropped_path.open(
        "w", encoding="utf-8"
    ) as fd:
        it = tqdm(fin, total=progress_total, desc="Filter by quality score", unit="lines")
        for line_no, line in enumerate(it, start=1):
            if args.max_docs > 0 and line_no > args.max_docs:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                counters["malformed_json"] += 1
                fd.write(json.dumps({"reason": "malformed_json", "line_no": line_no, "raw": raw[:500]}, ensure_ascii=False) + "\n")
                continue

            try:
                score = parse_score(obj, args.score_field)
            except Exception as e:
                counters["bad_score"] += 1
                dropped = dict(obj)
                dropped["drop_reason"] = str(e)
                fd.write(json.dumps(dropped, ensure_ascii=False) + "\n")
                continue

            if score >= args.threshold:
                fk.write(json.dumps(obj, ensure_ascii=False) + "\n")
                counters["kept"] += 1
                kept_scores[score] += 1
            else:
                dropped = dict(obj)
                dropped["drop_reason"] = f"{args.score_field}_below_threshold"
                fd.write(json.dumps(dropped, ensure_ascii=False) + "\n")
                counters["dropped"] += 1
                dropped_scores[score] += 1

    summary = {
        "input_jsonl": args.input_jsonl,
        "outputs": {"kept": str(kept_path), "dropped": str(dropped_path)},
        "params": {"threshold": args.threshold, "score_field": args.score_field, "max_docs": args.max_docs},
        "counts": dict(counters),
        "kept_score_distribution": {str(k): kept_scores[k] for k in range(11)},
        "dropped_score_distribution": {str(k): dropped_scores[k] for k in range(11)},
        "total_input_lines": total_lines,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"kept": counters["kept"], "dropped": counters["dropped"], "bad_score": counters["bad_score"], "out_dir": str(out_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
