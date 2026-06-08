#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 03: Exact deduplication.

Input:
- A cleaned jsonl (e.g. step02 kept.jsonl) with ShareGPT-ish schema.

Dedup rules:
- exact_pair_duplicate: same normalized(question + answer) already seen -> drop
- question_duplicate: same normalized(question) already seen -> keep the one with longer visible answer

Outputs:
- kept.jsonl
- dropped.jsonl (with reason and linkage to kept sample id when replaced)
- question_duplicates.jsonl (all question-level duplicate events)
- summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[，,。.!！?？；;：:\-—_、\"'“”‘’()\[\]{}<>《》【】]")


def normalize_for_key(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = SPACE_RE.sub("", text)
    text = PUNCT_RE.sub("", text)
    return text


def visible_len(text: str) -> int:
    return len(SPACE_RE.sub("", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", default="data/dxw/step03_exact_dedup")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = out_dir / "kept.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    question_duplicates_path = out_dir / "question_duplicates.jsonl"
    summary_path = out_dir / "summary.json"

    dropped = Counter()
    kept_n = 0
    drop_n = 0

    seen_pair = set()
    kept_by_question: Dict[str, Dict[str, Any]] = {}
    kept_answer_len_by_question: Dict[str, int] = {}

    # 先做完全重复对的去重，再做 question 维度的去重。
    # question_duplicate 采用“保留更长答案”的策略，因此需要记录当前 question 下的最佳样本。
    question_duplicate_groups: Dict[str, Dict[str, Any]] = {}

    with open(
        args.input_jsonl,
        "r",
        encoding="utf-8",
    ) as fin, dropped_path.open("w", encoding="utf-8") as fd, question_duplicates_path.open(
        "w",
        encoding="utf-8",
    ) as fq:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                dropped["malformed_json"] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "malformed_json", "line_no": line_no, "raw": line[:500]}, ensure_ascii=False) + "\n")
                continue

            try:
                q = obj["conversations"][0]["value"]
                a = obj["conversations"][1]["value"]
            except Exception:
                dropped["bad_schema"] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "bad_schema", "line_no": line_no, "id": obj.get("id")}, ensure_ascii=False) + "\n")
                continue

            # 归一化后再比较，避免空格、标点、大小写差异导致重复样本漏检。
            q_key = normalize_for_key(q)
            pair_key = normalize_for_key(q + "\n" + a)

            if pair_key in seen_pair:
                dropped["exact_pair_duplicate"] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "exact_pair_duplicate", "id": obj.get("id"), "meta": obj.get("meta", {})}, ensure_ascii=False) + "\n")
                continue
            seen_pair.add(pair_key)

            a_len = visible_len(a)
            if q_key not in kept_by_question:
                kept_by_question[q_key] = obj
                kept_answer_len_by_question[q_key] = a_len
                continue

            if q_key not in question_duplicate_groups:
                question_duplicate_groups[q_key] = {
                    "question": q,
                    "question_key": q_key,
                    "first_kept_id": kept_by_question[q_key].get("id"),
                    "duplicates": [],
                }

            # 同一个 question 只保留 visible answer 更长的那条，优先保留信息量更大的样本。
            old_len = kept_answer_len_by_question[q_key]
            if a_len > old_len:
                replaced = kept_by_question[q_key]
                kept_by_question[q_key] = obj
                kept_answer_len_by_question[q_key] = a_len
                question_duplicate_groups[q_key]["duplicates"].append(
                    {
                        "reason": "question_duplicate_replaced",
                        "dropped_id": replaced.get("id"),
                        "kept_id": obj.get("id"),
                        "dropped_answer_len": old_len,
                        "kept_answer_len": a_len,
                        "dropped_meta": replaced.get("meta", {}),
                        "kept_meta": obj.get("meta", {}),
                        "line_no": line_no,
                    }
                )
                # record replaced old sample as dropped
                dropped["question_duplicate_replaced"] += 1
                drop_n += 1
                fd.write(
                    json.dumps(
                        {
                            "reason": "question_duplicate_replaced",
                            "question": q,
                            "question_key": q_key,
                            "dropped_id": replaced.get("id"),
                            "kept_id": obj.get("id"),
                            "meta": replaced.get("meta", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # record current sample as kept; no drop record for current
            else:
                question_duplicate_groups[q_key]["duplicates"].append(
                    {
                        "reason": "question_duplicate",
                        "dropped_id": obj.get("id"),
                        "kept_id": kept_by_question[q_key].get("id"),
                        "dropped_answer_len": a_len,
                        "kept_answer_len": old_len,
                        "dropped_meta": obj.get("meta", {}),
                        "kept_meta": kept_by_question[q_key].get("meta", {}),
                        "line_no": line_no,
                    }
                )
                dropped["question_duplicate"] += 1
                drop_n += 1
                fd.write(
                    json.dumps(
                        {
                            "reason": "question_duplicate",
                            "question": q,
                            "question_key": q_key,
                            "dropped_id": obj.get("id"),
                            "kept_id": kept_by_question[q_key].get("id"),
                            "meta": obj.get("meta", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # Write kept in a stable order for reproducibility.
    # 按科室、来源文件、原始行号排序，保证每次运行输出顺序一致，方便 diff 和复现。
    kept_list = list(kept_by_question.values())
    kept_list.sort(key=lambda x: (str(x.get("meta", {}).get("top_department", "")), str(x.get("meta", {}).get("raw_file", "")), int(x.get("meta", {}).get("raw_row_id", 0))))

    aggregated_question_duplicates = []
    for q_key, group in question_duplicate_groups.items():
        final_kept = kept_by_question.get(q_key, {})
        aggregated_question_duplicates.append(
            {
                "question": group["question"],
                "question_key": group["question_key"],
                "first_kept_id": group["first_kept_id"],
                "final_kept_id": final_kept.get("id"),
                "final_kept_answer_len": kept_answer_len_by_question.get(q_key),
                "duplicate_count": len(group["duplicates"]),
                "duplicates": group["duplicates"],
            }
        )
    aggregated_question_duplicates.sort(key=lambda x: x["question_key"])

    top_10_question_duplicates_by_department: Dict[str, list[Dict[str, Any]]] = {}
    department_groups: Dict[str, list[Dict[str, Any]]] = {}
    for item in aggregated_question_duplicates:
        if item["duplicate_count"] <= 0:
            continue
        final_kept = kept_by_question.get(item["question_key"], {})
        meta = final_kept.get("meta", {}) if isinstance(final_kept, dict) else {}
        department = str(meta.get("top_department", "UNKNOWN") or "UNKNOWN")
        department_groups.setdefault(department, []).append(
            {
                "question": item["question"],
                "question_key": item["question_key"],
                "duplicate_count": item["duplicate_count"],
                "final_kept_id": item["final_kept_id"],
                "final_kept_answer_len": item["final_kept_answer_len"],
            }
        )

    for department, items in department_groups.items():
        items.sort(key=lambda x: (-x["duplicate_count"], x["question_key"]))
        top_10_question_duplicates_by_department[department] = items[:10]

    with question_duplicates_path.open("w", encoding="utf-8") as fq:
        for item in aggregated_question_duplicates:
            fq.write(json.dumps(item, ensure_ascii=False) + "\n")

    with kept_path.open("w", encoding="utf-8") as fk:
        for obj in kept_list:
            fk.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept_n += 1

    summary = {
        "input_jsonl": args.input_jsonl,
        "outputs": {
            "kept": str(kept_path),
            "dropped": str(dropped_path),
            "question_duplicates": str(question_duplicates_path),
        },
        "counts": {"kept": kept_n, "dropped": drop_n},
        "dropped_reasons": dict(dropped),
        "top_10_question_duplicates_by_department": top_10_question_duplicates_by_department,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kept": kept_n, "dropped": drop_n, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

