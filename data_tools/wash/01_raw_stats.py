#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01: Load raw CSVs and produce:
- kept.jsonl: minimally converted rows (no filtering), ShareGPT-ish structure
- dropped.jsonl: rows that are malformed/unreadable with a reason
- summary.json: basic file stats and length distributions

This script is intentionally self-contained (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


CSV_FILES = {
    "Andriatria_男科": "Andriatria_男科/男科5-13000.csv",
    "IM_内科": "IM_内科/内科5000-33000.csv",
    "OAGD_妇产科": "OAGD_妇产科/妇产科6-28000.csv",
    "Oncology_肿瘤科": "Oncology_肿瘤科/肿瘤科5-10000.csv",
    "Pediatric_儿科": "Pediatric_儿科/儿科5-14000.csv",
    "Surgical_外科": "Surgical_外科/外科5-14000.csv",
}


@dataclass
class LenStats:
    count: int = 0
    total: int = 0
    min: Optional[int] = None
    max: Optional[int] = None
    buckets: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.buckets = Counter()

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)
        if value <= 10:
            b = "<=10"
        elif value <= 30:
            b = "11-30"
        elif value <= 100:
            b = "31-100"
        elif value <= 300:
            b = "101-300"
        elif value <= 800:
            b = "301-800"
        else:
            b = ">800"
        self.buckets[b] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "avg": (self.total / self.count) if self.count else 0.0,
            "buckets": dict(self.buckets),
        }


def normalize_basic(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def is_placeholder(text: str) -> bool:
    """Return True if text is a placeholder like '无', '暂无', 'n/a' etc.

    This helps treat common filler values as empty when constructing the question.
    """
    if not text:
        return True
    t = text.strip().lower()
    # common Chinese placeholders and some latin ones
    placeholders = {"无", "暂无", "n/a", "na", "none", "-"}
    # also accept single punctuation or full-width variants already normalized by normalize_basic
    return t in placeholders


def remove_trailing_placeholder_lines(text: str) -> str:
    """Remove trailing lines that are placeholders (e.g. '\n无').

    Keeps earlier lines intact; trims only placeholder-only trailing lines.
    """
    if not text:
        return text
    lines = text.splitlines()
    # pop trailing placeholder lines
    while lines and is_placeholder(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def iter_csv_rows(path: Path, encoding: str) -> Iterable[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default="Chinese-medical-dialogue-data/Data_数据")
    ap.add_argument("--encoding", default="gb18030")
    ap.add_argument("--output_dir", default="data/dxw/step01_raw")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_path = out_dir / "kept.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    summary_path = out_dir / "summary.json"

    file_rows = {}
    file_dept_counts = {}
    file_empty_counts = {}
    title_len = LenStats()
    ask_len = LenStats()
    answer_len = LenStats()
    dropped_reasons = Counter()

    kept_n = 0
    dropped_n = 0

    with kept_path.open("w", encoding="utf-8") as kept_f, dropped_path.open("w", encoding="utf-8") as drop_f:
        for top_dept, rel in CSV_FILES.items():
            csv_path = input_root / rel
            if not csv_path.exists():
                raise FileNotFoundError(str(csv_path))

            rows = 0
            dept_counter = Counter()
            empty_counter = Counter()

            for row_id, row in iter_csv_rows(csv_path, args.encoding):
                rows += 1

                department = normalize_basic(row.get("department", ""))
                raw_title = normalize_basic(row.get("title", ""))
                raw_ask = normalize_basic(row.get("ask", ""))
                # Remove trailing placeholder lines such as '\n无' and then treat placeholders as empty
                cleaned_title = remove_trailing_placeholder_lines(raw_title)
                cleaned_ask = remove_trailing_placeholder_lines(raw_ask)
                title = "" if is_placeholder(cleaned_title) else cleaned_title
                ask = "" if is_placeholder(cleaned_ask) else cleaned_ask
                answer = normalize_basic(row.get("answer", ""))

                dept_counter[department] += 1

                for k, v in [("department", department), ("title", title), ("ask", ask), ("answer", answer)]:
                    if not v:
                        empty_counter[k] += 1

                # Minimal sanity: must have at least one of title/ask and must have answer.
                if (not title and not ask) or not answer:
                    reason = "empty_field"
                    dropped_reasons[reason] += 1
                    dropped_n += 1
                    drop_f.write(
                        json.dumps(
                            {
                                "reason": reason,
                                "top_department": top_dept,
                                "raw_file": str(csv_path),
                                "raw_row_id": row_id,
                                "row": {"department": department, "title": title, "ask": ask, "answer": answer},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                title_len.add(len(title))
                ask_len.add(len(ask))
                answer_len.add(len(answer))

                # Prefer `ask` as the actual user question; only fall back to `title` when `ask` is missing.
                question = ask if ask else (title if title else "")
                sample_id = stable_id(top_dept, department, question, answer)
                kept_f.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "conversations": [
                                {"from": "human", "value": question},
                                {"from": "gpt", "value": answer},
                            ],
                            "meta": {
                                "source": "Chinese-medical-dialogue-data",
                                "top_department": top_dept,
                                "department": department,
                                "title": title,
                                "raw_file": str(csv_path),
                                "raw_row_id": row_id,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept_n += 1

            file_rows[top_dept] = rows
            file_dept_counts[top_dept] = dept_counter.most_common(20)
            file_empty_counts[top_dept] = dict(empty_counter)

    summary = {
        "input_root": str(input_root),
        "encoding": args.encoding,
        "csv_files": CSV_FILES,
        "outputs": {"kept": str(kept_path), "dropped": str(dropped_path)},
        "counts": {"kept": kept_n, "dropped": dropped_n, "raw_total": sum(file_rows.values())},
        "dropped_reasons": dict(dropped_reasons),
        "per_file_rows": file_rows,
        "per_file_empty_counts": file_empty_counts,
        "per_file_department_top20": file_dept_counts,
        "length_stats": {"title_len": title_len.to_dict(), "ask_len": ask_len.to_dict(), "answer_len": answer_len.to_dict()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"kept": kept_n, "dropped": dropped_n, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
