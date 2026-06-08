#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 02: Base filtering + text cleanup.

Inputs:
- A jsonl like step01 output (kept.jsonl), with fields:
  id, conversations[{from,value}], meta{...}

Outputs (under --output_dir):
- kept.jsonl: passed samples (cleaned)
- dropped.jsonl: removed samples with reason
- summary.json: counters + key stats

Notes:
- No length-based filtering (per user request).
- No MinHash / PPL / embedding / LLM scoring.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")
CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CONTACT_RE = re.compile(
    r"(微信|微.?信|QQ|qq群|电话|手机号|热线|加群|公众号|扫一扫|二维码|vx|wechat)",
    re.IGNORECASE,
)


@dataclass
class LenStats:
    count: int = 0
    total: int = 0
    min: Optional[int] = None
    max: Optional[int] = None

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "avg": (self.total / self.count) if self.count else 0.0,
        }


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\\n", "\n").replace("\\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def count_chinese(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def chinese_ratio(text: str) -> float:
    v = visible_len(text)
    if v == 0:
        return 0.0
    return count_chinese(text) / v


def looks_like_contact_ad(text: str) -> bool:
    if URL_RE.search(text):
        return True
    if CONTACT_RE.search(text):
        return True
    if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", default="data/dxw/step02_prefilter")
    ap.add_argument("--min_chinese_chars_question", type=int, default=2)
    ap.add_argument("--min_chinese_chars_answer", type=int, default=1)
    ap.add_argument("--min_chinese_ratio", type=float, default=0.20)
    ap.add_argument("--keep_contact_ad", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_path = out_dir / "kept.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    summary_path = out_dir / "summary.json"

    dropped = Counter()
    top_dept_kept = Counter()
    sub_dept_kept = Counter()
    q_len = LenStats()
    a_len = LenStats()
    kept_n = 0
    drop_n = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, kept_path.open("w", encoding="utf-8") as fk, dropped_path.open(
        "w", encoding="utf-8"
    ) as fd:
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
                question_raw = obj["conversations"][0]["value"]
                answer_raw = obj["conversations"][1]["value"]
            except Exception:
                dropped["bad_schema"] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "bad_schema", "line_no": line_no, "id": obj.get("id")}, ensure_ascii=False) + "\n")
                continue

            question = normalize_text(question_raw)
            answer = normalize_text(answer_raw)

            reason: Optional[str] = None
            if not question or not answer:
                reason = "empty_field"
            elif count_chinese(question) < args.min_chinese_chars_question or count_chinese(answer) < args.min_chinese_chars_answer:
                reason = "non_chinese"
            elif chinese_ratio(question + "\n" + answer) < args.min_chinese_ratio:
                reason = "non_chinese"
            elif (not args.keep_contact_ad) and looks_like_contact_ad(question + "\n" + answer):
                reason = "contact_or_url"

            if reason:
                dropped[reason] += 1
                drop_n += 1
                fd.write(
                    json.dumps(
                        {
                            "reason": reason,
                            "id": obj.get("id"),
                            "meta": obj.get("meta", {}),
                            "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            # keep
            obj["conversations"][0]["value"] = question
            obj["conversations"][1]["value"] = answer
            fk.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept_n += 1

            meta = obj.get("meta", {}) or {}
            top = str(meta.get("top_department", ""))
            sub = str(meta.get("department", ""))
            top_dept_kept[top] += 1
            sub_dept_kept[sub] += 1
            q_len.add(visible_len(question))
            a_len.add(visible_len(answer))

    summary = {
        "input_jsonl": args.input_jsonl,
        "outputs": {"kept": str(kept_path), "dropped": str(dropped_path)},
        "counts": {"kept": kept_n, "dropped": drop_n},
        "dropped_reasons": dict(dropped),
        "kept_top_department_counts": dict(top_dept_kept),
        "kept_sub_department_top50": sub_dept_kept.most_common(50),
        "question_visible_len": q_len.to_dict(),
        "answer_visible_len": a_len.to_dict(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kept": kept_n, "dropped": drop_n, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

