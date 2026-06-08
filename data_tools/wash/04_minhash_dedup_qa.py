#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 04 (variant B): MinHash + LSH near-duplicate dedup on QUESTION + ANSWER.

Same I/O contract as 04_minhash_dedup_question.py, but uses concatenated
`question + "\\n" + answer` as the MinHash text.

Notes:
- Streaming-friendly: keeps the first occurrence; does not "replace" earlier kept items.
- Requires `datasketch` installed in the active environment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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


def build_minhash(text: str, ngram: int, num_perm: int, stop_ngrams: Optional[set[str]] = None) -> MinHash:
    # Local import so `--help` works even if `datasketch` isn't installed.
    from datasketch import MinHash  # type: ignore

    mh = MinHash(num_perm=num_perm)
    for gram in char_ngrams(text, ngram):
        if stop_ngrams and gram in stop_ngrams:
            continue
        mh.update(gram.encode("utf-8", errors="ignore"))
    return mh


def best_match(
    mh: MinHash, candidates: Iterable[str], stored: Dict[str, LeanMinHash]
) -> Tuple[Optional[str], float]:
    best_id = None
    best_sim = -1.0
    for cid in candidates:
        other = stored.get(cid)
        if other is None:
            continue
        sim = mh.jaccard(other)
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    if best_id is None:
        return None, 0.0
    return best_id, float(best_sim)

def maybe_total_lines(path: str) -> Optional[int]:
    # Use `wc -l` for speed on large files; fall back to unknown total on failure.
    try:
        out = subprocess.check_output(["wc", "-l", path], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore")
        return int(out.strip().split()[0])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ngram", type=int, default=3, help="Character n-gram size.")
    ap.add_argument("--num_perm", type=int, default=128, help="MinHash permutations.")
    ap.add_argument("--threshold", type=float, default=0.85, help="LSH Jaccard threshold.")
    ap.add_argument("--stop_ngrams_json", default="", help="Optional JSON file with `stop_ngrams` array.")
    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: stop after N docs.")
    args = ap.parse_args()

    try:
        from datasketch import LeanMinHash, MinHash, MinHashLSH  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency `datasketch`. Activate your conda env (e.g. `conda activate med`) "
            "and `pip install datasketch`."
        ) from e

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_path = out_dir / "kept.jsonl"
    dropped_path = out_dir / "dropped.jsonl"
    summary_path = out_dir / "summary.json"

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    stored: Dict[str, LeanMinHash] = {}

    stop_ngrams: set[str] = set()
    if args.stop_ngrams_json:
        try:
            j = json.loads(Path(args.stop_ngrams_json).read_text(encoding="utf-8"))
            stop_ngrams = set(map(str, j.get("stop_ngrams", [])))
        except Exception:
            raise SystemExit(f"Failed to load --stop_ngrams_json: {args.stop_ngrams_json}")

    kept_n = 0
    drop_n = 0
    dropped_reasons = Counter()
    kept_top = Counter()
    dropped_top = Counter()

    total_lines = maybe_total_lines(args.input_jsonl)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, kept_path.open("w", encoding="utf-8") as fk, dropped_path.open(
        "w", encoding="utf-8"
    ) as fd:
        it = tqdm(fin, total=total_lines, desc="MinHash dedup (q+a)", unit="lines")
        for i, line in enumerate(it, start=1):
            if args.max_docs > 0 and i > args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sample_id = str(obj.get("id", ""))
                q = obj["conversations"][0]["value"]
                a = obj["conversations"][1]["value"]
            except Exception:
                dropped_reasons["bad_schema"] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "bad_schema", "line_no": i, "raw": line[:500]}, ensure_ascii=False) + "\n")
                continue

            meta = obj.get("meta", {}) or {}
            top_dept = str(meta.get("top_department", ""))

            norm = normalize_for_minhash(q + "\n" + a)
            if not norm:
                dropped_reasons["empty_after_normalize"] += 1
                dropped_top[top_dept] += 1
                drop_n += 1
                fd.write(json.dumps({"reason": "empty_after_normalize", "id": sample_id, "meta": meta}, ensure_ascii=False) + "\n")
                continue

            mh = build_minhash(norm, args.ngram, args.num_perm, stop_ngrams=stop_ngrams)
            candidates = lsh.query(mh)
            if candidates:
                mid, sim = best_match(mh, candidates, stored)
                if mid is not None and sim >= args.threshold:
                    dropped_reasons["near_duplicate"] += 1
                    dropped_top[top_dept] += 1
                    drop_n += 1
                    fd.write(
                        json.dumps(
                            {
                                "reason": "near_duplicate",
                                "id": sample_id,
                                "matched_id": mid,
                                "estimated_jaccard": sim,
                                "meta": meta,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

            fk.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept_n += 1
            kept_top[top_dept] += 1
            lean = LeanMinHash(mh)
            stored[sample_id] = lean
            lsh.insert(sample_id, lean)

    summary = {
        "input_jsonl": args.input_jsonl,
        "outputs": {"kept": str(kept_path), "dropped": str(dropped_path)},
        "params": {
            "ngram": args.ngram,
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "stop_ngrams_json": args.stop_ngrams_json,
            "stop_ngrams_count": len(stop_ngrams),
        },
        "counts": {"kept": kept_n, "dropped": drop_n},
        "dropped_reasons": dict(dropped_reasons),
        "kept_top_department_counts": dict(kept_top),
        "dropped_top_department_counts": dict(dropped_top),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kept": kept_n, "dropped": drop_n, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
