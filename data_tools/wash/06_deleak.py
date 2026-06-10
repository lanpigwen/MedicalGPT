#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_deleak_fixed.py

deleak_train 脚本，修复 MinHash LSH 插入重复 key 报错。
"""

import json
import argparse
import unicodedata
import re
from pathlib import Path

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None
    MinHashLSH = None

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[，,。.!！?？；;：:\-—_、\"'“”‘’()\[\]{}<>《》【】]")

def normalize_for_exact(text):
    if not text:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s).lower()
    s = SPACE_RE.sub(" ", s).strip()
    return s

def normalize_for_minhash(text):
    if not text:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s).lower()
    s = SPACE_RE.sub("", s)
    s = PUNCT_RE.sub("", s)
    return s

def get_question(obj):
    conv = obj.get("conversations") or []
    if conv:
        return conv[0].get("value", "") or ""
    return obj.get("question") or obj.get("Question") or ""

def build_lsh(eval_rows, ngram=3, num_perm=64, threshold=0.9):
    if MinHash is None:
        raise RuntimeError("datasketch not installed. pip install datasketch")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, r in enumerate(eval_rows):
        q = normalize_for_minhash(get_question(r))
        if not q:
            continue
        mh = MinHash(num_perm=num_perm)
        for gram in [q[i:i+ngram] for i in range(len(q)-ngram+1)]:
            mh.update(gram.encode("utf-8"))
        # 用 idx 保证唯一 key
        key = str(r.get("id")) if r.get("id") is not None else f"eval_{idx}"
        lsh.insert(key, mh)
    return lsh

def deleak_train(train_rows, eval_rows, use_exact=True, use_minhash=False, ngram=3, num_perm=64, threshold=0.9):
    kept = []
    dropped = []

    # 1. exact match
    eval_exact = set()
    if use_exact:
        for idx, r in enumerate(eval_rows):
            q = normalize_for_exact(get_question(r))
            if q:
                eval_exact.add(q)

    # 2. minhash
    lsh = build_lsh(eval_rows, ngram=ngram, num_perm=num_perm, threshold=threshold) if use_minhash else None

    for idx, r in enumerate(train_rows):
        q_raw = get_question(r)
        q_norm = normalize_for_exact(q_raw)
        drop_reason = None

        # exact duplicate
        if use_exact and q_norm in eval_exact:
            drop_reason = "exact_duplicate"
        # near duplicate via minhash
        elif use_minhash and lsh:
            q_mh_norm = normalize_for_minhash(q_raw)
            mh = MinHash(num_perm=num_perm)
            for gram in [q_mh_norm[i:i+ngram] for i in range(len(q_mh_norm)-ngram+1)]:
                mh.update(gram.encode("utf-8"))
            candidates = lsh.query(mh)
            if candidates:
                drop_reason = "minhash_near_duplicate"

        if drop_reason:
            dropped.append({"id": r.get("id"), "reason": drop_reason, "sample": r})
            continue

        kept.append(r)

    return kept, dropped

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_exact", action="store_true")
    parser.add_argument("--use_minhash", action="store_true")
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--num_perm", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)

    print(f"Train samples: {len(train_rows)}, Eval samples: {len(eval_rows)}")

    kept, dropped = deleak_train(
        train_rows, eval_rows,
        use_exact=args.use_exact,
        use_minhash=args.use_minhash,
        ngram=args.ngram,
        num_perm=args.num_perm,
        threshold=args.threshold
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_cleaned.jsonl"
    dropped_out = out_dir / "dropped_from_train.jsonl"

    save_jsonl(kept, train_out)
    save_jsonl(dropped, dropped_out)

    print(f"Kept: {len(kept)}, Dropped: {len(dropped)}")
    print(f"Saved train_cleaned.jsonl and dropped_from_train.jsonl under {out_dir}")

if __name__ == "__main__":
    main()