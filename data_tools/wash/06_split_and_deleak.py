#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 05: Split train/val/test from a candidate pool AND de-leak training data
using val/test as "no-peek" holdout.

What this script does
---------------------
1) Read a ShareGPT-ish JSONL pool:
     {"id":..., "conversations":[{"from":"human","value":...},{"from":"gpt","value":...}], "meta": {...}}
2) Stratified split by a field (default: meta.top_department) into train/val/test.
   Two modes:
   - ratio mode: use --val_ratio/--test_ratio
   - target mode: use --val_target/--test_target (+ --eval_oversample, optional eval internal de-dup)
3) De-leak training set by removing any train samples that are:
   - exact duplicates of val/test by normalized question; and/or
   - near-duplicates found via MinHash+LSH query with val/test questions.

Outputs (under --output_dir)
----------------------------
- train.jsonl / val.jsonl / test.jsonl
- dropped_from_eval.jsonl: removed eval (val/test) items due to eval-internal de-dup
- dropped_from_train.jsonl: removed training items with reason + matched eval id/split
- summary.json: counts + per-department breakdown + deleak stats

Notes on near-duplicate deleak
------------------------------
MinHash+LSH deleak is designed to reduce evaluation contamination.
By default, we drop all LSH-returned candidates (conservative). This avoids storing
all training signatures in RAM. If you want to verify Jaccard scores, you can enable
--verify_jaccard, but it may consume substantial memory on large corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover

    def tqdm(it, *args, **kwargs):  # type: ignore
        return it


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[，,。.!！?？；;：:\-—_、\"'“”‘’()\[\]{}<>《》【】]")


def normalize_for_exact(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s).lower()
    s = SPACE_RE.sub(" ", s).strip()
    return s


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


def get_question(obj: Dict[str, Any]) -> str:
    conv = obj.get("conversations") or []
    if not conv:
        return ""
    v = conv[0].get("value")
    return "" if v is None else str(v)


def get_group(obj: Dict[str, Any], stratify_field: str) -> str:
    if not stratify_field:
        return ""
    # Only support "meta.xxx" and top-level "xxx" for now.
    if stratify_field.startswith("meta."):
        key = stratify_field.split(".", 1)[1]
        meta = obj.get("meta") or {}
        return str(meta.get(key, "") or "")
    return str(obj.get(stratify_field, "") or "")


def stable_hash(s: str) -> int:
    # Deterministic across runs/machines.
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:16], 16)


def split_indices(items: Sequence[Dict[str, Any]], val_ratio: float, test_ratio: float, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
    n = len(items)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)

    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test : n_test + n_val])
    train_idx = set(idx[n_test + n_val :])
    return train_idx, val_idx, test_idx


def stratified_split(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_field: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[get_group(r, stratify_field)].append(r)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for g, items in groups.items():
        # Group-specific seed so the split is stable even if group iteration order changes.
        g_seed = seed ^ stable_hash(g)
        tr_i, va_i, te_i = split_indices(items, val_ratio=val_ratio, test_ratio=test_ratio, seed=g_seed)
        for i, it in enumerate(items):
            if i in te_i:
                test.append(it)
            elif i in va_i:
                val.append(it)
            else:
                train.append(it)

    # Shuffle each split for nicer downstream behavior.
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def proportional_quotas(group_sizes: Dict[str, int], target: int) -> Dict[str, int]:
    """
    Allocate integer quotas across groups proportional to their sizes.
    Ensures sum(quotas) == target (when target > 0).
    """
    if target <= 0:
        return {g: 0 for g in group_sizes}
    total = sum(group_sizes.values())
    if total <= 0:
        return {g: 0 for g in group_sizes}

    raw = {g: target * (sz / total) for g, sz in group_sizes.items()}
    quotas = {g: int(raw[g]) for g in group_sizes}
    remain = target - sum(quotas.values())
    if remain > 0:
        fracs = sorted(((raw[g] - quotas[g], g) for g in group_sizes), reverse=True)
        for _, g in fracs[:remain]:
            quotas[g] += 1

    # Fix any accidental overshoot due to rounding.
    while sum(quotas.values()) > target:
        g = max(quotas, key=lambda k: quotas[k])
        if quotas[g] <= 0:
            break
        quotas[g] -= 1
    return quotas


def dedup_eval_exact(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Exact de-dup within an eval split by normalized question.
    Returns (kept, dropped_records).
    """
    seen: Dict[str, str] = {}
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for r in rows:
        rid = str(r.get("id", ""))
        q = normalize_for_exact(get_question(r))
        if not q:
            kept.append(r)
            continue
        prev = seen.get(q)
        if prev is None:
            seen[q] = rid
            kept.append(r)
        else:
            dropped.append(
                {
                    "reason": "eval_exact_question_duplicate",
                    "id": rid,
                    "matched_id": prev,
                    "sample": r,
                }
            )
    return kept, dropped


def stratified_split_target(
    rows: List[Dict[str, Any]],
    val_target: int,
    test_target: int,
    seed: int,
    stratify_field: str,
    eval_oversample: float,
    do_eval_dedup_exact: bool,
    accept_eval_overflow: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Target-size split with optional oversampling and eval internal exact de-dup.

    Default behavior (accept_eval_overflow=False):
    - After (optional) eval de-dup, we shuffle and then trim val/test to the requested targets.
    - Any overflow samples are discarded (NOT moved back to train) to avoid eval construction
      influencing the training distribution.

    When accept_eval_overflow=True:
    - Keep all de-duplicated eval samples even if they exceed the requested targets.

    Returns (train, val, test, dropped_eval_records).
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[get_group(r, stratify_field)].append(r)

    group_sizes = {g: len(items) for g, items in groups.items()}
    val_quota = proportional_quotas(group_sizes, val_target)
    test_quota = proportional_quotas(group_sizes, test_target)

    os = max(1.0, float(eval_oversample))
    val_cand_quota = {g: int(round(val_quota[g] * os)) for g in groups}
    test_cand_quota = {g: int(round(test_quota[g] * os)) for g in groups}

    rng_global = random.Random(seed)
    train: List[Dict[str, Any]] = []
    val_cand: List[Dict[str, Any]] = []
    test_cand: List[Dict[str, Any]] = []

    for g, items in groups.items():
        g_seed = seed ^ stable_hash(g)
        rng = random.Random(g_seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)

        t_take = min(len(idx), max(0, test_cand_quota.get(g, 0)))
        t_idx = set(idx[:t_take])
        remain = [i for i in idx if i not in t_idx]

        v_take = min(len(remain), max(0, val_cand_quota.get(g, 0)))
        v_idx = set(remain[:v_take])

        for i, it in enumerate(items):
            if i in t_idx:
                test_cand.append(it)
            elif i in v_idx:
                val_cand.append(it)
            else:
                train.append(it)

    rng_global.shuffle(val_cand)
    rng_global.shuffle(test_cand)

    dropped_eval: List[Dict[str, Any]] = []
    if do_eval_dedup_exact:
        val_cand, dropped_v = dedup_eval_exact(val_cand)
        test_cand, dropped_t = dedup_eval_exact(test_cand)
        dropped_eval.extend(dropped_v)
        dropped_eval.extend(dropped_t)

    if accept_eval_overflow:
        val = val_cand
        test = test_cand
        # Note: do NOT move overflow back to train.
        rng_global.shuffle(train)
        rng_global.shuffle(val)
        rng_global.shuffle(test)
        return train, val, test, dropped_eval

    # Otherwise shuffle and trim to target. Discard overflow (do NOT move to train).
    rng_global.shuffle(val_cand)
    rng_global.shuffle(test_cand)
    val = val_cand[: max(0, val_target)] if val_target > 0 else []
    test = test_cand[: max(0, test_target)] if test_target > 0 else []

    rng_global.shuffle(train)
    rng_global.shuffle(val)
    rng_global.shuffle(test)
    return train, val, test, dropped_eval


def load_stop_ngrams(path: str) -> Set[str]:
    if not path:
        return set()
    try:
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        return set(map(str, j.get("stop_ngrams", [])))
    except Exception:
        raise SystemExit(f"Failed to load --stop_ngrams_json: {path}")


def build_minhash(text: str, ngram: int, num_perm: int, stop_ngrams: Set[str]):
    from datasketch import MinHash  # type: ignore

    mh = MinHash(num_perm=num_perm)
    for gram in char_ngrams(text, ngram):
        if stop_ngrams and gram in stop_ngrams:
            continue
        mh.update(gram.encode("utf-8", errors="ignore"))
    return mh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.005, help="Ratio mode.")
    ap.add_argument("--test_ratio", type=float, default=0.005, help="Ratio mode.")
    ap.add_argument("--val_target", type=int, default=0, help="Target mode: desired val size (0 disables).")
    ap.add_argument("--test_target", type=int, default=0, help="Target mode: desired test size (0 disables).")
    ap.add_argument("--eval_oversample", type=float, default=1.4, help="Target mode: oversample multiplier for val/test candidates.")
    ap.add_argument("--dedup_eval_exact", action="store_true", help="Target mode: exact de-dup within val/test by question.")
    ap.add_argument(
        "--accept_eval_overflow",
        action="store_true",
        help="Target mode: if de-dup yields more than target, keep the larger eval set (do NOT trim).",
    )
    ap.add_argument("--stratify_field", default="meta.top_department")

    ap.add_argument("--deleak_exact", action="store_true", help="Remove train items with exact normalized-question match to val/test.")
    ap.add_argument("--deleak_minhash", action="store_true", help="Remove train items near-duplicate to val/test via MinHash+LSH.")
    ap.add_argument("--stop_ngrams_json", default="", help="Optional JSON with stop_ngrams for MinHash.")
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--num_perm", type=int, default=64)
    ap.add_argument("--lsh_threshold", type=float, default=0.9)
    ap.add_argument(
        "--verify_jaccard",
        action="store_true",
        help="Verify Jaccard for LSH candidates (uses more RAM; stores LeanMinHash for train).",
    )

    ap.add_argument("--max_docs", type=int, default=-1, help="Debug: stop after N docs from input.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    dropped_eval_path = out_dir / "dropped_from_eval.jsonl"
    dropped_train_path = out_dir / "dropped_from_train.jsonl"
    summary_path = out_dir / "summary.json"

    total_lines = maybe_total_lines(args.input_jsonl)
    rows: List[Dict[str, Any]] = []
    bad_schema = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        it = tqdm(fin, total=total_lines, desc="Load pool", unit="lines")
        for i, line in enumerate(it, start=1):
            if args.max_docs > 0 and len(rows) >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "conversations" not in obj:
                    raise ValueError("missing conversations")
                rows.append(obj)
            except Exception:
                bad_schema += 1

    dropped_eval: List[Dict[str, Any]] = []
    if args.val_target > 0 or args.test_target > 0:
        train, val, test, dropped_eval = stratified_split_target(
            rows,
            val_target=max(0, int(args.val_target)),
            test_target=max(0, int(args.test_target)),
            seed=args.seed,
            stratify_field=args.stratify_field,
            eval_oversample=args.eval_oversample,
            do_eval_dedup_exact=bool(args.dedup_eval_exact),
            accept_eval_overflow=bool(args.accept_eval_overflow),
        )
    else:
        train, val, test = stratified_split(
            rows, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed, stratify_field=args.stratify_field
        )

    eval_rows = [("val", r) for r in val] + [("test", r) for r in test]

    # Build eval lookup for exact deleak.
    eval_exact: Dict[str, Tuple[str, str]] = {}  # norm_q -> (split, eval_id)
    if args.deleak_exact:
        for split, r in eval_rows:
            q = normalize_for_exact(get_question(r))
            if not q:
                continue
            eval_exact[q] = (split, str(r.get("id", "")))

    # MinHash deleak index over train.
    stop_ngrams = load_stop_ngrams(args.stop_ngrams_json) if args.deleak_minhash else set()
    lsh = None
    train_sig = None
    if args.deleak_minhash:
        try:
            from datasketch import LeanMinHash, MinHashLSH  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Missing dependency `datasketch`. Activate conda env (e.g. `conda activate med`) and `pip install datasketch`."
            ) from e
        lsh = MinHashLSH(threshold=args.lsh_threshold, num_perm=args.num_perm)
        train_sig = {} if args.verify_jaccard else None
        it = tqdm(train, desc="Index train (LSH)", unit="docs")
        for r in it:
            tid = str(r.get("id", ""))
            qn = normalize_for_minhash(get_question(r))
            if not qn:
                continue
            mh = build_minhash(qn, ngram=args.ngram, num_perm=args.num_perm, stop_ngrams=stop_ngrams)
            if args.verify_jaccard:
                train_sig[tid] = LeanMinHash(mh)  # type: ignore[index]
                lsh.insert(tid, train_sig[tid])  # type: ignore[arg-type]
            else:
                lsh.insert(tid, mh)  # type: ignore[arg-type]

    # Decide which train items to drop.
    drop_train_ids: Dict[str, Dict[str, Any]] = {}
    exact_hits = 0
    lsh_hits = 0

    if args.deleak_exact or args.deleak_minhash:
        it = tqdm(eval_rows, desc="De-leak (query eval)", unit="docs")
        for split, r in it:
            eval_id = str(r.get("id", ""))
            q_raw = get_question(r)

            if args.deleak_exact:
                q_norm = normalize_for_exact(q_raw)
                # exact lookup happens when scanning train below (cheaper than building huge map of train->norm)
                # so here we just ensure eval_exact is built.
                _ = q_norm

            if args.deleak_minhash and lsh is not None:
                qn = normalize_for_minhash(q_raw)
                if not qn:
                    continue
                mhq = build_minhash(qn, ngram=args.ngram, num_perm=args.num_perm, stop_ngrams=stop_ngrams)
                cands = lsh.query(mhq)
                if not cands:
                    continue
                if args.verify_jaccard and train_sig is not None:
                    # verify Jaccard against stored LeanMinHash
                    for tid in cands:
                        other = train_sig.get(tid)
                        if other is None:
                            continue
                        sim = mhq.jaccard(other)
                        if sim >= args.lsh_threshold:
                            if tid not in drop_train_ids:
                                drop_train_ids[tid] = {
                                    "reason": "minhash_near_duplicate",
                                    "matched_eval_split": split,
                                    "matched_eval_id": eval_id,
                                    "estimated_jaccard": float(sim),
                                }
                            lsh_hits += 1
                else:
                    # conservative: drop all candidates returned by LSH
                    for tid in cands:
                        if tid not in drop_train_ids:
                            drop_train_ids[tid] = {
                                "reason": "minhash_lsh_candidate",
                                "matched_eval_split": split,
                                "matched_eval_id": eval_id,
                            }
                        lsh_hits += 1

    # Stream out train/val/test, performing exact deleak while writing train.
    kept_train = 0
    dropped_train = 0
    deleak_reasons = Counter()
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_dept = {"train": Counter(), "val": Counter(), "test": Counter()}

    def dept(r: Dict[str, Any]) -> str:
        meta = r.get("meta") or {}
        return str(meta.get("top_department", "") or "")

    # Write eval-internal dropped items (if any) for auditing.
    if dropped_eval:
        with dropped_eval_path.open("w", encoding="utf-8") as fev:
            for d in dropped_eval:
                fev.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        dropped_eval_path.write_text("", encoding="utf-8")

    with train_path.open("w", encoding="utf-8") as ftr, val_path.open("w", encoding="utf-8") as fva, test_path.open(
        "w", encoding="utf-8"
    ) as fte, dropped_train_path.open("w", encoding="utf-8") as fdrop:
        for r in val:
            fva.write(json.dumps(r, ensure_ascii=False) + "\n")
            split_counts["val"] += 1
            split_dept["val"][dept(r)] += 1
        for r in test:
            fte.write(json.dumps(r, ensure_ascii=False) + "\n")
            split_counts["test"] += 1
            split_dept["test"][dept(r)] += 1

        for r in tqdm(train, desc="Write train (deleak)", unit="docs"):
            tid = str(r.get("id", ""))
            drop_info = drop_train_ids.get(tid)

            # exact deleak: drop if exact normalized question matches eval.
            if drop_info is None and args.deleak_exact:
                q = normalize_for_exact(get_question(r))
                hit = eval_exact.get(q)
                if hit is not None and q:
                    drop_info = {
                        "reason": "exact_question_match",
                        "matched_eval_split": hit[0],
                        "matched_eval_id": hit[1],
                    }
                    exact_hits += 1

            if drop_info is not None:
                dropped_train += 1
                deleak_reasons[drop_info["reason"]] += 1
                fdrop.write(json.dumps({"id": tid, **drop_info, "sample": r}, ensure_ascii=False) + "\n")
                continue

            ftr.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept_train += 1
            split_counts["train"] += 1
            split_dept["train"][dept(r)] += 1

    summary = {
        "input_jsonl": args.input_jsonl,
        "bad_schema_lines": bad_schema,
        "split_params": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "val_target": args.val_target,
            "test_target": args.test_target,
            "eval_oversample": args.eval_oversample,
            "dedup_eval_exact": bool(args.dedup_eval_exact),
            "accept_eval_overflow": bool(args.accept_eval_overflow),
            "stratify_field": args.stratify_field,
        },
        "deleak_params": {
            "deleak_exact": bool(args.deleak_exact),
            "deleak_minhash": bool(args.deleak_minhash),
            "stop_ngrams_json": args.stop_ngrams_json,
            "stop_ngrams_count": len(stop_ngrams),
            "ngram": args.ngram,
            "num_perm": args.num_perm,
            "lsh_threshold": args.lsh_threshold,
            "verify_jaccard": bool(args.verify_jaccard),
        },
        "counts": {
            "pool_rows": len(rows),
            "train_before_deleak": len(train),
            "train_kept": kept_train,
            "train_dropped": dropped_train,
            "val": len(val),
            "test": len(test),
            "eval_dropped_internal": len(dropped_eval),
            "exact_hits": exact_hits,
            "lsh_hits": lsh_hits,
        },
        "deleak_reasons": dict(deleak_reasons),
        "per_split_top_department": {
            "train": dict(split_dept["train"]),
            "val": dict(split_dept["val"]),
            "test": dict(split_dept["test"]),
        },
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "dropped_from_eval": str(dropped_eval_path),
            "dropped_from_train": str(dropped_train_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "train_kept": kept_train, "train_dropped": dropped_train}, ensure_ascii=False))


if __name__ == "__main__":
    main()
