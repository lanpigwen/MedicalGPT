#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a tiny jsonl sample from step01 output for fast pipeline smoke tests.
This is optional; does not affect the main dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_lines", type=int, default=2000)
    args = ap.parse_args()

    inp = Path(args.input_jsonl)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in itertools.islice(fi, args.max_lines):
            if not line.strip():
                continue
            # ensure valid json
            obj = json.loads(line)
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"written": n, "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

