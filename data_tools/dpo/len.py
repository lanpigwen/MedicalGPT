#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np


def get_text_len(text):
    return len(str(text).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="DPO jsonl file"
    )
    args = parser.parse_args()

    chosen_lens = []
    rejected_lens = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")

                chosen_lens.append(get_text_len(chosen))
                rejected_lens.append(get_text_len(rejected))
                if get_text_len(chosen) == 0 or get_text_len(rejected) == 0:
                    print(f"Warning: Found empty chosen or rejected text. Skipping line.")

            except Exception as e:
                print(f"Skip bad line: {e}")

    if len(chosen_lens) == 0:
        print("No valid samples found.")
        return

    chosen_lens = np.array(chosen_lens)
    rejected_lens = np.array(rejected_lens)

    ratios = rejected_lens / np.maximum(chosen_lens, 1)
    diffs = rejected_lens - chosen_lens

    print("=" * 60)
    print(f"Samples: {len(chosen_lens)}")
    print("=" * 60)

    print("\n[Chosen]")
    print(f"Mean     : {chosen_lens.mean():.2f}")
    print(f"Median   : {np.median(chosen_lens):.2f}")
    print(f"Min      : {chosen_lens.min()}")
    print(f"Max      : {chosen_lens.max()}")
    print(f"P95      : {np.percentile(chosen_lens,95):.2f}")

    print("\n[Rejected]")
    print(f"Mean     : {rejected_lens.mean():.2f}")
    print(f"Median   : {np.median(rejected_lens):.2f}")
    print(f"Min      : {rejected_lens.min()}")
    print(f"Max      : {rejected_lens.max()}")
    print(f"P95      : {np.percentile(rejected_lens,95):.2f}")

    print("\n[Rejected / Chosen]")
    print(f"Mean Ratio   : {ratios.mean():.3f}")
    print(f"Median Ratio : {np.median(ratios):.3f}")
    print(f"P5 Ratio     : {np.percentile(ratios,5):.3f}")
    print(f"P95 Ratio    : {np.percentile(ratios,95):.3f}")

    print("\n[Length Difference]")
    print(f"Mean Diff    : {diffs.mean():.2f}")
    print(f"Median Diff  : {np.median(diffs):.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()