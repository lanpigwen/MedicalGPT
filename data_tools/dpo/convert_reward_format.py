import json
import argparse
from pathlib import Path


def convert_to_reward_format(example):
    """
    Convert:
    {
        "question": "...",
        "response_chosen": "...",
        "response_rejected": "..."
    }

    To:
    {
        "conversations": [
            {"from": "human", "value": "..."}
        ],
        "chosen": "...",
        "rejected": "..."
    }
    """

    question = str(example.get("question", "")).strip()
    chosen = example.get("response_chosen", "") or example.get("chosen", "")
    rejected = example.get("response_rejected", "") or example.get("rejected", "")

    new_example = {
        "conversations": [
            {
                "from": "human",
                "value": question
            }
        ],
        "chosen": chosen,
        "rejected": rejected
    }

    if "tools" in example:
        new_example["tools"] = example["tools"]

    return new_example


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw preference jsonl data to reward model format."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonl file."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output jsonl file."
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_id, line in enumerate(fin, start=1):
            line = line.strip()

            if not line:
                skipped += 1
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[Warning] Skip line {line_id}: JSON decode error: {e}")
                continue



            new_example = convert_to_reward_format(example)

            fout.write(
                json.dumps(new_example, ensure_ascii=False) + "\n"
            )

            total += 1

    print(f"Done.")
    print(f"Converted: {total}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()