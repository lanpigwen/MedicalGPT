import json
import argparse

        
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--key_to_keep", required=True)
    args = ap.parse_args()

    # 把jsonl的数据中的每一行json都只保留其中的一个字段，写入新的jsonl文件中
    with open(args.input_jsonl, 'r', encoding='utf-8') as infile, open(args.output_jsonl, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # 假设我们只保留字段 'text'
            new_data = {args.key_to_keep: data[args.key_to_keep]}
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            
if __name__ == "__main__":
    main()