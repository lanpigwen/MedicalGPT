import json
import random

input_file1 = '/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/step08_preference_grpo/path1_ref_chosen_cleaned.jsonl'
input_file2 = '/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/step08_preference_grpo/path2_ref_chosen_cleaned.jsonl'
output_file = '/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/step08_preference_grpo/merged_chosen_reject_95_ref_as_chosen.jsonl'

random.seed(42)

def get_question(item):
    return item.get("question") or item.get("prompt") or ""

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

data1 = load_jsonl(input_file1)
data2 = load_jsonl(input_file2)

data1_dict = {get_question(item): item for item in data1 if get_question(item)}
data2_dict = {get_question(item): item for item in data2 if get_question(item)}

all_questions = sorted(set(data1_dict.keys()) | set(data2_dict.keys()))

merged_data = []
path1_count = 0
path2_count = 0

for q in all_questions:
    item1 = data1_dict.get(q)
    item2 = data2_dict.get(q)
    
    rand_v = random.random()

    if item1 is not None and item2 is not None:
        # 随机选 path1 或 path2
        
        if rand_v < 0.6:
            item = item1
            source = "path1_ref_chosen"
            path1_count += 1
        else:
            item = item2
            source = "path2_pass5_ref_rank"
            path2_count += 1

    elif item1 is not None:
        item = item1
        source = "path1_ref_chosen"
        path1_count += 1

    elif item2 is not None:
        item = item2
        source = "path2_pass5_ref_rank"
        path2_count += 1

    else:
        continue

    item = item.copy()
    item["source"] = source
    new_item = {
        "question": item.get("question") or item.get("prompt") or "",
        "response_chosen": item.get("chosen", ""),
        "response_rejected": item.get("rejected", ""),
        # "source": source
    }
    
    if rand_v >= 0.6:
        # chosen 来自 path2, 按照0.5 可能 换回ref,也即path1的chosen
        new_item["response_chosen"] = item1.get("chosen", "") if item1 is not None and random.random() < 0.95 else new_item["response_chosen"]

    merged_data.append(new_item)

random.shuffle(merged_data)

with open(output_file, "w", encoding="utf-8") as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved to {output_file}")
print(f"Total: {len(merged_data)}")
print(f"path1: {path1_count}")
print(f"path2: {path2_count}")