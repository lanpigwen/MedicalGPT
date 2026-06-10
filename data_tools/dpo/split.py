import json
input_file = 'data/dxw/step08_preference_grpo/path2Afterpath1_cleaned_sharegpt.jsonl'
# 把这个分成 train/val/test 三份，比例为 8:1:1

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]
import random
random.shuffle(data)
total = len(data)
test_data = data[:500]
val_data = data[500:1000]
train_data = data[1000:]
with open('/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/dpo/test/test_2_+_1.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/dpo/val/val_2_+_1.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/dpo/train/train_2_+_1.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')