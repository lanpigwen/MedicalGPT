import json

input_file = 'data/dxw/step08_preference_grpo/path2Afterpath1.jsonl'
output_file = 'data/dxw/step08_preference_grpo/path2Afterpath1_cleaned.jsonl'

# 读取原始数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

new_data = []
for item in data:
    rejected = item['rejected']

    
    # 清洗：如果包含 "rejected\n" 字样，只保留后面的内容
    if "rejected\n" in rejected:
        rejected = rejected.split("rejected\n")[-1].strip()
        
    if "rejected：" in rejected:
        rejected = rejected.split("rejected：")[-1].strip()    
    if "rejected" in rejected:
        rejected = rejected.split("rejected")[-1].strip()
        
    if "rejected" in rejected:
        print(item['prompt'])
        
    if len(str(rejected).strip()) == 0:
        print(f"Warning: Found empty rejected text. Skipping item with prompt: {item['prompt']}")
        continue        
    
    new_item = {
        'question': item['prompt'],
        'chosen': item['chosen'],
        'rejected': rejected
    }
    if new_item['rejected'] != new_item['chosen']:  # 确保 rejected 和 chosen 不完全相同
        new_data.append(new_item)

# 保存（修复 ensure_ascii=False，中文不乱码）
with open(output_file, 'w', encoding='utf-8') as f:
    for item in new_data:
        # 把 ensure_ascii=False 放在 json.dumps 里
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        