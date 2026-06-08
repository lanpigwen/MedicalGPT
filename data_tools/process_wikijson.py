import json

# 输入 JSON 路径
input_path = "/home/apulis-dev/userdata/2026/MedicalGPT/data/medical/pretrain/wikipedia-cn-20230720-filtered.json"

# 输出 JSONL 路径（会覆盖旧文件，不会重复）
output_path = "wikipedia-cn-20230720-filtered.jsonl"

# 读取大 JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 一次性写入 JSONL（更快、更干净）
with open(output_path, "w", encoding="utf-8") as f:
    for item in data:
        # 只保留 text 字段，适配预训练格式
        output_item = {"text": item["completion"]}
        # 转成一行 JSON
        f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

print(f"✅ 转换完成！共 {len(data)} 条 → {output_path}")