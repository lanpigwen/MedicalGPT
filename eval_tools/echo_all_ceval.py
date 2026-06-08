import os
import json
import glob
from tabulate import tabulate
from collections import defaultdict

# ======================
# 配置
# ======================
RESULT_DIR = "./eval_results"

TASKS = [
    "ceval-valid_basic_medicine",
    "ceval-valid_clinical_medicine",
    "ceval-valid_physician",
    "ceval-valid_veterinary_medicine"
]

# ======================
# 批量计算（按 fewshot 分组）
# ======================
print("=" * 120)
print(f"📊 批量计算 {len(TASKS)} 个医学 C-Eval 任务平均分（按 FewShot 分组）".center(118))
print("=" * 120)

# key: fewshot类型(0shot/5shot...)，value: 模型列表
grouped_models = defaultdict(list)

for json_file in glob.glob(os.path.join(RESULT_DIR, "*.json")):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = []
        for task in TASKS:
            acc = data["results"][task]["acc_norm,none"]
            scores.append(acc)

        avg = sum(scores) / len(scores)
        
        fn = os.path.basename(json_file).replace(".json", "").split("_")
        model_name = fn[0]
        fewshot = fn[1]  # 分组依据

        # 按 fewshot 分组存入
        grouped_models[fewshot].append((model_name, avg, scores))

    except Exception as e:
        print(f"⚠️ 跳过 {json_file}：{e}")

# ======================
# 遍历每个分组，打印表格
# ======================
headers = [
    "模型名称",
    "医学平均分",
    "Basic",
    "Clinical",
    "Physician",
    "Veterinary"
]

total_model_count = 0

# 按 shot 数字排序输出（0shot → 5shot → 10shot...）
for fewshot in sorted(grouped_models.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)))):
    models = grouped_models[fewshot]
    total_model_count += len(models)
    
    # 🔴 核心：每组 按医学平均分 从高到低排序
    models.sort(key=lambda x: x[1], reverse=True)

    # 构造表格
    table_data = []
    for model_name, avg, scores in models:
        table_data.append([
            model_name,
            f"{avg:.4f}",
            f"{scores[0]:.3f}",
            f"{scores[1]:.3f}",
            f"{scores[2]:.3f}",
            f"{scores[3]:.3f}",
        ])

    # 打印当前分组
    print(f"\n🔹 {fewshot} 结果（共 {len(models)} 个模型）".center(120))
    print("-" * 120)
    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            stralign="left",
            numalign="center"
        )
    )

# ======================
# 最终统计
# ======================
print("\n" + "=" * 120)
print(f"✅ 全部完成！共统计 {len(grouped_models)} 个分组，总计 {total_model_count} 个模型".center(118))
print("=" * 120)