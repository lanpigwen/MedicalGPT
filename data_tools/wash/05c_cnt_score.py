import json
import pandas as pd

rows = []
jsonl_path = "/home/apulis-dev/userdata/2026/MedicalGPT/data/dxw/step05a_qa_quality_score/result.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        rows.append({
            "top_department":
                item.get("meta", {}).get("top_department", "Unknown"),
            "score":
                item.get("quality_score", -1)
        })

df = pd.DataFrame(rows)

pivot = pd.crosstab(
    df["top_department"],
    df["score"],
    margins=True
)

print(pivot)