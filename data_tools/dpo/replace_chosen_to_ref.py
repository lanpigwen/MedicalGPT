import json

path_ref = "data/dxw/step07_deleak_train_grpo/train_10.jsonl"




with open(path_ref, "r") as f:
    data_ref = [json.loads(line) for line in f]
    
data_ref_dict = {item["conversations"][0]["value"]: item["conversations"][1]["value"] for item in data_ref}


def replace_chosen_with_ref(path_chosen, data_ref_dict):
    with open(path_chosen, "r") as f:
        data_chosen = [json.loads(line) for line in f]
        
    new_data = []
    for item in data_chosen:
        question = item["conversations"][0]["value"]
        if question in data_ref_dict:
            item["chosen"] = data_ref_dict[question]
        else:
            print(f"Question not found in reference data: {question}")
        new_data.append(item)

    # 中文正常显示
    output_path = path_chosen.replace(".jsonl", "_ref.jsonl")
    with open(output_path, "w") as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        

# print(data_ref_dict)

path_chosen = "data/dxw/dpo/test/test_2_+_1.jsonl"
replace_chosen_with_ref(path_chosen, data_ref_dict)
path_chosen = "data/dxw/dpo/train/train_2_+_1.jsonl"
replace_chosen_with_ref(path_chosen, data_ref_dict)
path_chosen = "data/dxw/dpo/val/val_2_+_1.jsonl"
replace_chosen_with_ref(path_chosen, data_ref_dict)