import json
import random
path_ref = "data/dxw/step08_preference_grpo/path2Afterpath1_cleaned_sharegpt.jsonl"




with open(path_ref, "r") as f:
    data_ref = [json.loads(line) for line in f]
    
# path2的chosen
data_ref_dict = {item["conversations"][0]["value"]: item["chosen"] for item in data_ref}

# 把reject换成一部分是path2的chosen，一部分是原来的reject
def replace_reject_with_ref(path_reject, data_ref_dict):
    with open(path_reject, "r") as f:
        data_reject = [json.loads(line) for line in f]
        
    new_data = []
    for item in data_reject:
        question = item["conversations"][0]["value"]
        if question in data_ref_dict:
            random_num = random.random()
            if random_num < 0.5:
                item["rejected"] = data_ref_dict[question]
        else:
            print(f"Question not found in reference data: {question}")
        new_data.append(item)

    # 中文正常显示
    output_path = path_reject.replace(".jsonl", "_0.5rejected_is_path2_chosen.jsonl")
    with open(output_path, "w") as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        

# print(data_ref_dict)

path_reject = "data/dxw/dpo/test/test_2_+_1_ref.jsonl"
replace_reject_with_ref(path_reject, data_ref_dict)
path_reject = "data/dxw/dpo/train/train_2_+_1_ref.jsonl"
replace_reject_with_ref(path_reject, data_ref_dict)
path_reject = "data/dxw/dpo/val/val_2_+_1_ref.jsonl"
replace_reject_with_ref(path_reject, data_ref_dict)