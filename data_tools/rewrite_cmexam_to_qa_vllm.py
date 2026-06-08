import re
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_options(options):
    return "\n".join([f"{x['key']}. {x['value']}" for x in options])


def get_answer_text(item):
    ans = item["Answer"]
    for opt in item["Options"]:
        if opt["key"] == ans:
            return opt["value"]
    return ""


def build_prompt(item):
    question = item["Question"]
    options = format_options(item["Options"])
    answer = item["Answer"]
    answer_text = get_answer_text(item)
    explanation = item.get("Explanation", "")

    return f"""请将下面的中文医学选择题改写成普通医学知识问答，用于训练医学对话模型。

改写目标：
把“选择题”改成“自然的医学知识问答”，让问题看起来像真实用户在询问医学知识，而不是考试题。

严格要求：
1. 不要保留选项形式。
2. 不要输出 A/B/C/D/E。
3. 不要出现“下列”“以下选项”“哪一项”“最符合”“正确的是”“错误的是”“应选择”“答案是”等选择题式表达。
4. 问题必须改写成开放式医学知识问题，例如“……常见于什么情况？”、“……的原因是什么？”、“……应如何判断？”。
5. 答案要直接给出正确医学知识，并包含简洁解释。
6. 不要输出思考过程。
7. 只输出下面两行格式：

【问题】改写后的医学知识问题
【答案】直接回答正确知识，可以包含简洁解释

原题：
{question}

选项：
{options}

正确答案：
{answer}. {answer_text}

解析：
{explanation}
/no_think
"""


def build_chat_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def clean_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def parse_output(text):
    text = clean_think(text)

    q_match = re.search(r"【问题】\s*(.*?)(?=【答案】)", text, flags=re.S)
    a_match = re.search(r"【答案】\s*(.*)", text, flags=re.S)

    if q_match and a_match:
        human = q_match.group(1).strip()
        assistant = a_match.group(1).strip()
        if human and assistant:
            return human, assistant

    return None, None


def fallback_convert(item):
    question = item["Question"]
    answer_text = get_answer_text(item)
    explanation = item.get("Explanation", "")

    human = question
    assistant = f"{answer_text}。{explanation}".strip()

    return human, assistant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    data = load_jsonl(args.input_file)
    if args.limit:
        data = data[:args.limit]

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    tokenizer = llm.get_tokenizer()

    prompts = [
        build_chat_prompt(tokenizer, build_prompt(item))
        for item in data
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.05,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    failed = 0

    for item, out in tqdm(zip(data, outputs), total=len(data)):
        raw = out.outputs[0].text.strip()
        human, assistant = parse_output(raw)

        if human is None or assistant is None:
            failed += 1
            continue
            human, assistant = fallback_convert(item)

        results.append({
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": assistant}
            ],
            "source": "CMExam",
            "raw_question": item.get("Question", ""),
            "raw_answer": item.get("Answer", ""),
            "raw_generation": raw
        })

    save_jsonl(results, args.output_file)

    print(f"Saved to {args.output_file}")
    print(f"Failed parse: {failed}")


if __name__ == "__main__":
    main()