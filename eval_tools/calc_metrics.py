import json
import argparse

import jieba
import sacrebleu
from tqdm import tqdm
from rouge_chinese import Rouge
from prettytable import PrettyTable


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tokenize_zh(text):
    return " ".join(jieba.cut(text))


def calc_rouge(preds, refs):
    rouge = Rouge()

    scores_1 = []
    scores_2 = []
    scores_l = []

    for pred, ref in tqdm(zip(preds, refs), total=len(preds), desc="ROUGE"):
        if not pred.strip() or not ref.strip():
            continue

        pred_tok = tokenize_zh(pred)
        ref_tok = tokenize_zh(ref)

        try:
            r = rouge.get_scores(pred_tok, ref_tok)[0]
            scores_1.append(r["rouge-1"]["f"])
            scores_2.append(r["rouge-2"]["f"])
            scores_l.append(r["rouge-l"]["f"])
        except Exception:
            continue

    return {
        "rouge-1": sum(scores_1) / len(scores_1) if scores_1 else 0.0,
        "rouge-2": sum(scores_2) / len(scores_2) if scores_2 else 0.0,
        "rouge-l": sum(scores_l) / len(scores_l) if scores_l else 0.0,
    }


def calc_bleu_n(preds, refs, n):
    preds_tok = [tokenize_zh(p) for p in preds]
    refs_tok = [tokenize_zh(r) for r in refs]

    bleu_metric = sacrebleu.metrics.BLEU(
        tokenize="none",
        max_ngram_order=n,
        effective_order=True
    )

    bleu = bleu_metric.corpus_score(
        preds_tok,
        [refs_tok]
    )

    return bleu.score / 100.0


def calc_bleu_all(preds, refs):
    return {
        "bleu-1": calc_bleu_n(preds, refs, 1),
        "bleu-2": calc_bleu_n(preds, refs, 2),
        "bleu-3": calc_bleu_n(preds, refs, 3),
        "bleu-4": calc_bleu_n(preds, refs, 4),
    }


def calc_length(preds):
    lengths = [len(x) for x in preds]
    return sum(lengths) / len(lengths) if lengths else 0.0


def calc_repeat_rate(text, n=4):
    tokens = list(jieba.cut(text))

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    total = len(ngrams)
    unique = len(set(ngrams))

    return 1 - unique / total if total > 0 else 0.0


def calc_avg_repeat_rate(preds, n=4):
    rates = [calc_repeat_rate(x, n=n) for x in preds]
    return sum(rates) / len(rates) if rates else 0.0


def calc_bertscore(preds, refs):
    from bert_score import score

    P, R, F1 = score(
        preds,
        refs,
        lang="zh",
        verbose=True,
        rescale_with_baseline=False
    )

    return {
        "bertscore_p": P.mean().item(),
        "bertscore_r": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def evaluate_model(name, preds, refs, use_bertscore=False):
    print(f"\nEvaluating {name}...")

    rouge_scores = calc_rouge(preds, refs)
    bleu_scores = calc_bleu_all(preds, refs)

    result = {
        "model": name,
        "rouge-1": rouge_scores["rouge-1"],
        "rouge-2": rouge_scores["rouge-2"],
        "rouge-l": rouge_scores["rouge-l"],
        "bleu-1": bleu_scores["bleu-1"],
        "bleu-2": bleu_scores["bleu-2"],
        "bleu-3": bleu_scores["bleu-3"],
        "bleu-4": bleu_scores["bleu-4"],
        "avg_len": calc_length(preds),
        "repeat_rate_4gram": calc_avg_repeat_rate(preds, n=4),
    }

    if use_bertscore:
        result.update(calc_bertscore(preds, refs))

    return result


def collect_predictions(data):
    refs = []
    model_preds = {}

    # 先收集所有模型名，避免某些样本缺失导致长度错位
    model_names = ["base"]
    for item in data:
        for name in item.get("sft_results", {}).keys():
            if name not in model_names:
                model_names.append(name)

    for name in model_names:
        model_preds[name] = []

    for item in data:
        ref = item.get("reference", "").strip()
        if not ref:
            continue

        refs.append(ref)

        model_preds["base"].append(
            item.get("base_answer", "").strip()
        )

        sft_results = item.get("sft_results", {})

        for name in model_names:
            if name == "base":
                continue
            model_preds[name].append(
                sft_results.get(name, "").strip()
            )

    return refs, model_preds


def print_results(results):
    table = PrettyTable()

    table.field_names = [
        "Model",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
        "Avg Len",
        "Repeat-4gram",
        "BERTScore-P",
        "BERTScore-R",
        "BERTScore-F1",
    ]

    for r in results:
        table.add_row([
            r["model"],
            f"{r['rouge-1']:.4f}",
            f"{r['rouge-2']:.4f}",
            f"{r['rouge-l']:.4f}",
            f"{r['bleu-1']:.4f}",
            f"{r['bleu-2']:.4f}",
            f"{r['bleu-3']:.4f}",
            f"{r['bleu-4']:.4f}",
            f"{r['avg_len']:.1f}",
            f"{r['repeat_rate_4gram']:.4f}",
            f"{r.get('bertscore_p', 0):.4f}",
            f"{r.get('bertscore_r', 0):.4f}",
            f"{r.get('bertscore_f1', 0):.4f}",
        ])

    print("\n" + str(table))
    
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--use_bertscore", action="store_true")
    args = parser.parse_args()

    data = load_jsonl(args.input_file)

    refs, model_preds = collect_predictions(data)

    print(f"Loaded {len(refs)} valid samples.")
    print("Models found:")
    for name in model_preds:
        print(f"  {name}")

    results = []

    for model_name, preds in model_preds.items():
        results.append(
            evaluate_model(
                model_name,
                preds,
                refs,
                use_bertscore=args.use_bertscore
            )
        )
    print(args.input_file)
    table = print_results(results)
    # 再把结果保存到txt文件，方便后续分析
    output_path = args.input_file.replace(".jsonl", "_metrics.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(table))


if __name__ == "__main__":
    main()