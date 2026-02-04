import json
from pathlib import Path
from statistics import mean

MODEL_KEYS = [
    "llama3_2_1b",
    "gemma2_2b",
    "llama3_2_3b",
    "phi3_mini",
    "mistral_7b",
    "llama3_1_8b",
]

PRETTY_NAMES = {
    "llama3_2_1b": "Llama 3.2 1B",
    "gemma2_2b": "Gemma 2 2B",
    "llama3_2_3b": "Llama 3.2 3B",
    "phi3_mini": "Phi-3 Mini 3.8B",
    "mistral_7b": "Mistral 7B",
    "llama3_1_8b": "Llama 3.1 8B",
}

RAG_DIR = Path("metaqa_rag_results")
BASELINE_DIR = Path("metaqa_baseline_results")


def rag_path(model: str) -> Path:
    return RAG_DIR / f"metaqa_{model}.jsonl"


def baseline_path(model: str) -> Path:
    return BASELINE_DIR / f"metaqa_{model}.jsonl"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_file(path: Path):
    n = 0
    hits1_count = 0
    em_count = 0
    f1_values = []

    for sample in load_jsonl(path):
        n += 1

        if sample.get("top1_match"):
            hits1_count += 1

        if sample.get("exact_match"):
            em_count += 1

        f1 = sample.get("f1", None)
        if isinstance(f1, (int, float)):
            f1_values.append(float(f1))

    if n == 0:
        return {"n": 0, "hits1": 0.0, "macro_f1": 0.0, "em": 0.0}

    return {
        "n": n,
        "hits1": hits1_count / n,
        "macro_f1": mean(f1_values) if f1_values else 0.0,
        "em": em_count / n,
    }


def baseline_latex_table(metrics_dict):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline performance on MetaQA without RAG.}",
        r"\label{tab:baseline_metaqa_full}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Hits@1 (\%) & Macro F1 (\%) & Exact Match (\%) \\",
        r"\midrule",
    ]

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        macro_f1 = 100 * stats["macro_f1"]
        em = 100 * stats["em"]
        lines.append(f"{name} & {hits1:.2f} & {macro_f1:.2f} & {em:.2f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def rag_latex_table(metrics_dict):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RAG performance on MetaQA.}",
        r"\label{tab:rag_metaqa_full}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Hits@1 (\%) & Macro F1 (\%) & Exact Match (\%) \\",
        r"\midrule",
    ]

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        macro_f1 = 100 * stats["macro_f1"]
        em = 100 * stats["em"]
        lines.append(f"{name} & {hits1:.2f} & {macro_f1:.2f} & {em:.2f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    baseline_metrics = {}
    rag_metrics = {}

    for key in MODEL_KEYS:
        base_file = baseline_path(key)
        if base_file.exists():
            baseline_metrics[key] = aggregate_file(base_file)
        else:
            print(f"[WARN] Baseline file not found for {key}: {base_file}")
            baseline_metrics[key] = {"n": 0, "hits1": 0.0, "macro_f1": 0.0, "em": 0.0}

        rag_file = rag_path(key)
        if rag_file.exists():
            rag_metrics[key] = aggregate_file(rag_file)
        else:
            print(f"[WARN] RAG file not found for {key}: {rag_file}")
            rag_metrics[key] = {"n": 0, "hits1": 0.0, "macro_f1": 0.0, "em": 0.0}

    print("\n" + "=" * 70)
    print("Baseline (no-RAG) LaTeX table")
    print("=" * 70)
    print(baseline_latex_table(baseline_metrics))

    print("\n" + "=" * 70)
    print("RAG LaTeX table")
    print("=" * 70)
    print(rag_latex_table(rag_metrics))


if __name__ == "__main__":
    main()
