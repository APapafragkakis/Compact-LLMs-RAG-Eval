# tools/postprocess_metaqa_metrics.py
#
# Offline υπολογισμός metrics για MetaQA (baseline + RAG)
# ΔΕΝ ξανατρέχουμε κανένα LLM: απλώς διαβάζουμε τα .jsonl ανά δείγμα
# και βγάζουμε:
#   - Hits@1 (από το top1_match field)
#   - Macro F1 (mean token-level F1)
#   - Exact Match (EM)
# και τυπώνουμε δύο LaTeX tables (baseline + RAG).

import json
from pathlib import Path
from statistics import mean

# =====================================================================
# CONFIG
# =====================================================================

# Τα model keys όπως εμφανίζονται στα filenames:
#   metaqa_<model>_pure_jordan.jsonl
#   metaqa_<model>_baseline_no_rag.jsonl
MODEL_KEYS = [
    "llama3_2_1b",
    "gemma2_2b",
    "llama3_2_3b",
    "phi3_mini",
    "mistral_7b",
    "llama3_1_8b",
]

# Όμορφα ονόματα για LaTeX
PRETTY_NAMES = {
    "llama3_2_1b": "Llama 3.2 1B",
    "gemma2_2b":   "Gemma 2 2B",
    "llama3_2_3b": "Llama 3.2 3B",
    "phi3_mini":   "Phi-3 Mini 3.8B",
    "mistral_7b":  "Mistral 7B",
    "llama3_1_8b": "Llama 3.1 8B",
}

# Διευθύνσεις φακέλων για τα MetaQA αρχεία
RAG_DIR = Path("metaqa_rag_results")
BASELINE_DIR = Path("metaqa_baseline_results")


def rag_path(model: str) -> Path:
    return RAG_DIR / f"metaqa_{model}_pure_jordan.jsonl"


def baseline_path(model: str) -> Path:
    return BASELINE_DIR / f"metaqa_{model}_baseline_no_rag.jsonl"


# =====================================================================
# Helpers
# =====================================================================

def load_jsonl(path: Path):
    """Generator που διαβάζει ένα .jsonl και επιστρέφει dict ανά γραμμή."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_file(path: Path):
    """
    Υπολογίζει για ένα .jsonl αρχείο:
      - Hits@1  (χρησιμοποιεί το top1_match field που ήδη υπολογίστηκε)
      - Exact Match (exact_match field)
      - Macro F1 (mean token-level F1, field: "f1")
    """
    n = 0
    hits1_count = 0
    em_count = 0
    f1_values = []

    for sample in load_jsonl(path):
        n += 1

        # Hits@1: χρησιμοποιούμε το ήδη υπολογισμένο top1_match
        if sample.get("top1_match"):
            hits1_count += 1

        # Exact Match (bool)
        if sample.get("exact_match"):
            em_count += 1

        # F1 (token-level, ήδη υπολογισμένο per-sample)
        f1 = sample.get("f1", None)
        if isinstance(f1, (int, float)):
            f1_values.append(float(f1))

    if n == 0:
        return {
            "n": 0,
            "hits1": 0.0,
            "macro_f1": 0.0,
            "em": 0.0,
        }

    hits1 = hits1_count / n
    macro_f1 = mean(f1_values) if f1_values else 0.0
    em = em_count / n

    return {
        "n": n,
        "hits1": hits1,
        "macro_f1": macro_f1,
        "em": em,
    }


# =====================================================================
# LaTeX table builders
# =====================================================================

def baseline_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline performance on MetaQA without RAG.}")
    lines.append(r"\label{tab:baseline_metaqa_full}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Hits@1 (\%) & Macro F1 (\%) & Exact Match (\%) \\")
    lines.append(r"\midrule")

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        macro_f1 = 100 * stats["macro_f1"]
        em = 100 * stats["em"]
        lines.append(
            f"{name} & {hits1:.2f} & {macro_f1:.2f} & {em:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def rag_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAG performance on MetaQA.}")
    lines.append(r"\label{tab:rag_metaqa_full}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Hits@1 (\%) & Macro F1 (\%) & Exact Match (\%) \\")
    lines.append(r"\midrule")

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        macro_f1 = 100 * stats["macro_f1"]
        em = 100 * stats["em"]
        lines.append(
            f"{name} & {hits1:.2f} & {macro_f1:.2f} & {em:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    baseline_metrics = {}
    rag_metrics = {}

    for key in MODEL_KEYS:
        # Baseline
        base_file = baseline_path(key)
        if base_file.exists():
            baseline_metrics[key] = aggregate_file(base_file)
        else:
            print(f"[WARN] Baseline file not found for {key}: {base_file}")
            baseline_metrics[key] = {"n": 0, "hits1": 0.0, "macro_f1": 0.0, "em": 0.0}

        # RAG
        rag_file = rag_path(key)
        if rag_file.exists():
            rag_metrics[key] = aggregate_file(rag_file)
        else:
            print(f"[WARN] RAG file not found for {key}: {rag_file}")
            rag_metrics[key] = {"n": 0, "hits1": 0.0, "macro_f1": 0.0, "em": 0.0}

    print("\n" + "="*70)
    print("Baseline (no-RAG) LaTeX table")
    print("="*70)
    print(baseline_latex_table(baseline_metrics))

    print("\n" + "="*70)
    print("RAG LaTeX table")
    print("="*70)
    print(rag_latex_table(rag_metrics))


if __name__ == "__main__":
    main()