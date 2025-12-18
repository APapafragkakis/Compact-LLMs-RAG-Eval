# tools/postprocess_wc14_metrics.py
#
# Offline υπολογισμός metrics για WC14 (RAG + Baseline)
# Διαβάζουμε τα .jsonl ανά δείγμα και βγάζουμε ΜΟΝΟ:
#   - Hits@1 (από το top1_match field)
# και τυπώνουμε LaTeX tables και console summary για RAG και Baseline.

import json
from pathlib import Path
from statistics import mean

# =====================================================================
# CONFIG
# =====================================================================

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
    "gemma2_2b":   "Gemma 2 2B",
    "llama3_2_3b": "Llama 3.2 3B",
    "phi3_mini":   "Phi-3 Mini 3.8B",
    "mistral_7b":  "Mistral 7B",
    "llama3_1_8b": "Llama 3.1 8B",
}

RAG_DIR = Path(__file__).parent.parent / "wc14_rag_results"
BASELINE_DIR = Path(__file__).parent.parent / "wc14_baseline_results"


def rag_path(model: str) -> Path:
    return RAG_DIR / f"wc2014_{model}_jordan.jsonl"


def baseline_path(model: str) -> Path:
    return BASELINE_DIR / f"wc2014_{model}_baseline_no_rag.jsonl"


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
    """
    n = 0
    hits1_flags = []

    for sample in load_jsonl(path):
        n += 1
        hits1_flags.append(bool(sample.get("top1_match", False)))

    if n == 0:
        return {
            "n": 0,
            "hits1": 0.0,
        }

    hits1 = mean(hits1_flags)

    return {
        "n": n,
        "hits1": hits1,
    }


# =====================================================================
# LaTeX table builders (μόνο Hits@1)
# =====================================================================

def baseline_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline performance on WC14 dataset (no RAG).}")
    lines.append(r"\label{tab:baseline_wc14_full}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Hits@1 (\%) \\")
    lines.append(r"\midrule")

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        lines.append(f"{name} & {hits1:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def rag_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAG performance on WC14 dataset.}")
    lines.append(r"\label{tab:rag_wc14_full}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Hits@1 (\%) \\")
    lines.append(r"\midrule")

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        hits1 = 100 * stats["hits1"]
        lines.append(f"{name} & {hits1:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# Console summaries
# =====================================================================

def print_comparison_summary(baseline_metrics, rag_metrics):
    """Τυπώνει side-by-side comparison μόνο για Hits@1."""
    print("\n" + "="*100)
    print("WC14 RESULTS COMPARISON: BASELINE vs RAG")
    print("="*100)
    print(f"{'Model':<25} {'Baseline Hits@1':>15} {'RAG Hits@1':>15} {'Improvement':>15}")
    print("-"*100)

    for key in MODEL_KEYS:
        b_stats = baseline_metrics.get(key)
        r_stats = rag_metrics.get(key)

        if (not b_stats or b_stats["n"] == 0) and (not r_stats or r_stats["n"] == 0):
            continue

        name = PRETTY_NAMES.get(key, key)

        b_hits1 = 100 * b_stats["hits1"] if b_stats and b_stats["n"] > 0 else 0.0
        r_hits1 = 100 * r_stats["hits1"] if r_stats and r_stats["n"] > 0 else 0.0
        improvement = r_hits1 - b_hits1

        improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"

        print(f"{name:<25} {b_hits1:>14.2f}% {r_hits1:>14.2f}% {improvement_str:>15}")

    print("="*100)


def print_detailed_summary(title, metrics_dict):
    """Τυπώνει detailed metrics για ένα dataset (Samples + Hits@1)."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    print(f"{'Model':<25} {'Samples':>10} {'Hits@1':>12}")
    print("-"*80)

    for key in MODEL_KEYS:
        stats = metrics_dict.get(key)
        if not stats or stats["n"] == 0:
            continue
        name = PRETTY_NAMES.get(key, key)
        n = stats["n"]
        hits1 = 100 * stats["hits1"]
        print(f"{name:<25} {n:>10} {hits1:>11.2f}%")

    print("="*80)


# =====================================================================
# Main
# =====================================================================

def main():
    baseline_metrics = {}
    rag_metrics = {}

    print("Processing WC14 datasets...")
    print("\n" + "-"*50)
    print("BASELINE (No RAG)")
    print("-"*50)
    for key in MODEL_KEYS:
        baseline_file = baseline_path(key)
        if baseline_file.exists():
            print(f"  Processing {key}...")
            baseline_metrics[key] = aggregate_file(baseline_file)
        else:
            print(f"  [WARN] File not found: {baseline_file}")
            baseline_metrics[key] = {"n": 0, "hits1": 0.0}

    print("\n" + "-"*50)
    print("RAG")
    print("-"*50)
    for key in MODEL_KEYS:
        rag_file = rag_path(key)
        if rag_file.exists():
            print(f"  Processing {key}...")
            rag_metrics[key] = aggregate_file(rag_file)
        else:
            print(f"  [WARN] File not found: {rag_file}")
            rag_metrics[key] = {"n": 0, "hits1": 0.0}

    # Print comparison
    print_comparison_summary(baseline_metrics, rag_metrics)

    # Print detailed summaries
    print_detailed_summary("WC14 BASELINE (NO-RAG) DETAILED RESULTS", baseline_metrics)
    print_detailed_summary("WC14 RAG DETAILED RESULTS", rag_metrics)

    # Print LaTeX tables
    print("\n" + "="*100)
    print("LaTeX Table - BASELINE")
    print("="*100)
    print(baseline_latex_table(baseline_metrics))

    print("\n" + "="*100)
    print("LaTeX Table - RAG")
    print("="*100)
    print(rag_latex_table(rag_metrics))
    print("\n")


if __name__ == "__main__":
    main()
