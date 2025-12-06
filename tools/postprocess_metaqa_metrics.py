# tools/postprocess_metaqa_metrics.py
#
# Offline υπολογισμός metrics για MetaQA (baseline + RAG)
# Χωρίς κανένα νέο LLM call. Διαβάζει τα JSONL αρχεία ανά δείγμα
# και βγάζει:
#   - EM (Exact Match)
#   - Hits@1
#   - token-level F1 (ήδη υπάρχει ως "f1" per sample)
#   - mean latency (total_latency)
# και τυπώνει δύο LaTeX tables (baseline + RAG).

import json
from pathlib import Path
from statistics import mean

# =====================================================================
# CONFIG: Μοντέλα & paths
# =====================================================================

# Short names, όπως εμφανίζονται στα filenames:
#   metaqa_<model>_pure_jordan.jsonl
#   metaqa_<model>_baseline_no_rag.jsonl
MODELS = [
    "llama3_2_1b",
    "llama3_1_8b",
    "mistral_7b",
    "phi3_mini",
    "llama3_2_3b",
    "gemma2_2b",
]

# RAG files: rag_results/metaqa_<model>_pure_jordan.jsonl
# Baseline files: baseline_results/metaqa_<model>_baseline_no_rag.jsonl
RAG_DIR = Path("rag_results")
BASELINE_DIR = Path("baseline_results")


def rag_path(model: str) -> Path:
    return RAG_DIR / f"metaqa_{model}_pure_jordan.jsonl"


def baseline_path(model: str) -> Path:
    return BASELINE_DIR / f"metaqa_{model}_baseline_no_rag.jsonl"


# =====================================================================
# Helpers
# =====================================================================

def load_jsonl(path: Path):
    """Generator που διαβάζει ένα .jsonl αρχείο και επιστρέφει dict ανά γραμμή."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_file(path: Path):
    """
    Υπολογίζει:
      - EM (exact_match)
      - Hits@1 (gold answer ως substring στο prediction)
      - mean token-level F1 (field: "f1")
      - mean latency (field: "total_latency")
    για ένα .jsonl αρχείο.
    """
    n = 0
    em_count = 0
    hits1_count = 0
    f1_values = []
    latencies = []

    for sample in load_jsonl(path):
        n += 1

        # Exact Match (bool)
        if sample.get("exact_match"):
            em_count += 1

        # Token-level F1 (ήδη υπάρχει per-sample)
        f1 = sample.get("f1", None)
        if isinstance(f1, (int, float)):
            f1_values.append(f1)

        # Latency (total latency = retrieval + generation για RAG,
        # ή μόνο generation για baseline)
        lat = sample.get("total_latency", None)
        if isinstance(lat, (int, float)):
            latencies.append(lat)

        # Hits@1: αν κάποιο από τα gold answers εμφανίζεται ως substring
        gold = sample.get("gold_answer", [])
        if isinstance(gold, str):
            gold = [gold]

        pred = sample.get("prediction") or ""
        pred_low = pred.lower()

        hit = False
        for g in gold:
            g = (g or "").strip()
            if not g:
                continue
            if g.lower() in pred_low:
                hit = True
                break

        if hit:
            hits1_count += 1

    if n == 0:
        return {
            "n": 0,
            "em": 0.0,
            "hits1": 0.0,
            "f1": 0.0,
            "latency": 0.0,
        }

    em = em_count / n
    hits1 = hits1_count / n
    mean_f1 = mean(f1_values) if f1_values else 0.0
    mean_lat = mean(latencies) if latencies else 0.0

    return {
        "n": n,
        "em": em,
        "hits1": hits1,
        "f1": mean_f1,
        "latency": mean_lat,
    }


# =====================================================================
# LaTeX table builders
# =====================================================================

def baseline_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline (χωρίς RAG) απόδοση στο MetaQA 1-hop.}")
    lines.append(r"\label{tab:baseline_metaqa_full}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & EM (\%) & Hits@1 (\%) & F1 (\%) & Latency (s) \\")
    lines.append(r"\midrule")

    for m, stats in metrics_dict.items():
        if stats is None or stats["n"] == 0:
            continue
        em = 100 * stats["em"]
        hits1 = 100 * stats["hits1"]
        f1 = 100 * stats["f1"]
        lat = stats["latency"]
        lines.append(
            f"{m} & {em:.2f} & {hits1:.2f} & {f1:.2f} & {lat:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def rag_latex_table(metrics_dict):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAG απόδοση στο MetaQA 1-hop (pure Jordan replication).}")
    lines.append(r"\label{tab:rag_metaqa_full}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & EM (\%) & Hits@1 (\%) & F1 (\%) & Latency (s) \\")
    lines.append(r"\midrule")

    for m, stats in metrics_dict.items():
        if stats is None or stats["n"] == 0:
            continue
        em = 100 * stats["em"]
        hits1 = 100 * stats["hits1"]
        f1 = 100 * stats["f1"]
        lat = stats["latency"]
        lines.append(
            f"{m} & {em:.2f} & {hits1:.2f} & {f1:.2f} & {lat:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    rag_metrics = {}
    baseline_metrics = {}

    for m in MODELS:
        # RAG
        rag_file = rag_path(m)
        if rag_file.exists():
            rag_metrics[m] = aggregate_file(rag_file)
        else:
            print(f"[WARN] RAG file not found for {m}: {rag_file}")
            rag_metrics[m] = None

        # Baseline
        base_file = baseline_path(m)
        if base_file.exists():
            baseline_metrics[m] = aggregate_file(base_file)
        else:
            print(f"[WARN] Baseline file not found for {m}: {base_file}")
            baseline_metrics[m] = None

    # -----------------------
    # Εκτύπωση LaTeX tables
    # -----------------------
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
