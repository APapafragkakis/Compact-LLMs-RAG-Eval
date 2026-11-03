"""Recompute Hits@1, F1, EM and pairwise McNemar tests from the outputs in out/.

Run from the repository root:
    python tools/compute_metrics.py
"""

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

MODELS = [
    ("llama3_2_1b", "L3.2-1B"),
    ("gemma2_2b", "G2-2B"),
    ("llama3_2_3b", "L3.2-3B"),
    ("phi3_mini", "Phi-3"),
    ("mistral_7b", "M-7B"),
    ("llama3_1_8b", "L3.1-8B"),
]

DATASETS = {
    "MetaQA": {
        "rag": "out/metaqa_rag_results/metaqa_{}_pure_.jsonl",
        "baseline": "out/metaqa_baseline_results/metaqa_{}_baseline_no_rag.jsonl",
    },
    "WC2014QA": {
        "rag": "out/wc14_rag_results/wc2014_{}_.jsonl",
        "baseline": "out/wc14_baseline_results/wc2014_{}_baseline_no_rag.jsonl",
    },
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["id"])


def normalize_wc14(pred):
    # Same normalization as src/eval_wc14_rag.py
    return pred.rstrip().replace(" ", "_").lower().split(",")[0]


def token_f1(pred, gold):
    pred_tokens = pred.replace("_", " ").split()
    gold_tokens = gold.replace("_", " ").split()
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_metaqa(rows):
    # Per-question scores are already stored by src/eval_metaqa_rag.py
    hits = [bool(r["top1_match"]) for r in rows]
    em = [bool(r["exact_match"]) for r in rows]
    f1 = [r["f1"] for r in rows]
    return hits, f1, em


def score_wc14(rows):
    hits, f1, em = [], [], []
    for r in rows:
        pred = normalize_wc14(r["prediction"])
        valid = [a.lower() for a in r["valid_answers"]]
        hits.append(pred in valid)
        em.append(pred == valid[0])
        f1.append(max(token_f1(pred, a) for a in valid))
    return hits, f1, em


def mcnemar(a, b):
    """McNemar test with continuity correction. Returns the two-sided p-value."""
    only_b = sum((not x) and y for x, y in zip(a, b))
    only_a = sum(x and (not y) for x, y in zip(a, b))
    if only_a + only_b == 0:
        return 1.0
    chi2 = (abs(only_b - only_a) - 1) ** 2 / (only_a + only_b)
    return math.erfc(math.sqrt(chi2 / 2))


def significance(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def pct(values):
    return 100 * sum(values) / len(values)


def main():
    for name, paths in DATASETS.items():
        score = score_metaqa if name == "MetaQA" else score_wc14
        rag_hits = {}

        print(f"\n{name}")
        print(f"{'Model':<10} {'Base H@1':>9} {'Base F1':>8} {'Base EM':>8} {'RAG H@1':>8} {'RAG F1':>7} {'RAG EM':>7}")

        for key, label in MODELS:
            base = score(load_jsonl(paths["baseline"].format(key)))
            rag = score(load_jsonl(paths["rag"].format(key)))
            rag_hits[label] = rag[0]
            values = [pct(m) for m in base + rag]
            print(f"{label:<10} " + " ".join(f"{v:>8.2f}" for v in values))

        print(f"\n{name} McNemar (RAG Hits@1)")
        print(f"{'Comparison':<20} {'Delta pp':>9} {'p-value':>10}  Sig.")

        for (_, a), (_, b) in combinations(MODELS, 2):
            delta = pct(rag_hits[b]) - pct(rag_hits[a])
            p = mcnemar(rag_hits[a], rag_hits[b])
            print(f"{a + ' vs ' + b:<20} {delta:>+9.2f} {p:>10.2g}  {significance(p)}")


if __name__ == "__main__":
    main()
