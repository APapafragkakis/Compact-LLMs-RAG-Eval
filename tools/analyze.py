import json
import statistics
import re
import argparse
import csv
from typing import List, Dict, Tuple
from collections import defaultdict

DEFAULT_RESULTS_PATH = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation\results\metaqa_llama1b_entity.jsonl"


def normalize_text(s: str) -> str:
    """
    Κανονικοποίηση για δίκαιη σύγκριση:
    - lower
    - αφαιρεί brackets/markup/στίξη που συχνά μπερδεύει
    - συμπτύσσει whitespace
    """
    if s is None:
        return ""
    s = s.strip().strip('"').strip("'")
    s = s.lower()
    s = re.sub(r"[<>\"“”‘’]", " ", s)
    s = re.sub(r"[\[\]\(\),:;!?]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> List[str]:
    return normalize_text(s).split()


def f1_score(pred: str, gold_list: List[str]) -> float:
    """Best token F1 over πολλαπλά golds."""
    best = 0.0
    pred_toks = tokenize(pred)
    pred_counts = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    for gold in gold_list:
        gold_toks = tokenize(gold)
        gold_counts = {}
        for t in gold_toks:
            gold_counts[t] = gold_counts.get(t, 0) + 1

        common = 0
        for t in pred_counts:
            if t in gold_counts:
                common += min(pred_counts[t], gold_counts[t])

        if common == 0:
            f1 = 0.0
        else:
            precision = common / len(pred_toks) if pred_toks else 0.0
            recall = common / len(gold_toks) if gold_toks else 0.0
            f1 = 0.0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

        if f1 > best:
            best = f1

    return best


def exact_match(pred: str, gold_list: List[str]) -> float:
    pred_norm = normalize_text(pred)
    for gold in gold_list:
        if pred_norm == normalize_text(gold):
            return 1.0
    return 0.0


def hits_at_1(pred: str, gold_list: List[str]) -> float:
    """True αν το gold υπάρχει ως substring μέσα στο prediction (lenient)."""
    pred_norm = normalize_text(pred)
    for gold in gold_list:
        gold_norm = normalize_text(gold)
        if gold_norm and gold_norm in pred_norm:
            return 1.0
    return 0.0


def summarize_metric_tuples(values: List[Tuple[float, float, float]]) -> Dict[str, float]:
    if not values:
        return {"EM": 0.0, "F1": 0.0, "Hits@1": 0.0}
    ems = [v[0] for v in values]
    f1s = [v[1] for v in values]
    h1s = [v[2] for v in values]
    return {
        "EM": sum(ems) / len(ems),
        "F1": sum(f1s) / len(f1s),
        "Hits@1": sum(h1s) / len(h1s),
    }


def summarize_latency(lats: List[float]) -> Dict[str, float]:
    if not lats:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    l_sorted = sorted(lats)
    idx = max(0, min(len(l_sorted) - 1, int(round(0.95 * (len(l_sorted) - 1)))))
    return {
        "mean": sum(l_sorted) / len(l_sorted),
        "median": statistics.median(l_sorted),
        "p95": l_sorted[idx],
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze MetaQA results (EM/F1/H@1/Latency)")
    ap.add_argument("--results", default=DEFAULT_RESULTS_PATH, help="Path to results JSONL")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    by_split = defaultdict(list)
    latencies_by_split = defaultdict(list)
    counts_by_split = defaultdict(int)
    total_lines = 0
    skipped = 0

    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue
            obj = json.loads(line)

            if not obj.get("ok"):
                skipped += 1
                continue

            split = obj.get("source_split", "unknown")
            counts_by_split[split] += 1

            gold_answers = obj.get("gold_answer", [])
            expanded_gold: List[str] = []
            for g in gold_answers:
                parts = [p.strip() for p in re.split(r"\|", g)]
                expanded_gold.extend([p for p in parts if p])

            pred_raw = obj.get("model_output_raw", "") or ""
            latency = obj.get("latency", None)

            em = exact_match(pred_raw, expanded_gold)
            f1 = f1_score(pred_raw, expanded_gold)
            h1 = hits_at_1(pred_raw, expanded_gold)

            by_split[split].append((em, f1, h1))
            if latency is not None:
                latencies_by_split[split].append(latency)

    print("=== PER-SPLIT METRICS ===")
    ordered_splits = ["1-hop", "2-hop", "3-hop", "unknown"]
    for split in ordered_splits:
        metrics = summarize_metric_tuples(by_split[split])
        latstats = summarize_latency(latencies_by_split[split])
        n = counts_by_split[split]
        print(f"\n[{split}]")
        print(f"Samples: {n}")
        print(f"EM:      {metrics['EM']:.3f}")
        print(f"F1:      {metrics['F1']:.3f}")
        print(f"Hits@1:  {metrics['Hits@1']:.3f}")
        print(f"Latency mean:   {latstats['mean']:.3f}s")
        print(f"Latency median: {latstats['median']:.3f}s")
        print(f"Latency p95:    {latstats['p95']:.3f}s")

    all_metric_tuples: List[Tuple[float, float, float]] = []
    all_lats: List[float] = []
    total_ok = 0
    for split, vals in by_split.items():
        all_metric_tuples.extend(vals)
        total_ok += len(vals)
    for split, lats in latencies_by_split.items():
        all_lats.extend(lats)

    global_m = summarize_metric_tuples(all_metric_tuples)
    global_l = summarize_latency(all_lats)

    print("\n=== GLOBAL METRICS ===")
    print(f"Total OK samples: {total_ok} / lines read: {total_lines} (skipped: {skipped})")
    print(f"EM:      {global_m['EM']:.3f}")
    print(f"F1:      {global_m['F1']:.3f}")
    print(f"Hits@1:  {global_m['Hits@1']:.3f}")
    print(f"Latency mean:   {global_l['mean']:.3f}s")
    print(f"Latency median: {global_l['median']:.3f}s")
    print(f"Latency p95:    {global_l['p95']:.3f}s")

    if args.csv:
        os_rows = []
        for split in ordered_splits:
            m = summarize_metric_tuples(by_split[split])
            l = summarize_latency(latencies_by_split[split])
            os_rows.append({
                "split": split,
                "samples": counts_by_split[split],
                "EM": f"{m['EM']:.3f}",
                "F1": f"{m['F1']:.3f}",
                "Hits@1": f"{m['Hits@1']:.3f}",
                "lat_mean_s": f"{l['mean']:.3f}",
                "lat_median_s": f"{l['median']:.3f}",
                "lat_p95_s": f"{l['p95']:.3f}",
            })
        os_rows.append({
            "split": "GLOBAL",
            "samples": total_ok,
            "EM": f"{global_m['EM']:.3f}",
            "F1": f"{global_m['F1']:.3f}",
            "Hits@1": f"{global_m['Hits@1']:.3f}",
            "lat_mean_s": f"{global_l['mean']:.3f}",
            "lat_median_s": f"{global_l['median']:.3f}",
            "lat_p95_s": f"{global_l['p95']:.3f}",
        })
        with open(args.csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(os_rows[0].keys()))
            writer.writeheader()
            writer.writerows(os_rows)
        print(f"\n[CSV] Saved metrics to {args.csv}")


if __name__ == "__main__":
    main()
