import json
from pathlib import Path
from statistics import mean
import csv

# =========================
# CONFIG
# =========================
DATASETS = {
    "metaqa": Path("metaqa_rag_results"),
    "wc14": Path("wc14_rag_results"),
}

OUTPUT_CSV = "rag_latency_summary.csv"


# =========================
# CORE
# =========================
def collect_from_folder(folder: Path, dataset_name: str):
    rows = []

    if not folder.exists():
        print(f"[WARN] Folder not found: {folder}")
        return rows

    for file in sorted(folder.glob("*.jsonl")):
        retr, gen, total = [], [], []

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                retr.append(obj.get("retrieval_latency", 0.0))
                gen.append(obj.get("generation_latency", 0.0))
                total.append(obj.get("total_latency", 0.0))

        if not total:
            continue

        rows.append({
            "dataset": dataset_name,
            "model": file.stem,
            "num_queries": len(total),

            # averages (per query)
            "retrieval_avg_sec": mean(retr),
            "llm_avg_sec": mean(gen),
            "endpoint_avg_sec": mean(total),

            # totals (pure endpoint time)
            "retrieval_total_sec": sum(retr),
            "llm_total_sec": sum(gen),
            "endpoint_total_sec": sum(total),
        })

    return rows


def main():
    all_rows = []

    for dataset, folder in DATASETS.items():
        rows = collect_from_folder(folder, dataset)
        all_rows.extend(rows)

    if not all_rows:
        print("No results found.")
        return

    # Pretty print
    print(
        f"{'Dataset':<8} {'Model':<40} {'N':<6} "
        f"{'Retr(s)':<10} {'LLM(s)':<10} {'Total(s)':<10}"
    )
    print("-" * 95)

    for r in all_rows:
        print(
            f"{r['dataset']:<8} "
            f"{r['model']:<40} "
            f"{r['num_queries']:<6} "
            f"{r['retrieval_avg_sec']:<10.2f} "
            f"{r['llm_avg_sec']:<10.2f} "
            f"{r['endpoint_avg_sec']:<10.2f}"
        )

    # CSV output (Excel / pandas / LaTeX friendly)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
