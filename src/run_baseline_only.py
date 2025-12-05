# run_baseline_only.py - Τρέχει ΜΟΝΟ Baseline (no-RAG) για όλα τα models

import json
from pathlib import Path
from time import perf_counter, sleep
import importlib
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = [
    "llama3.2:1b",
    "llama3.1:8b",
    "mistral:7b",
    "phi3:mini",
    "llama3.2:3b",
    "gemma2:2b",
]

DATASET = "data/metaqa_1hop_only.jsonl"
SLEEP_BETWEEN_MODELS = 10  # seconds


# ============================================================================
# RUN BASELINE EVALUATIONS ONLY
# ============================================================================

def run_all_baseline():
    print("\n" + "="*70)
    print("RUNNING BASELINE (NO-RAG) EVALUATIONS FOR ALL MODELS")
    print("="*70)
    print(f"Models to test: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Dataset: {DATASET}")
    print(f"Sleep between models: {SLEEP_BETWEEN_MODELS} sec")
    print("="*70 + "\n")

    # Prepare output folder
    outdir = Path("results_baseline_only")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # CSV για συγκεντρωτικά αποτελέσματα
    summary_path = outdir / "baseline_summary.json"
    
    # Για να μπορούμε να συνεχίσουμε αν κοπεί κάπου
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # Για να μην ξανατρέχουμε όσα έχουν ήδη τρέξει
    done_models = {r["model"] for r in results}

    # Χρονόμετρο για συνολικό χρόνο
    global_start = perf_counter()

    for model_name in MODELS:
        if model_name in done_models:
            print(f"Skipping {model_name} (already in summary).")
            continue

        print("\n" + "-"*70)
        print(f"Running BASELINE for model: {model_name}")
        print("-"*70)

        # Φορτώνουμε δυναμικά το remote_baseline.py (ή όπως λέγεται το script σου)
        try:
            # Αν είναι σε module π.χ. evaluation.remote_baseline, το αλλάζεις:
            baseline_module_name = "remote_baseline"
            if baseline_module_name in sys.modules:
                importlib.reload(sys.modules[baseline_module_name])
                baseline = sys.modules[baseline_module_name]
            else:
                baseline = importlib.import_module(baseline_module_name)
        except Exception as e:
            print(f"ERROR: Could not import baseline module: {e}")
            results.append({
                "model": model_name,
                "status": "import_error",
                "accuracy": None,
                "error": str(e)
            })
            # Αποθήκευση ενδιάμεσα
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

        # Ρυθμίζουμε το μοντέλο που θα χρησιμοποιήσει το baseline
        try:
            # Υποθέτουμε ότι στο remote_baseline.py υπάρχει μια μεταβλητή GENERATION_MODEL
            baseline.GENERATION_MODEL = model_name
            print(f"Set GENERATION_MODEL = {baseline.GENERATION_MODEL}")
        except Exception as e:
            print(f"ERROR: Could not set GENERATION_MODEL: {e}")
            results.append({
                "model": model_name,
                "status": "config_error",
                "accuracy": None,
                "error": str(e)
            })
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            continue

        # Τρέχουμε την αξιολόγηση baseline
        try:
            start = perf_counter()
            # Υποθέτουμε ότι υπάρχει συνάρτηση evaluate(dataset_path) που επιστρέφει accuracy
            acc = baseline.evaluate(DATASET)
            end = perf_counter()
            elapsed = end - start

            print(f"✓ Baseline complete: Acc={acc:.4f}")
            print(f"  Time: {elapsed:.2f} sec")

            results.append({
                "model": model_name,
                "status": "success",
                "accuracy": acc,
                "time_sec": elapsed
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"ERROR while running baseline for {model_name}: {e}")
            results.append({
                "model": model_name,
                "status": "runtime_error",
                "accuracy": None,
                "error": str(e)
            })

        # Αποθήκευση μετά από κάθε μοντέλο
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Μικρή παύση ανάμεσα στα models
        print(f"Sleeping {SLEEP_BETWEEN_MODELS} sec before next model...")
        sleep(SLEEP_BETWEEN_MODELS)

    # Τελικό summary στην κονσόλα
    total_time = perf_counter() - global_start
    print("\n" + "="*70)
    print("BASELINE EVALUATION SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"{'Model':<20} {'Accuracy':<12} {'Status':<10}")
    print("-"*70)
    for r in results:
        model = r["model"]
        if r["accuracy"] is not None:
            acc = f"{r['accuracy']:.4f}"
        else:
            acc = "FAILED"
        status = "✓" if r["status"] == "success" else "✗"
        print(f"{model:<20} {acc:<12} {status:<10}")

    # Save τελικό summary ξανά (με total_time)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "total_time_sec": total_time,
            "total_time_minutes": total_time / 60,
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("Results files: simple_results/metaqa_<model>.jsonl")
    print("\n" + "="*70)
    print("BASELINE EVALUATIONS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_all_baseline()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

