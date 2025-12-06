# run_baseline_only.py - Τρέχει ΜΟΝΟ Baseline (no-RAG) για όλα τα models
# Με μοντέλο-level resume (summary) και sample-level resume από eval_metaqa_simple

import json
from pathlib import Path
from time import perf_counter, sleep
import importlib
import sys

# ====================================================================
# CONFIGURATION
# ====================================================================

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


# ====================================================================
# RUN BASELINE EVALUATIONS ONLY
# ====================================================================

def run_all_baseline():
    print("\n" + "="*70)
    print("RUNNING BASELINE (NO-RAG) EVALUATIONS FOR ALL MODELS")
    print("="*70)
    print(f"Models to test: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Dataset: {DATASET}")
    print(f"Sleep between models: {SLEEP_BETWEEN_MODELS} sec")
    print("="*70 + "\n")

    # Output folder for summary
    outdir = Path("results_baseline_only")
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "baseline_summary.json"

    # Load previous results (για resume σε επίπεδο model)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            prev = json.load(f)
        results = prev.get("results", []) if isinstance(prev, dict) else prev
    else:
        results = []

    done_models = {r["model"] for r in results if "model" in r}

    global_start = perf_counter()

    for model_name in MODELS:
        # Αν το model υπάρχει ήδη στο summary ως success, μην το ξανατρέχεις
        if model_name in done_models:
            print(f"Skipping {model_name} (already in summary).")
            continue

        print("\n" + "-"*70)
        print(f"Running BASELINE for model: {model_name}")
        print("-"*70)

        # Load eval_metaqa_simple
        try:
            module_name = "eval_metaqa_simple"
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                baseline = sys.modules[module_name]
            else:
                baseline = importlib.import_module(module_name)
        except Exception as e:
            print(f"ERROR: Could not import baseline module: {e}")
            results.append({
                "model": model_name,
                "status": "import_error",
                "accuracy": None,
                "error": str(e),
            })
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)
            continue

        # Set LLM model name
        try:
            baseline.GENERATION_MODEL = model_name
            print(f"Set GENERATION_MODEL = {baseline.GENERATION_MODEL}")
        except Exception as e:
            print(f"ERROR: Could not set GENERATION_MODEL: {e}")
            results.append({
                "model": model_name,
                "status": "config_error",
                "accuracy": None,
                "error": str(e),
            })
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)
            continue

        # Run evaluation (με sample-level resume από το eval_metaqa_simple)
        try:
            start = perf_counter()
            # Χρησιμοποιούμε πάντα resume=True για να συνεχίζει από όπου σταμάτησε
            eval_res = baseline.evaluate(DATASET, resume=True)
            end = perf_counter()
            elapsed_wrapper = end - start

            acc = eval_res.get("accuracy", None)
            total = eval_res.get("total", None)
            elapsed_eval = eval_res.get("time", None)

            if acc is not None:
                print(f"✓ Baseline complete: Acc={acc:.4f}")
            else:
                print("✓ Baseline complete (no accuracy returned)")

            if elapsed_eval is not None:
                print(f"  Eval time (inside evaluate): {elapsed_eval:.2f} sec")
            print(f"  Wrapper time: {elapsed_wrapper:.2f} sec")

            results.append({
                "model": model_name,
                "status": "success",
                "accuracy": acc,
                "total_samples": total,
                "time_sec": elapsed_eval if elapsed_eval is not None else elapsed_wrapper,
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"ERROR while running baseline for {model_name}: {e}")
            results.append({
                "model": model_name,
                "status": "runtime_error",
                "accuracy": None,
                "error": str(e),
            })

        # Save intermediate summary
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)

        print(f"Sleeping {SLEEP_BETWEEN_MODELS} sec before next model...")
        sleep(SLEEP_BETWEEN_MODELS)

    total_time = perf_counter() - global_start

    # Final summary
    print("\n" + "="*70)
    print("BASELINE EVALUATION SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"{'Model':<20} {'Accuracy':<12} {'Status':<10}")
    print("-"*70)
    for r in results:
        model = r.get("model", "?")
        acc = r.get("accuracy", None)
        status = "✓" if r.get("status") == "success" else "✗"
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "FAILED"
        print(f"{model:<20} {acc_str:<12} {status:<10}")

    # Save final summary
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "method": "no_rag_baseline",
            "results": results,
            "total_time_sec": total_time,
            "total_time_minutes": total_time / 60,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")
    print("Per-sample files: baseline_results/metaqa_<model>_baseline_no_rag.jsonl")
    print("\n" + "="*70)
    print("BASELINE EVALUATIONS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_all_baseline()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
