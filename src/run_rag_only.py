# run_rag_only.py - Τρέχει ΜΟΝΟ RAG για όλα τα models

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
    "gemma2:2b"
]

DATASET = "data/metaqa_1hop_only.jsonl"
SLEEP_BETWEEN_MODELS = 10  # seconds


# ============================================================================
# RUN RAG EVALUATIONS ONLY
# ============================================================================

def run_all_rag():
    print("\n" + "="*70)
    print("RUNNING RAG EVALUATIONS FOR ALL MODELS")
    print("="*70)
    print(f"Models to test: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Dataset: {DATASET}")
    print("="*70 + "\n")
    
    results = []
    total_start = perf_counter()
    
    for idx, model in enumerate(MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# RAG MODEL {idx}/{len(MODELS)}: {model}")
        print(f"{'#'*70}\n")
        
        try:
            # Import and reload
            if 'eval_metaqa_rag' in sys.modules:
                importlib.reload(sys.modules['eval_metaqa_rag'])
            import eval_metaqa_rag
            
            # Set model
            eval_metaqa_rag.GENERATION_MODEL = model
            
            # Run evaluation
            result = eval_metaqa_rag.evaluate(DATASET)
            acc = result["accuracy"]
            time_taken = result["time"]
            
            results.append({
                "model": model,
                "accuracy": acc,
                "time": time_taken,
                "status": "success"
            })
            
            print(f"✓ RAG complete: {acc:.2f}%")
            
        except Exception as e:
            print(f"✗ RAG failed: {e}")
            results.append({
                "model": model,
                "accuracy": None,
                "time": None,
                "status": "failed",
                "error": str(e)
            })
        
        # Sleep between models
        if idx < len(MODELS):
            print(f"\nSleeping {SLEEP_BETWEEN_MODELS}s before next model...\n")
            sleep(SLEEP_BETWEEN_MODELS)
    
    total_end = perf_counter()
    total_time = total_end - total_start
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n\n" + "="*70)
    print("RAG RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<15} {'Status':<10}")
    print("-"*70)
    
    for r in results:
        model = r["model"]
        acc = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else "FAILED"
        status = "✓" if r['status'] == "success" else "✗"
        print(f"{model:<20} {acc:<15} {status:<10}")
    
    print("="*70)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print("="*70)
    
    # Save summary
    summary_path = Path("rag_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "method": "rag_with_retrieval",
            "models": MODELS,
            "dataset": DATASET,
            "results": results,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("Results files: rag_results/metaqa_<model>.jsonl")
    print("\n" + "="*70)
    print("RAG EVALUATIONS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_all_rag()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")