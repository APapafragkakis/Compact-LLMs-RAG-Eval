# run_rag_only.py - UNCHANGED

import json
from pathlib import Path
from time import perf_counter, sleep
import importlib
import sys

MODELS = [
#    "llama3.2:1b",
#    "llama3.1:8b",
#   "mistral:7b",
#    "phi3:mini",
    "llama3.2:3b",
    "gemma2:2b",
]

DATASET = "data/metaqa.jsonl"
SLEEP_BETWEEN_MODELS = 10


def run_all_rag():
    print("\n" + "="*70)
    print("PURE JORDAN REPLICATION - ALL MODELS")
    print("="*70 + "\n")
    
    results = []
    total_start = perf_counter()
    
    for idx, model in enumerate(MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {idx}/{len(MODELS)}: {model}")
        print(f"{'#'*70}\n")
        
        try:
            if 'eval_metaqa_rag' in sys.modules:
                importlib.reload(sys.modules['eval_metaqa_rag'])
            import eval_metaqa_rag
            
            eval_metaqa_rag.GENERATION_MODEL = model
            result = eval_metaqa_rag.evaluate(DATASET)
            
            results.append({
                "model": model,
                "accuracy": result["accuracy"],
                "exact_match": result["exact_match"],
                "macro_f1": result["macro_f1"],
                "time": result["time"],
                "status": "success"
            })
            
            print(f"✓ Complete: Acc={result['accuracy']:.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({"model": model, "status": "failed", "error": str(e)})
        
        if idx < len(MODELS):
            print(f"\nSleeping {SLEEP_BETWEEN_MODELS}s...\n")
            sleep(SLEEP_BETWEEN_MODELS)
    
    total_end = perf_counter()
    
    print("\n\n" + "="*70)
    print("PURE JORDAN REPLICATION - FINAL SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'EM':<12} {'Macro F1':<12}")
    print("-"*70)
    
    for r in results:
        if r["status"] == "success":
            print(f"{r['model']:<20} {r['accuracy']:.4f}       {r['exact_match']:.4f}       {r['macro_f1']:.4f}")
        else:
            print(f"{r['model']:<20} FAILED")
    
    print("="*70)
    print(f"Total time: {(total_end - total_start)/60:.1f} minutes")
    print("\nComparison to Jordan (gemma 12B): 0.9044")
    print("="*70 + "\n")
    
    with open("rag_summary_pure_jordan.json", "w") as f:
        json.dump({
            "method": "pure_jordan_replication",
            "reference": "Jordan's gemma 12B accuracy: 0.9044",
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    try:
        run_all_rag()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")