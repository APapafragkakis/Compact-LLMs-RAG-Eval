# run_rag_only.py - EXACT Jordan Replication Runner

import json
from pathlib import Path
from time import perf_counter, sleep
import importlib
import sys

MODELS = [
    "llama3.2:1b",
    "llama3.1:8b",
    "mistral:7b",
    "phi3:mini",
    "llama3.2:3b",
    "gemma2:2b",
]
DATASET = "data/WC-P1.txt"
SLEEP_BETWEEN_MODELS = 10


def run_all_rag():
    print("\n" + "="*70)
    print("EXACT JORDAN REPLICATION - ALL MODELS")
    print("="*70 + "\n")
    
    results = []
    total_start = perf_counter()
    
    for idx, model in enumerate(MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {idx}/{len(MODELS)}: {model}")
        print(f"{'#'*70}\n")
        
        try:
            if 'eval_wc14_rag' in sys.modules:
                importlib.reload(sys.modules['eval_wc14_rag'])
            import eval_wc14_rag
            
            eval_wc14_rag.GENERATION_MODEL = model
            result = eval_wc14_rag.evaluate(DATASET)
            
            results.append({
                "model": model,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "time": result["time"],
                "status": "success"
            })
            
            print(f"✓ Complete: hits@1={result['accuracy']:.4f} ({result['correct']}/{result['total']})")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({"model": model, "status": "failed", "error": str(e)})
        
        if idx < len(MODELS):
            print(f"\nSleeping {SLEEP_BETWEEN_MODELS}s...\n")
            sleep(SLEEP_BETWEEN_MODELS)
    
    total_end = perf_counter()
    
    print("\n\n" + "="*70)
    print("EXACT JORDAN REPLICATION - FINAL SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'hits@1 (Accuracy)':<20} {'Correct/Total':<15}")
    print("-"*70)
    
    for r in results:
        if r["status"] == "success":
            print(f"{r['model']:<20} {r['accuracy']:.4f}               {r['correct']}/{r['total']}")
        else:
            print(f"{r['model']:<20} FAILED")
    
    print("="*70)
    print(f"Total time: {(total_end - total_start)/60:.1f} minutes")
    print("\nComparison to Jordan (gemma 12B): 0.9044")
    print("="*70 + "\n")
    
    with open("rag_summary_exact_jordan.json", "w") as f:
        json.dump({
            "method": "exact_jordan_replication",
            "reference": "Jordan's gemma 12B hits@1: 0.9044",
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    try:
        run_all_rag()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")