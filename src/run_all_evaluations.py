# run_all_evaluations.py - Master Script to Run All Model Evaluations

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
SLEEP_BETWEEN_MODELS = 10  # seconds between different models


# ============================================================================
# RUN ALL EVALUATIONS
# ============================================================================

def run_all():
    print("\n" + "="*70)
    print("RUNNING EVALUATIONS FOR ALL MODELS")
    print("="*70)
    print(f"Models to test: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Dataset: {DATASET}")
    print(f"Sleep between requests: 0.7s")
    print(f"Sleep between models: {SLEEP_BETWEEN_MODELS}s")
    print("="*70 + "\n")
    
    results_summary = []
    total_start = perf_counter()
    
    for idx, model in enumerate(MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {idx}/{len(MODELS)}: {model}")
        print(f"{'#'*70}\n")
        
        # 1. Run RAG evaluation
        print(f"[1/2] Running RAG evaluation for {model}...")
        rag_acc = None
        rag_time = None
        try:
            # Import and reload to get fresh module
            if 'eval_metaqa_rag' in sys.modules:
                importlib.reload(sys.modules['eval_metaqa_rag'])
            import eval_metaqa_rag
            
            # Set model
            eval_metaqa_rag.GENERATION_MODEL = model
            
            # Run evaluation
            rag_result = eval_metaqa_rag.evaluate(DATASET)
            rag_acc = rag_result["accuracy"]
            rag_time = rag_result["time"]
            print(f"✓ RAG evaluation complete: {rag_acc:.2f}%")
        except Exception as e:
            print(f"✗ RAG evaluation failed: {e}")
        
        # Small sleep between RAG and Baseline
        sleep(2)
        
        # 2. Run Baseline (no-RAG) evaluation
        print(f"\n[2/2] Running Baseline (no-RAG) evaluation for {model}...")
        baseline_acc = None
        baseline_time = None
        try:
            # Import and reload to get fresh module
            if 'eval_metaqa_simple' in sys.modules:
                importlib.reload(sys.modules['eval_metaqa_simple'])
            import eval_metaqa_simple
            
            # Set model
            eval_metaqa_simple.GENERATION_MODEL = model
            
            # Run evaluation
            baseline_result = eval_metaqa_simple.evaluate(DATASET)
            baseline_acc = baseline_result["accuracy"]
            baseline_time = baseline_result["time"]
            print(f"✓ Baseline evaluation complete: {baseline_acc:.2f}%")
        except Exception as e:
            print(f"✗ Baseline evaluation failed: {e}")
        
        # Calculate improvement
        improvement = None
        if rag_acc is not None and baseline_acc is not None:
            improvement = rag_acc - baseline_acc
        
        # Store results
        results_summary.append({
            "model": model,
            "rag_accuracy": rag_acc,
            "baseline_accuracy": baseline_acc,
            "improvement": improvement,
            "rag_time": rag_time,
            "baseline_time": baseline_time,
        })
        
        # Print summary for this model
        print(f"\n{'-'*70}")
        print(f"Summary for {model}:")
        if rag_acc is not None:
            print(f"  RAG:        {rag_acc:.2f}%")
        else:
            print(f"  RAG:        FAILED")
        if baseline_acc is not None:
            print(f"  Baseline:   {baseline_acc:.2f}%")
        else:
            print(f"  Baseline:   FAILED")
        if improvement is not None:
            print(f"  Improvement: {improvement:+.2f}%")
        print(f"{'-'*70}")
        
        # Sleep between models (except after last one)
        if idx < len(MODELS):
            print(f"\nSleeping {SLEEP_BETWEEN_MODELS}s before next model...\n")
            sleep(SLEEP_BETWEEN_MODELS)
    
    total_end = perf_counter()
    total_time = total_end - total_start
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Baseline':<12} {'RAG':<12} {'Improvement':<15}")
    print("-"*70)
    
    for r in results_summary:
        model = r["model"]
        baseline = f"{r['baseline_accuracy']:.2f}%" if r['baseline_accuracy'] is not None else "FAILED"
        rag = f"{r['rag_accuracy']:.2f}%" if r['rag_accuracy'] is not None else "FAILED"
        improvement = f"{r['improvement']:+.2f}%" if r['improvement'] is not None else "N/A"
        
        print(f"{model:<20} {baseline:<12} {rag:<12} {improvement:<15}")
    
    print("="*70)
    print(f"Total evaluation time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print("="*70)
    
    # Save summary to JSON
    summary_path = Path("evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "models": MODELS,
            "dataset": DATASET,
            "results": results_summary,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
        }, f, indent=2)
    
    print(f"\nDetailed summary saved to: {summary_path}")
    print("\nResults files location:")
    print("  RAG results:      rag_results/metaqa_<model>.jsonl")
    print("  Baseline results: simple_results/metaqa_<model>.jsonl")
    print("\n" + "="*70)
    print("ALL EVALUATIONS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_all()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("Partial results may be saved in rag_results/ and simple_results/")