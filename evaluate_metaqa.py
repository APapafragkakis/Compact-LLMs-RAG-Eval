# evaluate_metaqa.py - COMPLETE VERSION

import json
from pathlib import Path
from time import perf_counter
from remote_rag import run_rag


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_gold_answer(ans):
    if isinstance(ans, list) and ans:
        return ans[0].strip()
    if isinstance(ans, str):
        return ans.strip()
    return ""


def compare_prediction(pred: str, gold: str) -> bool:
    """
    Σύγκριση με normalize + exact set matching
    """
    if not pred or not gold:
        return pred == gold  # both empty = correct
    
    # Normalize
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    
    # Split by |
    pred_parts = set(p.strip() for p in pred.split("|") if p.strip())
    gold_parts = set(g.strip() for g in gold.split("|") if g.strip())
    
    # Must match exactly (all gold entities in prediction, no extra)
    return pred_parts == gold_parts


def evaluate(jsonl_path: str, output_path: str = "predictions_metaqa.jsonl", max_samples: int = None):
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)

    total = 0
    correct = 0
    t_start = perf_counter()

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            if max_samples and total >= max_samples:
                break
                
            qid = sample.get("id")
            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_rag(question)
            pred = res["answer"]

            ok = compare_prediction(pred, gold)
            total += 1
            if ok:
                correct += 1

            out_obj = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "gold_norm": gold,
                "prediction": pred,
                "retrieval_latency": res["retrieval_latency"],
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "raw_answer": res["raw_answer"],
                "raw_retrieval": res["raw_retrieval"],
                "correct": ok,
            }
            fout.write(json.dumps(out_obj) + "\n")

            # Progress
            if total % 10 == 0:
                acc = correct / total * 100
                print(f"[{total}] accuracy: {acc:.2f}% | last: {'✓' if ok else '✗'} {question[:50]}")

    t_end = perf_counter()
    acc = correct / total * 100 if total > 0 else 0.0

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"File: {jsonl_path}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Time: {t_end - t_start:.2f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    # Test με 50 samples
    evaluate("data/metaqa_1hop_only.jsonl", 
             output_path="predictions_metaqa.jsonl",
             max_samples=50)
    
    # Για full run, σχολίασε το max_samples:
    # evaluate("data/metaqa_1hop_only.jsonl", 
    #          output_path="predictions_metaqa.jsonl")