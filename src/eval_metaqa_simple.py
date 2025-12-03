# evaluate_baseline.py - No-RAG Baseline Evaluation for MetaQA

import http.client
import json
from time import perf_counter
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

HOST = "demos.isl.ics.forth.gr"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

GENERATION_MODEL = "llama3.1:8b"  # Change this to test different models


# ============================================================================
# BASELINE GENERATION (NO RAG)
# ============================================================================

def generate_baseline(question: str):
    """
    Direct LLM generation without retrieval - baseline comparison
    """
    system_message = (
        "You are a movie knowledge expert. Answer questions about movies, actors, and directors.\n\n"
        "Rules:\n"
        "1. Use ONLY your internal knowledge\n"
        "2. Return ONLY entity names (movie titles, actor names, etc.)\n"
        "3. Multiple answers: separate with | (e.g. 'Movie1|Movie2')\n"
        "4. DO NOT include:\n"
        "   - Explanations or reasoning\n"
        "   - Extra punctuation\n"
        "   - Phrases like 'The answer is...'\n\n"
        "Example:\n"
        "Question: what does [Tom Hanks] appear in\n"
        "Answer: Forrest Gump|Cast Away|Saving Private Ryan\n"
    )
    
    user_message = (
        f"Question: {question}\n\n"
        f"Answer with entity names only (separate multiple with |). No explanations."
    )

    payload = {
        "model": GENERATION_MODEL,
        "conversation": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }

    body = json.dumps(payload)
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    conn = http.client.HTTPSConnection(HOST, timeout=60)
    t0 = perf_counter()
    conn.request("POST", GENERATE_ENDPOINT, body, headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    t1 = perf_counter()
    conn.close()

    latency = t1 - t0
    return raw.strip(), latency


def normalize_baseline_answer(raw_answer: str, question: str) -> str:
    """
    Normalize LLM output - same logic as RAG version for fair comparison
    """
    if not raw_answer.strip():
        return ""
    
    # Extract question entity (in square brackets) - don't include in answer
    question_entity = None
    match = re.search(r'\[([^\]]+)\]', question)
    if match:
        question_entity = match.group(1).strip().lower()
    
    # Parse raw answer (handle | and newlines)
    raw = raw_answer.strip()
    tmp = raw.replace("|", "\n")
    parts = [p.strip() for p in tmp.splitlines() if p.strip()]
    
    # Filter out question entity
    filtered = []
    for p in parts:
        p_lower = p.lower()
        
        # Skip if it's the question entity
        if question_entity and p_lower == question_entity:
            continue
        
        filtered.append(p)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for e in filtered:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "|".join(unique)


def run_baseline(question: str):
    """
    Run baseline evaluation without RAG retrieval
    """
    raw_answer, latency = generate_baseline(question)
    final_answer = normalize_baseline_answer(raw_answer, question)

    return {
        "question": question,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "generation_latency": latency,
        "total_latency": latency,  # No retrieval, so total = generation
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

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
    if not pred or not gold:
        return pred == gold
    
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    
    pred_parts = set(p.strip() for p in pred.split("|") if p.strip())
    gold_parts = set(g.strip() for g in gold.split("|") if g.strip())
    
    return pred_parts == gold_parts


def evaluate(jsonl_path: str, output_path: str = None):
    jsonl_path = Path(jsonl_path)
    
    # Create results directory for baseline (no-RAG) results
    results_dir = Path("simple_results")
    results_dir.mkdir(exist_ok=True)
    
    # Auto-generate output filename if not provided
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"metaqa_{model_name}.jsonl"
    else:
        output_path = Path(output_path)

    total = 0
    correct = 0
    t_start = perf_counter()

    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION (NO RAG): {GENERATION_MODEL}")
    print(f"{'='*60}\n")

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            qid = sample.get("id")
            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_baseline(question)
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
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "raw_answer": res["raw_answer"],
                "correct": ok,
                "method": "baseline_no_rag",
            }
            fout.write(json.dumps(out_obj) + "\n")

            if total % 10 == 0:
                acc = correct / total * 100
                print(f"[{total}] accuracy: {acc:.2f}% | last: {'✓' if ok else '✗'} {question[:50]}")

    t_end = perf_counter()
    acc = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print("BASELINE EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Method: Direct LLM (NO RAG)")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Total time: {t_end - t_start:.2f}s")
    print(f"Avg latency: {(t_end - t_start) / total:.2f}s per sample")
    print(f"Results saved: {output_path}")
    print(f"{'='*60}\n")
    
    return {"accuracy": acc, "correct": correct, "total": total, "time": t_end - t_start}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run baseline evaluation on MetaQA 1-hop
    evaluate("data/metaqa_1hop_only.jsonl")
# ============================================================================