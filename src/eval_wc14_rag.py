# eval_rag.py - EXACT Jordan Replication with Auto-Resume

import http.client
import json
import ast
import string
import re
from time import perf_counter, sleep
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

RETRIEVAL_MODEL = "wc2014qa"
GENERATION_MODEL = "llama3.1:8b"

SLEEP_BETWEEN_REQUESTS = 0.7


# ============================================================================
# RAG FUNCTIONS - EXACT JORDAN
# ============================================================================

def remote_retrieve(question: str):
    """Retrieval - returns top facts"""
    payload = {"model": RETRIEVAL_MODEL, "prompt": question}
    body = json.dumps(payload)
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}

    conn = http.client.HTTPSConnection(HOST, timeout=60)
    t0 = perf_counter()
    conn.request("POST", RETRIEVE_ENDPOINT, body, headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    t1 = perf_counter()
    conn.close()

    latency = t1 - t0

    try:
        lst = ast.literal_eval(raw)
    except Exception:
        lst = []

    facts = []
    for s in lst:
        if isinstance(s, str) and s.startswith("-: "):
            s = s[3:]
        facts.append(s)

    return facts, latency, raw


def build_generation_prompt(question: str, facts: list[str]) -> str:
    """EXACT Jordan's RAG prompt from config.yaml"""
    if not facts:
        return "Question: " + question + "\nAnswer:"

    facts_text = "\n".join(f"- {f}" for f in facts)
    
    # EXACT from Jordan's config.yaml - lines 13-18
    rag_prompt = (
        "Your task is to extract the answer from the documents. "
        "There is always relevant information present. "
        "Find the entity mentioned in the question, then extract the specific "
        "information requested about that entity from the documents. "
        "Answer with only the exact word or name needed. "
        "Use only text that appears in the documents. "
        "Never respond with \"None\" - always extract something relevant from the text. "
        "If there are multiple answers, only provide one - the most fitting."
    )
    
    return (
        rag_prompt + "\n\n"
        "Documents:\n" + facts_text + "\n\n"
        "Question: " + question + "\n"
        "Answer:"
    )


def remote_generate(user_text: str):
    """EXACT Jordan's system prompt from config.yaml line 10"""
    system_message = "Respond to all questions directly without any explanation"

    payload = {
        "model": GENERATION_MODEL,
        "conversation": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
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


def jordan_postprocess(llm_pred_str: str) -> str:
    """EXACT Jordan's post-processing from run.py line 27
    ONLY: pred.replace("\\n", " ")
    """
    return llm_pred_str.replace("\n", " ")


def jordan_normalize_pred(pred_str: str) -> str:
    """EXACT Jordan's evaluation normalization from evaluate.py line 18
    top1 = line_pred.rstrip().replace(" ", "_").lower().split(",")[0]
    """
    return pred_str.rstrip().replace(" ", "_").lower().split(",")[0]


def jordan_normalize_gold(gold_str: str) -> str:
    """EXACT Jordan's gold normalization from evaluate.py line 16
    valid_answers = [ans.lower() for ans in valid_answers]
    """
    return gold_str.lower()


def jordan_metrics(gold: str, pred: str) -> dict:
    """EXACT Jordan's metrics from evaluate.py
    
    Comparison logic:
    - Normalize prediction: rstrip().replace(" ", "_").lower().split(",")[0]
    - Normalize gold: lower()
    - Check: if normalized_pred in normalized_gold_list
    """
    
    # For single gold answer
    pred_normalized = jordan_normalize_pred(pred)
    gold_normalized = jordan_normalize_gold(gold)
    
    # Simple exact match
    top1_match = (pred_normalized == gold_normalized)
    
    return {
        "top1_match": top1_match,
        "pred_normalized": pred_normalized,
        "gold_normalized": gold_normalized
    }


def run_rag(question: str):
    """Pure Jordan RAG pipeline"""
    # 1. Retrieval
    facts, t_retr, raw_retr = remote_retrieve(question)
    
    # 2. Build prompt (NO filtering - give all facts to LLM)
    user_msg = build_generation_prompt(question, facts)
    
    # 3. Generation
    raw_answer, t_gen = remote_generate(user_msg)
    
    # 4. Jordan's post-processing ONLY (run.py line 27)
    processed_answer = jordan_postprocess(raw_answer)
    
    total_latency = t_retr + t_gen

    return {
        "question": question,
        "facts": facts,
        "raw_answer": raw_answer,
        "processed_answer": processed_answer,
        "retrieval_latency": t_retr,
        "generation_latency": t_gen,
        "total_latency": total_latency,
        "raw_retrieval": raw_retr,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_gold_answer(ans):
    """Handle MetaQA format - get first answer if list"""
    if isinstance(ans, list) and ans:
        return ans[0].strip()
    if isinstance(ans, str):
        return ans.strip()
    return ""


def evaluate(jsonl_path: str, output_path: str = None, resume: bool = True):
    """
    Pure Jordan evaluation, with optional auto-resume.
    
    - Αν resume=True και υπάρχει ήδη output αρχείο:
      * Διαβάζει τα ήδη αποθηκευμένα αποτελέσματα
      * Ξαναϋπολογίζει τα metrics για αυτά
      * Συνεχίζει μόνο για τα δείγματα (id) που λείπουν.
    - Αν resume=False ή δεν υπάρχει αρχείο:
      * Ξεκινάει από την αρχή και κάνει overwrite.
    """
    jsonl_path = Path(jsonl_path)
    
    results_dir = Path("wc14_rag_results")
    results_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"wc2014_{model_name}_pure_jordan.jsonl"
    else:
        output_path = Path(output_path)

    # Αν υπάρχει αρχείο και θέλουμε resume, θα το χρησιμοποιήσουμε
    resume_mode = resume and output_path.exists()

    total = 0
    correct = 0
    
    processed_ids = set()

    # ===========================
    # 1) Αν resume_mode, φόρτωσε ήδη υπάρχοντα αποτελέσματα
    # ===========================
    if resume_mode:
        print(f"\n[RESUME] Loading existing results from {output_path}")
        with open(output_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid_prev = obj.get("id")
                processed_ids.add(qid_prev)

                gold_prev_raw = obj.get("gold_answer")
                gold_prev = normalize_gold_answer(gold_prev_raw)
                pred_prev = obj.get("prediction", "")

                # Recalculate metrics
                metrics_prev = jordan_metrics(gold_prev, pred_prev)

                total += 1
                if metrics_prev["top1_match"]:
                    correct += 1

        print(f"[RESUME] Found {len(processed_ids)} existing samples. Will skip these.\n")

    t_start = perf_counter()

    print(f"\n{'='*70}")
    print(f"EXACT JORDAN REPLICATION: {GENERATION_MODEL}")
    print(f"{'='*70}")
    print("Prompts: EXACT Jordan (config.yaml)")
    print("Post-processing: EXACT Jordan (run.py line 27)")
    print("Metrics: EXACT Jordan (evaluate.py line 18)")
    print("NO modifications, NO tricks, NO filtering")
    if resume_mode:
        print(f"Mode: RESUME (append remaining samples to {output_path})")
    else:
        print(f"Mode: FRESH RUN (overwrite {output_path})")
    print(f"{'='*70}\n")

    # Αν κάνουμε resume, ανοίγουμε σε append. Αλλιώς overwrite.
    mode = "a" if resume_mode else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            qid = sample.get("id")

            # Αν έχουμε ήδη αυτό το id σε προηγούμενο run, κάνε skip
            if resume_mode and qid in processed_ids:
                continue

            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_rag(question)
            
            # Jordan's metrics (evaluate.py)
            metrics = jordan_metrics(gold, res["processed_answer"])
            
            total += 1
            if metrics["top1_match"]:
                correct += 1

            out_obj = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "prediction": res["processed_answer"],
                "raw_answer": res["raw_answer"],
                "retrieval_latency": res["retrieval_latency"],
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "raw_retrieval": res["raw_retrieval"],
                "top1_match": metrics["top1_match"],
                "pred_normalized": metrics["pred_normalized"],
                "gold_normalized": metrics["gold_normalized"]
            }
            fout.write(json.dumps(out_obj) + "\n")
            fout.flush()

            if total % 10 == 0:
                acc = correct / total * 100
                print(f"[{total}] Accuracy: {acc:.2f}% ({correct}/{total}) | {'✓' if metrics['top1_match'] else '✗'}")

            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()
    
    # Final metrics (Jordan's hits@1)
    accuracy = correct / total if total else 0.0

    print(f"\n{'='*70}")
    print("EXACT JORDAN REPLICATION - RESULTS")
    print(f"{'='*70}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples (including resumed): {total}")
    print(f"\nFinal Score:")
    print(f"hits@1 (accuracy): {accuracy:.4f}  ({correct}/{total})")
    print(f"\nTime: {t_end - t_start:.2f}s ({(t_end - t_start)/60:.1f} min)")
    print(f"Results: {output_path}")
    print(f"{'='*70}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time": t_end - t_start
    }


if __name__ == "__main__":
    evaluate("data/wc2014qa.jsonl")