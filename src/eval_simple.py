# eval_metaqa_simple.py - No-RAG Baseline Evaluation for MetaQA
# Schema & metrics aligned with eval_metaqa_rag.py, with auto-resume per sample

import http.client
import json
from time import perf_counter, sleep
from pathlib import Path
import re
import string

# ====================================================================
# CONFIGURATION
# ====================================================================

HOST = "demos.isl.ics.forth.gr"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

GENERATION_MODEL = "llama3.1:8b"  # θα το αλλάζει το runner
SLEEP_BETWEEN_REQUESTS = 0.7  # seconds - prevent endpoint overload


# ====================================================================
# BASELINE GENERATION (NO RAG)
# ====================================================================

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
    Normalize LLM output (baseline-side), αλλά η αξιολόγηση είναι Jordan-style
    όπως και στο RAG.
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


# ====================================================================
# SHARED HELPERS (ίδια λογική με eval_metaqa_rag)
# ====================================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_gold_answer(ans):
    """Handle MetaQA format (ίδιο με RAG)"""
    if isinstance(ans, list) and ans:
        return ans[0].strip()
    if isinstance(ans, str):
        return ans.strip()
    return ""


# ===== Jordan-style normalization & metrics (ίδια μορφή με eval_metaqa_rag) =====

def jordan_normalize(s: str) -> str:
    """EXACT Jordan-style normalization"""
    s = s.strip().lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s


def jordan_to_set(items_str: str):
    """Jordan's to_set"""
    if not items_str:
        return set()
    items = items_str.split("|")
    return {jordan_normalize(x) for x in items if x.strip()}


def jordan_metrics(gold: str, pred: str) -> dict:
    """
    Jordan-style metrics (ίδιο format με RAG):
    - top1_match
    - exact_match
    - precision, recall, f1
    - counts
    """
    gset = jordan_to_set(gold)
    pset = jordan_to_set(pred)
    
    # Top-1 accuracy
    pred_parts = [p.strip() for p in pred.split("|") if p.strip()]
    top_pred = jordan_normalize(pred_parts[0]) if pred_parts else None
    top1_match = top_pred in gset if top_pred else False
    
    # Exact Match
    exact_match = gset == pset
    
    # Precision, Recall, F1
    tp = len(gset & pset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    return {
        "top1_match": top1_match,
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "pred_count": len(pset),
        "gold_count": len(gset)
    }


# ====================================================================
# EVALUATION (με auto-resume όπως eval_metaqa_rag)
# ====================================================================

def evaluate(jsonl_path: str, output_path: str = None, resume: bool = True):
    """
    Baseline (NO-RAG) evaluation με:
    - Ίδιο JSON schema ανά sample με το RAG (id, question, gold_answer, prediction,
      raw_answer, retrieval_latency, generation_latency, total_latency, raw_retrieval,
      top1_match, exact_match, precision, recall, f1)
    - Ίδιο aggregate metrics dict (accuracy, exact_match, macro/micro κτλ)
    - Auto-resume: αν υπάρχει ήδη αρχείο, διαβάζει τα υπάρχοντα samples,
      ξαναϋπολογίζει metrics και συνεχίζει μόνο για όσα ids λείπουν.
    """
    jsonl_path = Path(jsonl_path)
    
    # Αποθήκευση σε ξεχωριστό φάκελο για baseline
    results_dir = Path("baseline_results")
    results_dir.mkdir(exist_ok=True)
    
    # Auto-generate output filename if not provided
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"metaqa_{model_name}_baseline_no_rag.jsonl"
    else:
        output_path = Path(output_path)

    # Resume mode όπως στο eval_metaqa_rag
    resume_mode = resume and output_path.exists()

    total = 0
    top1_correct = 0
    exact_matches = 0
    
    macro_f1s = []
    macro_precisions = []
    macro_recalls = []
    
    total_tp = 0
    total_pred = 0
    total_gold = 0

    processed_ids = set()

    # ------------------------------------------------------------
    # 1) Αν resume_mode, φόρτωσε ήδη υπάρχοντα αποτελέσματα
    # ------------------------------------------------------------
    if resume_mode:
        print(f"\n[RESUME] Loading existing baseline results from {output_path}")
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

                metrics_prev = jordan_metrics(gold_prev, pred_prev)

                total += 1
                if metrics_prev["top1_match"]:
                    top1_correct += 1
                if metrics_prev["exact_match"]:
                    exact_matches += 1

                macro_f1s.append(metrics_prev["f1"])
                macro_precisions.append(metrics_prev["precision"])
                macro_recalls.append(metrics_prev["recall"])

                total_tp += metrics_prev["tp"]
                total_pred += metrics_prev["pred_count"]
                total_gold += metrics_prev["gold_count"]

        print(f"[RESUME] Found {len(processed_ids)} existing samples. Will skip these.\n")

    t_start = perf_counter()

    print("\n" + "="*70)
    print(f"BASELINE (NO-RAG) EVALUATION: {GENERATION_MODEL}")
    print("="*70)
    print("Metrics format: Jordan-style (same as RAG)")
    print("JSON schema: matched to eval_metaqa_rag output")
    print(f"Dataset: {jsonl_path}")
    print(f"Results file: {output_path}")
    if resume_mode:
        print("Mode: RESUME (append remaining samples)")
    else:
        print("Mode: FRESH RUN (overwrite existing results)")
    print("="*70 + "\n")

    mode = "a" if resume_mode else "w"

    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            qid = sample.get("id")

            # Αν έχει ήδη γίνει σε προηγούμενο run, skip
            if resume_mode and qid in processed_ids:
                continue

            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_baseline(question)
            pred = res["answer"]

            # Jordan-style metrics
            metrics = jordan_metrics(gold, pred)

            total += 1
            if metrics["top1_match"]:
                top1_correct += 1
            if metrics["exact_match"]:
                exact_matches += 1
            
            macro_f1s.append(metrics["f1"])
            macro_precisions.append(metrics["precision"])
            macro_recalls.append(metrics["recall"])
            
            total_tp += metrics["tp"]
            total_pred += metrics["pred_count"]
            total_gold += metrics["gold_count"]

            # ===== per-sample JSON object - ΙΔΙΑ ΜΟΡΦΗ ΜΕ RAG =====
            out_obj = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "prediction": pred,
                "raw_answer": res["raw_answer"],
                "retrieval_latency": 0.0,               # baseline: no retrieval
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "raw_retrieval": None,                  # baseline: no retrieval payload
                "top1_match": metrics["top1_match"],
                "exact_match": metrics["exact_match"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            fout.write(json.dumps(out_obj) + "\n")
            fout.flush()

            if total % 10 == 0:
                acc = top1_correct / total * 100 if total else 0.0
                em = exact_matches / total * 100 if total else 0.0
                avg_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
                print(
                    f"[{total}] Acc: {acc:.2f}% | EM: {em:.2f}% | "
                    f"F1: {avg_f1:.3f} | {'✓' if metrics['top1_match'] else '✗'} "
                    f"{question[:50]}"
                )

            # Sleep to prevent endpoint overload
            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()
    
    # ===== Final Jordan-style aggregate metrics (όπως στο RAG) =====
    accuracy = top1_correct / total if total else 0.0
    exact_match_rate = exact_matches / total if total else 0.0
    
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    macro_p = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_r = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    
    micro_p = total_tp / total_pred if total_pred else 0.0
    micro_r = total_tp / total_gold if total_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    print("\n" + "="*70)
    print("BASELINE (NO-RAG) EVALUATION COMPLETE")
    print("="*70)
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples (including resumed): {total}")
    print("\nFinal Scores:")
    print(f"Accuracy (top-1 match):       {accuracy:.4f}")
    print(f"Exact Match:                  {exact_match_rate:.4f}")
    print(f"Macro Precision:              {macro_p:.4f}")
    print(f"Macro Recall:                 {macro_r:.4f}")
    print(f"Macro F1:                     {macro_f1:.4f}")
    print(f"Micro Precision:              {micro_p:.4f}")
    print(f"Micro Recall:                 {micro_r:.4f}")
    print(f"Micro F1:                     {micro_f1:.4f}")
    print(f"\nTime: {t_end - t_start:.2f}s ({(t_end - t_start)/60:.1f} min)")
    print(f"Results: {output_path}")
    print("="*70 + "\n")
    
    return {
        "accuracy": accuracy,
        "exact_match": exact_match_rate,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "total": total,
        "time": t_end - t_start
    }


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    # Run baseline evaluation on MetaQA 1-hop (με auto-resume στον φάκελο baseline_results)
    evaluate("data/metaqa_1hop_only.jsonl", resume=True)
