# eval_metaqa_rag.py - Pure Jordan Replication with Auto-Resume

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
# RAG FUNCTIONS - PURE JORDAN
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
    
    # EXACT from Jordan's config.yaml
    rag_prompt = (
        "Your task is to extract the answer from the documents. "
        "There is always relevant information present.\n\n"
        "Find the entity mentioned in the question, then extract the specific "
        "information requested about that entity from the documents.\n\n"
        "Answer with only the exact word, name, or date needed. "
        "Use only text that appears in the documents.\n\n"
        "Never respond with 'None' - always extract something relevant from the text.\n\n"
        "If multiple answers are present, separate with | (e.g., X | Y | Z)"
    )
    
    return (
        rag_prompt + "\n\n"
        "Documents:\n" + facts_text + "\n\n"
        "Question: " + question + "\n"
        "Answer:"
    )


def remote_generate(user_text: str):
    """EXACT Jordan's system prompt from config.yaml"""
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
    """EXACT Jordan's post-processing from run.py - NOTHING ELSE"""
    
    # 1. rstrip
    llm_pred = llm_pred_str.rstrip()
    
    # 2. replace newlines with space
    llm_pred = llm_pred.replace("\n", " ")
    
    # 3. split by |
    llm_pred_list = llm_pred.split("|")
    
    # 4. strip each answer
    llm_pred_list = [answer.strip() for answer in llm_pred_list]
    
    # 5. remove duplicates (preserving order)
    llm_pred_list = list(dict.fromkeys(llm_pred_list))
    
    # Join back with |
    return "|".join(llm_pred_list)


def jordan_normalize(s: str) -> str:
    """EXACT Jordan's normalization from evaluate.py"""
    s = s.strip().lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s


def jordan_to_set(items_str: str):
    """EXACT Jordan's to_set from evaluate.py"""
    if not items_str:
        return set()
    items = items_str.split("|")
    return {jordan_normalize(x) for x in items if x.strip()}


def jordan_metrics(gold: str, pred: str) -> dict:
    """EXACT Jordan's metrics from evaluate.py"""
    
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


def run_rag(question: str):
    """Pure Jordan RAG pipeline"""
    # 1. Retrieval
    facts, t_retr, raw_retr = remote_retrieve(question)
    
    # 2. Build prompt (NO filtering - give all facts to LLM)
    user_msg = build_generation_prompt(question, facts)
    
    # 3. Generation
    raw_answer, t_gen = remote_generate(user_msg)
    
    # 4. Jordan's post-processing ONLY
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
    """Handle MetaQA format"""
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
    
    results_dir = Path("rag_results")
    results_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"wc2014_{model_name}_pure_jordan.jsonl"

    else:
        output_path = Path(output_path)

    # Αν υπάρχει αρχείο και θέλουμε resume, θα το χρησιμοποιήσουμε
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

    print(f"\n{'='*70}")
    print(f"PURE JORDAN REPLICATION: {GENERATION_MODEL}")
    print(f"{'='*70}")
    print("Prompts: EXACT Jordan (config.yaml)")
    print("Post-processing: EXACT Jordan (run.py)")
    print("Metrics: EXACT Jordan (evaluate.py)")
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
            
            # Jordan's metrics
            metrics = jordan_metrics(gold, res["processed_answer"])
            
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
                "exact_match": metrics["exact_match"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            fout.write(json.dumps(out_obj) + "\n")
            fout.flush()

            if total % 10 == 0:
                acc = top1_correct / total * 100
                em = exact_matches / total * 100
                avg_f1 = sum(macro_f1s) / len(macro_f1s)
                print(f"[{total}] Acc: {acc:.2f}% | EM: {em:.2f}% | F1: {avg_f1:.3f} | {'✓' if metrics['top1_match'] else '✗'}")

            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()
    
    # Final metrics (Jordan's format)
    accuracy = top1_correct / total if total else 0.0
    exact_match_rate = exact_matches / total if total else 0.0
    
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    macro_p = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_r = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    
    micro_p = total_tp / total_pred if total_pred else 0.0
    micro_r = total_tp / total_gold if total_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    print(f"\n{'='*70}")
    print("PURE JORDAN REPLICATION - RESULTS")
    print(f"{'='*70}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples (including resumed): {total}")
    print(f"\nFinal Scores:")
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
    print(f"{'='*70}\n")
    
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


if __name__ == "__main__":
    evaluate("data/metaqa_1hop_only.jsonl")
