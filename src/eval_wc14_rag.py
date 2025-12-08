# eval_wc14_rag.py - EXACT Jordan Replication (WC-P1.txt format) + Timing

import http.client
import json
import ast
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
    
    # EXACT from Jordan's config.yaml
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
    """EXACT Jordan's post-processing from run.py line 27"""
    return llm_pred_str.replace("\n", " ")


def jordan_normalize_pred(pred_str: str) -> str:
    """EXACT Jordan's evaluation from evaluate.py line 18
    top1 = line_pred.rstrip().replace(" ", "_").lower().split(",")[0]
    """
    return pred_str.rstrip().replace(" ", "_").lower().split(",")[0]


def jordan_normalize_gold(gold_str: str) -> str:
    """EXACT Jordan's gold normalization from evaluate.py line 16"""
    return gold_str.lower()


def jordan_metrics(valid_answers: list, pred: str) -> dict:
    """EXACT Jordan's evaluation from evaluate.py
    
    valid_answers = [ans.lower() for ans in valid_answers]
    top1 = line_pred.rstrip().replace(" ", "_").lower().split(",")[0]
    if top1 in valid_answers: correct += 1
    """
    
    # Normalize valid answers
    valid_normalized = [jordan_normalize_gold(ans) for ans in valid_answers]
    
    # Normalize prediction
    pred_normalized = jordan_normalize_pred(pred)
    
    # Check if prediction is in ANY valid answer
    top1_match = pred_normalized in valid_normalized
    
    return {
        "top1_match": top1_match,
        "pred_normalized": pred_normalized,
        "valid_normalized": valid_normalized
    }


def run_rag(question: str):
    """Pure Jordan RAG pipeline"""
    # 1. Retrieval
    facts, t_retr, raw_retr = remote_retrieve(question)
    
    # 2. Build prompt
    user_msg = build_generation_prompt(question, facts)
    
    # 3. Generation
    raw_answer, t_gen = remote_generate(user_msg)
    
    # 4. Post-processing
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

def load_wc14_txt(path):
    """Parse Jordan's WC-P1.txt format
    
    Format: question?\tprimary_answer\tpath\tvalid_answers/\tother_data
    Example: "who plays for Mexico ?\tAlan_PULIDO\t...\tCarlos_SALCIDO/Alan_PULIDO/\t..."
    """
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Split by "?"
            parts = line.split("?")
            if len(parts) < 2:
                continue
            
            question = parts[0].strip()
            ansData = parts[1]
            
            # Extract valid answers from column 4 (tab-separated)
            tabs = ansData.strip().split("\t")
            if len(tabs) < 3:
                continue
            
            # Column 4 (index 2): "answer1/answer2/answer3/"
            # Jordan's evaluate.py line 15: valid_answers = ansData.strip().split("\t")[2][:-1].split("/")
            valid_answers_str = tabs[2][:-1] if tabs[2].endswith("/") else tabs[2]
            valid_answers = valid_answers_str.split("/")
            valid_answers = [ans.strip() for ans in valid_answers if ans.strip()]
            
            yield {
                "id": idx,
                "question": question,
                "valid_answers": valid_answers
            }


def evaluate(txt_path: str, output_path: str = None, resume: bool = True):
    """
    EXACT Jordan evaluation with WC-P1.txt format
    """
    txt_path = Path(txt_path)
    
    results_dir = Path("wc14_rag_results")
    results_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"wc2014_{model_name}_jordan.jsonl"
    else:
        output_path = Path(output_path)

    resume_mode = resume and output_path.exists()

    total = 0
    correct = 0
    
    processed_ids = set()

    # PURE TIMING ACCUMULATORS (δεν αλλάζουν το output format)
    llm_time_total = 0.0          # sum of generation_latency
    retrieval_time_total = 0.0    # sum of retrieval_latency
    endpoint_time_total = 0.0     # sum of (retrieval + generation)

    # Resume mode
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

                valid_prev = obj.get("valid_answers", [])
                pred_prev = obj.get("prediction", "")

                metrics_prev = jordan_metrics(valid_prev, pred_prev)

                total += 1
                if metrics_prev["top1_match"]:
                    correct += 1

                # Αν είχαν ήδη αποθηκευτεί latency fields, τα προσθέτουμε
                retrieval_time_total += obj.get("retrieval_latency", 0.0)
                llm_time_total += obj.get("generation_latency", 0.0)
                endpoint_time_total += obj.get("total_latency", 0.0)

        print(f"[RESUME] Found {len(processed_ids)} existing samples. Will skip these.\n")

    t_start = perf_counter()

    print(f"\n{'='*70}")
    print(f"EXACT JORDAN REPLICATION: {GENERATION_MODEL}")
    print(f"{'='*70}")
    print("Dataset: WC-P1.txt (Jordan's format)")
    print("Evaluation: EXACT Jordan (evaluate.py)")
    print("Multiple valid answers per question supported")
    if resume_mode:
        print(f"Mode: RESUME")
    else:
        print(f"Mode: FRESH RUN")
    print(f"{'='*70}\n")

    mode = "a" if resume_mode else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in load_wc14_txt(txt_path):
            qid = sample["id"]

            if resume_mode and qid in processed_ids:
                continue

            question = sample["question"]
            valid_answers = sample["valid_answers"]

            res = run_rag(question)
            
            # Jordan's metrics
            metrics = jordan_metrics(valid_answers, res["processed_answer"])
            
            total += 1
            if metrics["top1_match"]:
                correct += 1

            # PURE TIMING ACCUMULATION (μόνο in-memory)
            retrieval_time_total += res["retrieval_latency"]
            llm_time_total += res["generation_latency"]
            endpoint_time_total += res["total_latency"]

            out_obj = {
                "id": qid,
                "question": question,
                "valid_answers": valid_answers,
                "prediction": res["processed_answer"],
                "raw_answer": res["raw_answer"],
                "retrieval_latency": res["retrieval_latency"],
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "top1_match": metrics["top1_match"],
                "pred_normalized": metrics["pred_normalized"],
                "valid_normalized": metrics["valid_normalized"]
            }
            fout.write(json.dumps(out_obj) + "\n")
            fout.flush()

            if total % 10 == 0:
                acc = correct / total * 100
                print(f"[{total}] Accuracy: {acc:.2f}% ({correct}/{total}) | {'✓' if metrics['top1_match'] else '✗'}")

            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()
    
    # Final metrics
    accuracy = correct / total if total else 0.0
    wall_time = t_end - t_start

    print(f"\n{'='*70}")
    print("EXACT JORDAN REPLICATION - RESULTS")
    print(f"{'='*70}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {txt_path}")
    print(f"Total samples: {total}")
    print(f"\nFinal Score:")
    print(f"hits@1 (accuracy): {accuracy:.4f}  ({correct}/{total})")
    print(f"\nWall-clock Time (with sleep/I-O): {wall_time:.2f}s ({wall_time/60:.1f} min)")
    print(f"Results: {output_path}")

    # PURE TIMING SUMMARY (χωρίς sleep / Python overhead)
    if total > 0:
        print("\n--- PURE ENDPOINT TIMING (no sleep, no Python overhead) ---")
        print(f"  Retrieval total time:        {retrieval_time_total:.2f}s ({retrieval_time_total/60:.2f} min)")
        print(f"  LLM total generation time:   {llm_time_total:.2f}s ({llm_time_total/60:.2f} min)")
        print(f"  Endpoint total (retr+gen):   {endpoint_time_total:.2f}s ({endpoint_time_total/60:.2f} min)")
        print(f"  Avg retrieval latency:       {retrieval_time_total/total:.2f}s")
        print(f"  Avg LLM latency:             {llm_time_total/total:.2f}s")
        print(f"  Avg endpoint latency:        {endpoint_time_total/total:.2f}s")
    print(f"{'='*70}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time": wall_time,  # wall-clock with sleeps

        # extra fields για να τα βλέπεις στο run_wc14_rag.py αν θέλεις
        "llm_time_total": llm_time_total,
        "retrieval_time_total": retrieval_time_total,
        "endpoint_time_total": endpoint_time_total,
        "llm_avg_latency": llm_time_total / total if total else 0.0,
        "retrieval_avg_latency": retrieval_time_total / total if total else 0.0,
        "endpoint_avg_latency": endpoint_time_total / total if total else 0.0,
    }


if __name__ == "__main__":
    evaluate("data/WC-P1.txt")
