# eval_wc14_baseline.py - WC14 Baseline (NO RAG) Evaluation

import http.client
import json
from time import perf_counter, sleep
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

HOST = "demos.isl.ics.forth.gr"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

GENERATION_MODEL = "llama3.1:8b"
SLEEP_BETWEEN_REQUESTS = 0.7


# ============================================================================
# BASELINE FUNCTIONS (NO RAG)
# ============================================================================

def build_baseline_prompt(question: str) -> str:
    """Simple baseline prompt - NO retrieval, NO documents"""
    return "Question: " + question + "\nAnswer:"


def remote_generate(user_text: str):
    """Same generation as RAG but without retrieval context"""
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
    """EXACT Jordan's evaluation from evaluate.py"""
    
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


def run_baseline(question: str):
    """Pure baseline pipeline - NO RAG"""
    # 1. Build simple prompt (NO retrieval)
    user_msg = build_baseline_prompt(question)
    
    # 2. Generation only
    raw_answer, t_gen = remote_generate(user_msg)
    
    # 3. Post-processing
    processed_answer = jordan_postprocess(raw_answer)

    return {
        "question": question,
        "raw_answer": raw_answer,
        "processed_answer": processed_answer,
        "generation_latency": t_gen,
        "total_latency": t_gen,
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
    WC14 Baseline (NO RAG) evaluation with WC-P1.txt format
    """
    txt_path = Path(txt_path)
    
    results_dir = Path("baseline_results")
    results_dir.mkdir(exist_ok=True)
    
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"wc2014_{model_name}_baseline_no_rag.jsonl"
    else:
        output_path = Path(output_path)

    resume_mode = resume and output_path.exists()

    total = 0
    correct = 0
    
    processed_ids = set()

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

        print(f"[RESUME] Found {len(processed_ids)} existing samples. Will skip these.\n")

    t_start = perf_counter()

    print(f"\n{'='*70}")
    print(f"WC14 BASELINE (NO RAG): {GENERATION_MODEL}")
    print(f"{'='*70}")
    print("Dataset: WC-P1.txt")
    print("Method: Baseline (NO retrieval)")
    print("Evaluation: EXACT Jordan (evaluate.py)")
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

            res = run_baseline(question)
            
            # Jordan's metrics
            metrics = jordan_metrics(valid_answers, res["processed_answer"])
            
            total += 1
            if metrics["top1_match"]:
                correct += 1

            out_obj = {
                "id": qid,
                "question": question,
                "valid_answers": valid_answers,
                "prediction": res["processed_answer"],
                "raw_answer": res["raw_answer"],
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

    print(f"\n{'='*70}")
    print("WC14 BASELINE (NO RAG) - RESULTS")
    print(f"{'='*70}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {txt_path}")
    print(f"Total samples: {total}")
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
    evaluate("data/WC-P1.txt")