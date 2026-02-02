import http.client
import json
from time import perf_counter, sleep
from pathlib import Path

HOST = "demos.isl.ics.forth.gr"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

GENERATION_MODEL = "llama3.1:8b"
SLEEP_BETWEEN_REQUESTS = 0.7


def build_baseline_prompt(question: str) -> str:
    return "Question: " + question + "\nAnswer:"


def remote_generate(user_text: str):
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

    return raw.strip(), (t1 - t0)


def postprocess_answer(llm_pred_str: str) -> str:
    return llm_pred_str.replace("\n", " ")


def normalize_pred(pred_str: str) -> str:
    return pred_str.rstrip().replace(" ", "_").lower().split(",")[0]


def normalize_gold(gold_str: str) -> str:
    return gold_str.lower()


def compute_metrics(valid_answers: list, pred: str) -> dict:
    valid_normalized = [normalize_gold(ans) for ans in valid_answers]
    pred_normalized = normalize_pred(pred)
    top1_match = pred_normalized in valid_normalized
    return {
        "top1_match": top1_match,
        "pred_normalized": pred_normalized,
        "valid_normalized": valid_normalized,
    }


def run_baseline(question: str):
    user_msg = build_baseline_prompt(question)
    raw_answer, t_gen = remote_generate(user_msg)
    processed_answer = postprocess_answer(raw_answer)
    return {
        "question": question,
        "raw_answer": raw_answer,
        "processed_answer": processed_answer,
        "generation_latency": t_gen,
        "total_latency": t_gen,
    }


def load_wc14_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split("?")
            if len(parts) < 2:
                continue

            question = parts[0].strip()
            ans_data = parts[1]

            tabs = ans_data.strip().split("\t")
            if len(tabs) < 3:
                continue

            valid_answers_str = tabs[2][:-1] if tabs[2].endswith("/") else tabs[2]
            valid_answers = [a.strip() for a in valid_answers_str.split("/") if a.strip()]

            yield {"id": idx, "question": question, "valid_answers": valid_answers}


def evaluate(txt_path: str, output_path: str = None, resume: bool = True):
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

    if resume_mode:
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

                metrics_prev = compute_metrics(valid_prev, pred_prev)

                total += 1
                if metrics_prev["top1_match"]:
                    correct += 1

    t_start = perf_counter()

    mode = "a" if resume_mode else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in load_wc14_txt(txt_path):
            qid = sample["id"]
            if resume_mode and qid in processed_ids:
                continue

            question = sample["question"]
            valid_answers = sample["valid_answers"]

            res = run_baseline(question)
            metrics = compute_metrics(valid_answers, res["processed_answer"])

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
                "valid_normalized": metrics["valid_normalized"],
            }
            fout.write(json.dumps(out_obj) + "\n")
            fout.flush()

            if total % 10 == 0:
                acc = correct / total * 100 if total else 0.0
                print(f"[{total}] Accuracy: {acc:.2f}% ({correct}/{total}) | {'✓' if metrics['top1_match'] else '✗'}")

            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()

    accuracy = correct / total if total else 0.0

    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {txt_path}")
    print(f"Total samples: {total}")
    print(f"hits@1 (accuracy): {accuracy:.4f} ({correct}/{total})")
    print(f"Time: {t_end - t_start:.2f}s ({(t_end - t_start)/60:.1f} min)")
    print(f"Results: {output_path}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time": t_end - t_start,
    }


if __name__ == "__main__":
    evaluate("data/WC-P1.txt")
