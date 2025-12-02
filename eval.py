import http.client
import json
import time
import os
from typing import Dict, Any, List, Set, Optional
from time import perf_counter

BASE_DIR = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation"

INPUT_PATH = os.path.join(BASE_DIR, "data", "metaqa_full.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "metaqa_llama3b_entity.jsonl")

MODEL_NAME = "llama3.2:3b"
HOST = "demos.isl.ics.forth.gr"
ENDPOINT = "/SemanticRAG/generate"

SLEEP_BETWEEN_CALLS = 0.7
MAX_RETRIES = 3
RETRY_SLEEP = 1.0
TIMEOUT = 60
MAX_QUESTIONS: Optional[int] = None


def build_user_prompt(question: str) -> str:
    return f"""Question:
{question}

Answer with ONLY the correct entity (one line, no punctuation, no explanations).""".strip()


def load_metaqa(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            q_id = obj.get("id", line_idx)
            q_text = (obj.get("question") or "").strip()
            gold_ans = obj.get("answer", [])
            src = obj.get("source_split", "unknown")
            if isinstance(gold_ans, str):
                gold_ans = [gold_ans]
            if not q_text:
                continue
            items.append({
                "id": q_id,
                "question": q_text,
                "gold_answer": gold_ans,
                "source_split": src,
            })
    return items


def append_result(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_done_ids(path: str) -> Set[Any]:
    done: Set[Any] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj.get("id"))
            except Exception:
                continue
    return done


def call_endpoint(question_text: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "conversation": [
            {
                "role": "system",
                "content": (
                    "You are a factual QA model. "
                    "Answer with the correct entity ONLY. "
                    "No explanations, no punctuation, no extra words. "
                    "Return a single line with the answer string."
                ),
            },
            {
                "role": "user",
                "content": build_user_prompt(question_text),
            },
        ],
    }

    body_json = json.dumps(payload)
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body_json))}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = http.client.HTTPSConnection(HOST, timeout=TIMEOUT)
            start_t = perf_counter()
            conn.request("POST", ENDPOINT, body_json, headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            end_t = perf_counter()
            conn.close()

            latency = end_t - start_t
            status = resp.status

            if status == 200:
                return {
                    "ok": True,
                    "status": status,
                    "response_text": raw,
                    "error": None,
                    "latency": latency,
                }
            else:
                time.sleep(RETRY_SLEEP)
        except Exception:
            time.sleep(RETRY_SLEEP)

    return {"ok": False, "status": None, "response_text": None, "error": "max retries exceeded", "latency": None}


def main():
    dataset = load_metaqa(INPUT_PATH)
    if MAX_QUESTIONS is not None:
        dataset = dataset[:MAX_QUESTIONS]
    total = len(dataset)
    print(f"[INFO] Loaded {total} questions from {INPUT_PATH}")

    done_ids = read_done_ids(OUTPUT_PATH)
    if done_ids:
        print(f"[INFO] Resume mode: {len(done_ids)} results already exist")

    processed = 0
    for idx, sample in enumerate(dataset, start=1):
        qid = sample["id"]
        if qid in done_ids:
            continue

        question = sample["question"]
        gold_ans = sample["gold_answer"]
        src = sample["source_split"]

        res = call_endpoint(question)

        record = {
            "id": qid,
            "question": question,
            "gold_answer": gold_ans,
            "source_split": src,
            "ok": res["ok"],
            "status": res["status"],
            "latency": res["latency"],
            "model_output_raw": res["response_text"],
            "error": res["error"],
            "model": MODEL_NAME,
        }

        append_result(OUTPUT_PATH, record)
        processed += 1

        if (processed % 50 == 0) or (not res["ok"]):
            print(f"[{processed}/{total}] id={qid} hop={src} ok={res['ok']} "
                  f"status={res['status']} latency={res['latency']}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("[DONE] Evaluation complete.")
    print(f"[SAVED] Results in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
