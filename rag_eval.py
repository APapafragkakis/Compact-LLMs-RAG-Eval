import os
import sys
import json
import time
import argparse
from time import perf_counter
from typing import Dict, Any, List, Set, Optional

# === ΣΗΜΑΝΤΙΚΟ ===
# Βάλε εδώ το path προς τον φάκελο "src" που περιέχει το KG_RAG.
# Παράδειγμα:
#   C:\\Users\\alexp\\OneDrive\\Υπολογιστής\\KG_RAG\\src
KG_RAG_SRC_PATH = r"C:\Users\alexp\OneDrive\Υπολογιστής\KG-RAG_Comparison_System\KG-RAG-master\src"

if KG_RAG_SRC_PATH not in sys.path:
    sys.path.append(KG_RAG_SRC_PATH)

from KG_RAG.pipeline import RAGAgent  # type: ignore


# ===== Paths μέσα στο RAG-LLM-BENCHMARKING repo =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "metaqa_1hop_only.jsonl")

# Νέο αρχείο αποτελεσμάτων για RAG
DEFAULT_OUTPUT_PATH = os.path.join(
    BASE_DIR, "results", "metaqa_llama3b_RAG_1hop_only.jsonl"
)

# Ο φάκελος όπου βρίσκεται το index που έφτιαξε ο Ingestor
# (πρέπει να είναι ίδιος με το persist_dir που έδωσες στο index_metaqa.py
#  ή ό,τι χρησιμοποιεί ήδη ο συμφοιτητής σου).
DEFAULT_PERSIST_DIR = os.path.join(BASE_DIR, "rag_indexes", "metaqa")

SLEEP_BETWEEN_CALLS = 0.1


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

            items.append(
                {
                    "id": q_id,
                    "question": q_text,
                    "gold_answer": gold_ans,
                    "source_split": src,
                }
            )
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


def call_rag(agent: RAGAgent, question_text: str, persist_dir: str) -> Dict[str, Any]:
    """
    Καλεί το RAG pipeline για μια ερώτηση.
    Χρησιμοποιεί τη μέθοδο generate_rag_persist του RAGAgent.
    """
    try:
        start_t = perf_counter()

        # Η generate_rag_persist - σύμφωνα με το pipeline.py -
        # επιστρέφει (answer, documents)
        answer, docs = agent.generate_rag_persist(
            prompt=question_text,
            persist_dir=persist_dir,
            retrieval_config={},  # αν θέλετε π.χ. {"k": 5}, βάλτε το εδώ
        )

        end_t = perf_counter()
        latency = end_t - start_t

        # answer: string, docs: λίστα με retrieved documents (δεν τα γράφουμε στο JSONL)
        return {
            "ok": True,
            "status": 200,
            "response_text": answer,
            "error": None,
            "latency": latency,
        }
    except Exception as e:
        # Σε περίπτωση σφάλματος, κρατάμε τι πήγε στραβά για debug
        return {
            "ok": False,
            "status": None,
            "response_text": None,
            "error": str(e),
            "latency": None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="RAG evaluation για MetaQA (1-hop) με KG_RAG.RAGAgent"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="JSONL αρχείο με ερωτήσεις MetaQA (default: data/metaqa_1hop_only.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL αρχείο εξόδου με αποτελέσματα RAG",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=DEFAULT_PERSIST_DIR,
        help="Φάκελος με το RAG index (persist_dir)",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Μέγιστος αριθμός ερωτήσεων για evaluation (π.χ. 100 για test).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    persist_dir = os.path.abspath(args.persist_dir)
    max_questions: Optional[int] = args.max_questions

    print(f"[INFO] Input:  {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Persist dir: {persist_dir}")

    dataset = load_metaqa(input_path)
    if max_questions is not None:
        dataset = dataset[:max_questions]
    total = len(dataset)
    print(f"[INFO] Loaded {total} questions.")

    done_ids = read_done_ids(output_path)
    if done_ids:
        print(f"[INFO] Resume mode: {len(done_ids)} already processed.")

    agent = RAGAgent()

    processed = 0
    for idx, sample in enumerate(dataset, start=1):
        qid = sample["id"]
        if qid in done_ids:
            continue

        question = sample["question"]
        gold_ans = sample["gold_answer"]
        src = sample["source_split"]

        res = call_rag(agent, question, persist_dir)

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
            "model": "llama3.2:3b+RAG",
        }

        append_result(output_path, record)
        processed += 1

        if (processed % 50 == 0) or (not res["ok"]):
            print(
                f"[{processed}/{total}] id={qid} hop={src} ok={res['ok']} "
                f"status={res['status']} latency={res['latency']} "
                f"error={res['error']}"
            )

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("[DONE] RAG evaluation complete.")
    print(f"[SAVED] Results in {output_path}")


if __name__ == "__main__":
    main()
