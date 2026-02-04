import http.client
import json
import ast
import string
import re
from time import perf_counter, sleep
from pathlib import Path

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

RETRIEVAL_MODEL = "metaqa"
GENERATION_MODEL = "llama3.1:8b"

SLEEP_BETWEEN_REQUESTS = 0.7


def remote_retrieve(question: str):
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
    if not facts:
        return "Question: " + question + "\nAnswer:"

    facts_text = "\n".join(f"- {f}" for f in facts)

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
    llm_pred = llm_pred_str.rstrip()
    llm_pred = llm_pred.replace("\n", " ")
    llm_pred_list = llm_pred.split("|")
    llm_pred_list = [answer.strip() for answer in llm_pred_list]
    llm_pred_list = list(dict.fromkeys(llm_pred_list))
    return "|".join(llm_pred_list)


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s


def to_set(items_str: str):
    if not items_str:
        return set()
    items = items_str.split("|")
    return {normalize_text(x) for x in items if x.strip()}


def compute_metrics(gold: str, pred: str) -> dict:
    gset = to_set(gold)
    pset = to_set(pred)

    pred_parts = [p.strip() for p in pred.split("|") if p.strip()]
    top_pred = normalize_text(pred_parts[0]) if pred_parts else None
    top1_match = (top_pred in gset) if top_pred else False

    exact_match = (gset == pset)

    tp = len(gset & pset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "top1_match": top1_match,
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "pred_count": len(pset),
        "gold_count": len(gset),
    }


def run_rag(question: str):
    facts, t_retr, raw_retr = remote_retrieve(question)
    user_msg = build_generation_prompt(question, facts)
    raw_answer, t_gen = remote_generate(user_msg)
    processed_answer = postprocess_answer(raw_answer)
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


def evaluate(jsonl_path: str, output_path: str = None, resume: bool = True):
    jsonl_path = Path(jsonl_path)

    results_dir = Path("metaqa_rag_results")
    results_dir.mkdir(exist_ok=True)

    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"metaqa_{model_name}.jsonl"
    else:
        output_path = Path(output_path)

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

    llm_time_total = 0.0
    retrieval_time_total = 0.0
    endpoint_time_total = 0.0

    if resume_mode:
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

                metrics_prev = compute_metrics(gold_prev, pred_prev)

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

                retrieval_time_total += obj.get("retrieval_latency", 0.0)
                llm_time_total += obj.get("generation_latency", 0.0)
                endpoint_time_total += obj.get("total_latency", 0.0)

    t_start = perf_counter()

    mode = "a" if resume_mode else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            qid = sample.get("id")
            if resume_mode and qid in processed_ids:
                continue

            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_rag(question)
            metrics = compute_metrics(gold, res["processed_answer"])

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

            retrieval_time_total += res["retrieval_latency"]
            llm_time_total += res["generation_latency"]
            endpoint_time_total += res["total_latency"]

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
                acc = top1_correct / total * 100 if total else 0.0
                em = exact_matches / total * 100 if total else 0.0
                avg_f1 = (sum(macro_f1s) / len(macro_f1s)) if macro_f1s else 0.0
                print(f"[{total}] Acc: {acc:.2f}% | EM: {em:.2f}% | F1: {avg_f1:.3f} | {'✓' if metrics['top1_match'] else '✗'}")

            sleep(SLEEP_BETWEEN_REQUESTS)

    t_end = perf_counter()

    accuracy = top1_correct / total if total else 0.0
    exact_match_rate = exact_matches / total if total else 0.0

    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    macro_p = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_r = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0

    micro_p = total_tp / total_pred if total_pred else 0.0
    micro_r = total_tp / total_gold if total_gold else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    wall_time = t_end - t_start

    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples: {total}")
    print(f"Accuracy (top-1 match): {accuracy:.4f}")
    print(f"Exact Match: {exact_match_rate:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall: {micro_r:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Wall-clock Time: {wall_time:.2f}s ({wall_time/60:.1f} min)")
    print(f"Results: {output_path}")

    if total > 0:
        print(f"Retrieval total time: {retrieval_time_total:.2f}s")
        print(f"LLM total generation time: {llm_time_total:.2f}s")
        print(f"Endpoint total (retr+gen): {endpoint_time_total:.2f}s")
        print(f"Avg retrieval latency: {retrieval_time_total/total:.2f}s")
        print(f"Avg LLM latency: {llm_time_total/total:.2f}s")
        print(f"Avg endpoint latency: {endpoint_time_total/total:.2f}s")

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
        "time": wall_time,
        "llm_time_total": llm_time_total,
        "retrieval_time_total": retrieval_time_total,
        "endpoint_time_total": endpoint_time_total,
        "llm_avg_latency": llm_time_total / total if total else 0.0,
        "retrieval_avg_latency": retrieval_time_total / total if total else 0.0,
        "endpoint_avg_latency": endpoint_time_total / total if total else 0.0,
    }


if __name__ == "__main__":
    evaluate("data/metaqa_1hop_only.jsonl")
