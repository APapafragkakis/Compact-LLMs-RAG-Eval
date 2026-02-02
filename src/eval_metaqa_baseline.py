import http.client
import json
from time import perf_counter, sleep
from pathlib import Path
import re
import string

HOST = "demos.isl.ics.forth.gr"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

GENERATION_MODEL = "llama3.1:8b"
SLEEP_BETWEEN_REQUESTS = 0.7


def generate_baseline(question: str):
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

    return raw.strip(), (t1 - t0)


def normalize_baseline_answer(raw_answer: str, question: str) -> str:
    raw_answer = raw_answer.strip()
    if not raw_answer:
        return ""

    question_entity = None
    m = re.search(r"\[([^\]]+)\]", question)
    if m:
        question_entity = m.group(1).strip().lower()

    tmp = raw_answer.replace("|", "\n")
    parts = [p.strip() for p in tmp.splitlines() if p.strip()]

    filtered = []
    for p in parts:
        if question_entity and p.strip().lower() == question_entity:
            continue
        filtered.append(p)

    seen = set()
    unique = []
    for e in filtered:
        if e not in seen:
            seen.add(e)
            unique.append(e)

    if not unique:
        return ""
    return unique[0] if len(unique) == 1 else "|".join(unique)


def run_baseline(question: str):
    raw_answer, latency = generate_baseline(question)
    final_answer = normalize_baseline_answer(raw_answer, question)
    return {
        "question": question,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "generation_latency": latency,
        "total_latency": latency,
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
