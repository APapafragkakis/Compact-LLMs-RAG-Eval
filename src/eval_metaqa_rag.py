# evaluate_complete.py - Complete RAG Evaluation for MetaQA

import http.client
import json
import ast
from time import perf_counter
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

RETRIEVAL_MODEL = "metaqa"
GENERATION_MODEL = "llama3.1:8b"  # Change this to test different models


# ============================================================================
# RAG FUNCTIONS
# ============================================================================

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
        return "Question: " + question + "\nAnswer only the entity name without punctuation."

    facts_text = "\n".join(f"- {f}" for f in facts)
    return (
        "Facts:\n" + facts_text + "\n\n"
        "Question: " + question + "\n"
        "Answer with entity names only (separate multiple with |). No explanations."
    )


def remote_generate(user_text: str):
    system_message = (
        "You answer questions using ONLY the facts provided.\n\n"
        "Rules:\n"
        "1. Extract ONLY entities that appear in the facts\n"
        "2. For 'what does [X] appear in', answer with the MOVIE/SHOW names, NOT [X]\n"
        "3. Multiple answers: separate with | (e.g. 'Movie1|Movie2')\n"
        "4. DO NOT include:\n"
        "   - The entity in square brackets from the question\n"
        "   - Any entities not mentioned in the facts\n"
        "   - Any explanations or extra text\n\n"
        "Example:\n"
        "Facts: Before the Rain starred actors Grégoire Colin\n"
        "Question: what does [Grégoire Colin] appear in\n"
        "Answer: Before the Rain\n"
        "(NOT 'Grégoire Colin' - that's the question entity)\n"
    )

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


def extract_entities_from_facts(facts: list[str]) -> set:
    entities = set()
    
    for fact in facts:
        match = re.match(r'^(.+?)\s+starred actors\s+(.+?)$', fact)
        if match:
            entities.add(match.group(1).strip())
            entities.add(match.group(2).strip())
            continue
        
        match = re.match(r'^(.+?)\s+belongs to\s+', fact)
        if match:
            entities.add(match.group(1).strip())
            continue
            
        match = re.match(r'^(.+?)\s+has tag\s+(.+?)$', fact)
        if match:
            entities.add(match.group(1).strip())
            continue
        
        for keyword in [' starred actors ', ' directed by ', ' released in ']:
            if keyword in fact:
                parts = fact.split(keyword)
                for p in parts:
                    p = p.strip()
                    if p and len(p) > 2:
                        entities.add(p)
    
    return entities


def filter_relevant_facts(facts: list[str], question: str) -> list[str]:
    match = re.search(r'\[([^\]]+)\]', question)
    if not match:
        return facts
    
    entity = match.group(1).strip().lower()
    
    filtered = []
    for fact in facts:
        fact_lower = fact.lower()
        if "starred actors" in fact_lower and entity in fact_lower:
            filtered.append(fact)
    
    return filtered if filtered else facts


def normalize_answer(raw_answer: str, facts: list[str], question: str) -> str:
    if not raw_answer.strip():
        return ""
    
    question_entity = None
    match = re.search(r'\[([^\]]+)\]', question)
    if match:
        question_entity = match.group(1).strip().lower()
    
    valid_entities = extract_entities_from_facts(facts)
    valid_entities_lower = {e.lower() for e in valid_entities}
    
    raw = raw_answer.strip()
    tmp = raw.replace("|", "\n")
    parts = [p.strip() for p in tmp.splitlines() if p.strip()]
    
    filtered = []
    for p in parts:
        p_lower = p.lower()
        
        if question_entity and p_lower == question_entity:
            continue
        
        if p_lower in valid_entities_lower:
            for e in valid_entities:
                if e.lower() == p_lower:
                    filtered.append(e)
                    break
    
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


def run_rag(question: str):
    facts, t_retr, raw_retr = remote_retrieve(question)
    facts = filter_relevant_facts(facts, question)
    user_msg = build_generation_prompt(question, facts)
    raw_answer, t_gen = remote_generate(user_msg)
    final_answer = normalize_answer(raw_answer, facts, question)
    total_latency = t_retr + t_gen

    return {
        "question": question,
        "facts": facts,
        "prompt": user_msg,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "retrieval_latency": t_retr,
        "generation_latency": t_gen,
        "total_latency": total_latency,
        "raw_retrieval": raw_retr,
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

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


def compare_prediction(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return pred == gold
    
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    
    pred_parts = set(p.strip() for p in pred.split("|") if p.strip())
    gold_parts = set(g.strip() for g in gold.split("|") if g.strip())
    
    return pred_parts == gold_parts


def evaluate(jsonl_path: str, output_path: str = None):
    jsonl_path = Path(jsonl_path)
    
    # Create results directory
    results_dir = Path("rag_results")
    results_dir.mkdir(exist_ok=True)
    
    # Auto-generate output filename if not provided
    if output_path is None:
        model_name = GENERATION_MODEL.replace(":", "_").replace(".", "_")
        output_path = results_dir / f"metaqa_{model_name}.jsonl"
    else:
        output_path = Path(output_path)

    total = 0
    correct = 0
    t_start = perf_counter()

    print(f"\n{'='*60}")
    print(f"EVALUATION: {GENERATION_MODEL}")
    print(f"{'='*60}\n")

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in load_jsonl(jsonl_path):
            qid = sample.get("id")
            question = sample.get("question", "")
            gold_raw = sample.get("answer")
            gold = normalize_gold_answer(gold_raw)

            res = run_rag(question)
            pred = res["answer"]

            ok = compare_prediction(pred, gold)
            total += 1
            if ok:
                correct += 1

            out_obj = {
                "id": qid,
                "question": question,
                "gold_answer": gold_raw,
                "gold_norm": gold,
                "prediction": pred,
                "retrieval_latency": res["retrieval_latency"],
                "generation_latency": res["generation_latency"],
                "total_latency": res["total_latency"],
                "raw_answer": res["raw_answer"],
                "raw_retrieval": res["raw_retrieval"],
                "correct": ok,
            }
            fout.write(json.dumps(out_obj) + "\n")

            if total % 10 == 0:
                acc = correct / total * 100
                print(f"[{total}] accuracy: {acc:.2f}% | last: {'✓' if ok else '✗'} {question[:50]}")

    t_end = perf_counter()
    acc = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Dataset: {jsonl_path}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Total time: {t_end - t_start:.2f}s")
    print(f"Avg latency: {(t_end - t_start) / total:.2f}s per sample")
    print(f"Results saved: {output_path}")
    print(f"{'='*60}\n")
    
    return {"accuracy": acc, "correct": correct, "total": total, "time": t_end - t_start}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run evaluation on MetaQA 1-hop
    evaluate("data/metaqa_1hop_only.jsonl")