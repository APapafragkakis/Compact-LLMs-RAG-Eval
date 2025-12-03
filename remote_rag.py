import http.client
import json
import ast
from time import perf_counter

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"
MODEL_NAME = "metaqa"

def remote_retrieve(question: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
    }
    body = json.dumps(payload)
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }

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
        return f"Question:\n{question}\n\nAnswer with ONLY the correct entity."

    facts_block = "\n".join(f"- {f}" for f in facts)

    return f"""
You are a QA system over a knowledge graph.
Use ONLY the following facts to answer the question.

Facts:
{facts_block}

Question:
{question}

Answer with ONLY the correct entity (no punctuation).
""".strip()

def remote_generate(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
    }
    body = json.dumps(payload)
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }

    conn = http.client.HTTPSConnection(HOST, timeout=60)
    t0 = perf_counter()
    conn.request("POST", GENERATE_ENDPOINT, body, headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    t1 = perf_counter()
    conn.close()

    latency = t1 - t0
    answer_text = raw.strip()

    return answer_text, latency

def run_rag(question: str):
    facts, t_retr, raw_retr = remote_retrieve(question)
    prompt = build_generation_prompt(question, facts)
    answer, t_gen = remote_generate(prompt)
    total_latency = t_retr + t_gen

    return {
        "question": question,
        "facts": facts,
        "prompt": prompt,
        "answer": answer,
        "retrieval_latency": t_retr,
        "generation_latency": t_gen,
        "total_latency": total_latency,
        "raw_retrieval": raw_retr,
    }

if __name__ == "__main__":
    q = "what does [Grégoire Colin] appear in"

    print("=== Running RAG for one question ===\n")

    res = run_rag(q)

    print("Question:", res["question"])
    print("\nRetrieved facts:")
    for f in res["facts"]:
        print("  -", f)

    print("\nPrompt sent to /generate:\n")
    print(res["prompt"])

    print("\nANSWER FROM /generate:\n")
    print(res["answer"])

    print("\nLatencies (sec):")
    print("  retrieval:", res["retrieval_latency"])
    print("  generation:", res["generation_latency"])
    print("  total:", res["total_latency"])
