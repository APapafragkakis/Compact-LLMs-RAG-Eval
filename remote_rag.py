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
    """
    Build a simple prompt using retrieved facts as context.
    No agent-style phrasing, just context + question + answer slot.
    """
    if not facts:
        return (
            f"Question:\n{question}\n\n"
            f"Answer: (only the entity name, no punctuation)"
        )

    facts_block = "\n".join(f"- {f}" for f in facts)

    return (
        f"Facts:\n"
        f"{facts_block}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Answer: (only the entity name, no punctuation)"
    )


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
