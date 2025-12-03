import http.client
import json
import ast
from time import perf_counter

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"
MODEL_NAME = "metaqa"


def remote_retrieve(question: str):
    """
    Κλήση στο /SemanticRAG/generateEmbeddings
    Payload: { "model": MODEL_NAME, "prompt": question }
    Επιστροφή: λίστα από facts (ως strings), latency, raw response.
    """
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

    # raw π.χ. "['-: Before the Rain starred actors Grégoire Colin']"
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
    Χτίζει το user μήνυμα που θα πάει στο LLM (ως 'content' του role=user).
    Δεν χρησιμοποιούμε πια πεδίο 'prompt' στο JSON, αλλά conversation.
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


def remote_generate(user_text: str):
    """
    Κλήση στο /SemanticRAG/generate

    Ακολουθούμε ΑΚΡΙΒΩΣ το schema που σου έστειλαν:

    data = {
        "model": model_name,
        "conversation": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": user_text}
        ]
    }
    """
    system_message = (
        "You answer questions about movies using only the provided facts. "
        "Return only the correct entity name as the answer, without punctuation or extra text."
    )

    payload = {
        "model": MODEL_NAME,
        "conversation": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
        ],
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
    """
    Πλήρες RAG pipeline:
    - Retrieval στο /generateEmbeddings
    - Χτίσιμο user prompt από facts + question
    - Generation στο /generate με model + conversation
    """
    facts, t_retr, raw_retr = remote_retrieve(question)
    user_msg = build_generation_prompt(question, facts)
    answer, t_gen = remote_generate(user_msg)
    total_latency = t_retr + t_gen

    return {
        "question": question,
        "facts": facts,
        "prompt": user_msg,
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

    print("\nPrompt sent as USER message:\n")
    print(res["prompt"])

    print("\nANSWER FROM /generate:\n")
    print(res["answer"])

    print("\nLatencies (sec):")
    print("  retrieval:", res["retrieval_latency"])
    print("  generation:", res["generation_latency"])
    print("  total:", res["total_latency"])
