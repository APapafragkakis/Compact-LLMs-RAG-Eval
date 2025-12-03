# remote_rag.py - COMPLETE FIXED VERSION

import http.client
import json
import ast
from time import perf_counter
import re

HOST = "demos.isl.ics.forth.gr"
RETRIEVE_ENDPOINT = "/SemanticRAG/generateEmbeddings"
GENERATE_ENDPOINT = "/SemanticRAG/generate"

RETRIEVAL_MODEL = "metaqa"
GENERATION_MODEL = "llama3.2:1b"


def remote_retrieve(question: str):
    payload = {
        "model": RETRIEVAL_MODEL,
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
        return (
            "Question: " + question + "\n"
            "Answer only the entity name without punctuation."
        )

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
    return raw.strip(), latency


def extract_entities_from_facts(facts: list[str]) -> set:
    """Εξάγει όλες τις entities που υπάρχουν στα facts"""
    entities = set()
    
    for fact in facts:
        # Extract movie/show titles (usually before "starred actors" or after "has tag")
        # Pattern: "TITLE starred actors NAME" or "TITLE has tag TAG"
        
        # Before "starred actors"
        match = re.match(r'^(.+?)\s+starred actors\s+(.+?)$', fact)
        if match:
            entities.add(match.group(1).strip())
            entities.add(match.group(2).strip())
            continue
        
        # Before "belongs to"
        match = re.match(r'^(.+?)\s+belongs to\s+', fact)
        if match:
            entities.add(match.group(1).strip())
            continue
            
        # After "has tag"
        match = re.match(r'^(.+?)\s+has tag\s+(.+?)$', fact)
        if match:
            entities.add(match.group(1).strip())
            continue
        
        # Generic: split by common keywords
        for keyword in [' starred actors ', ' directed by ', ' released in ']:
            if keyword in fact:
                parts = fact.split(keyword)
                for p in parts:
                    p = p.strip()
                    if p and len(p) > 2:
                        entities.add(p)
    
    return entities


def filter_relevant_facts(facts: list[str], question: str) -> list[str]:
    """Κρατάει facts που περιέχουν το entity ΚΑΙ 'starred actors'"""
    match = re.search(r'\[([^\]]+)\]', question)
    if not match:
        return facts
    
    entity = match.group(1).strip().lower()
    
    filtered = []
    for fact in facts:
        fact_lower = fact.lower()
        
        # Κρατάει "starred actors" facts που περιέχουν το entity
        if "starred actors" in fact_lower and entity in fact_lower:
            filtered.append(fact)
    
    # Αν δεν βρήκε τίποτα με "starred actors", κράτα ΟΛΑ τα facts
    return filtered if filtered else facts


def normalize_answer(raw_answer: str, facts: list[str], question: str) -> str:
    """
    Φιλτράρει το raw answer:
    1. Κρατάει ΜΟΝΟ entities που υπάρχουν στα facts
    2. Αφαιρεί την entity που είναι στα square brackets της ερώτησης
    """
    if not raw_answer.strip():
        return ""
    
    # Extract question entity (in square brackets)
    question_entity = None
    match = re.search(r'\[([^\]]+)\]', question)
    if match:
        question_entity = match.group(1).strip().lower()
    
    # Get all valid entities from facts
    valid_entities = extract_entities_from_facts(facts)
    valid_entities_lower = {e.lower() for e in valid_entities}
    
    # Parse raw answer (handle | and newlines)
    raw = raw_answer.strip()
    tmp = raw.replace("|", "\n")
    parts = [p.strip() for p in tmp.splitlines() if p.strip()]
    
    # Filter: keep only if entity exists in facts AND is not the question entity
    filtered = []
    for p in parts:
        p_lower = p.lower()
        
        # Skip if it's the question entity
        if question_entity and p_lower == question_entity:
            continue
        
        # Keep if it appears in facts
        if p_lower in valid_entities_lower:
            # Get original casing from facts
            for e in valid_entities:
                if e.lower() == p_lower:
                    filtered.append(e)
                    break
    
    # Remove duplicates while preserving order
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
    
    # ΦΙΛΤΡΑΡΙΣΜΑ - κρατάει μόνο relevant facts
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


if __name__ == "__main__":
    test_cases = [
        "what does [Grégoire Colin] appear in",
        "[Joe Thomas] appears in which movies",
        "what films did [Michelle Trachtenberg] star in",
    ]

    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        
        res = run_rag(q)
        
        print(f"\nFacts: {res['facts']}")
        print(f"\nRaw answer: {res['raw_answer']}")
        print(f"Final answer: {res['answer']}")
        print(f"Latency: {res['total_latency']:.2f}s")