import os
import json

PATH_WC = r"C:\Users\alexp\OneDrive\Υπολογιστής\RAG-LLM-Benchmarking\WC-P1.txt"
OUTPUT  = r"C:\Users\alexp\OneDrive\Υπολογιστής\RAG-LLM-Benchmarking\wc2014qa.jsonl"

def main():
    out = open(OUTPUT, "w", encoding="utf-8")

    with open(PATH_WC, "r", encoding="utf-8") as f:
        i = 0
        for line in f:
            parts = line.strip().split("\t")

            # expected structure: q \t a \t support \t distractors \t triples
            if len(parts) < 2:
                continue

            q = parts[0].strip()
            a = parts[1].strip()

            if not q or not a:
                continue

            rec = {
                "id": i,
                "question": q,
                "answer": [a],
                "source_split": "wc2014-1hop"
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            i += 1

    out.close()
    print(f"[DONE] Wrote {i} samples to {OUTPUT}")

if __name__ == "__main__":
    main()
