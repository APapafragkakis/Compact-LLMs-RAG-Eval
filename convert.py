import os, json

PATH_1H = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation\MetaQA\1-hop\vanilla\qa_test_qtype.txt"
PATH_2H = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation\MetaQA\2-hop\vanilla\qa_test_qtype.txt"
PATH_3H = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation\MetaQA\3-hop\vanilla\qa_test_qtype.txt"

BASE_DIR = r"C:\Users\alexp\OneDrive\Υπολογιστής\Endpoint_Evaluation"
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "metaqa_full.jsonl")

def parse_line(line: str):
    s = line.strip()
    if not s:
        return None, None
    parts = s.split("\t")
    if len(parts) >= 2:
        q = parts[0].strip()
        a = "\t".join(parts[1:]).strip()
        return q, a
    if "|||" in s:
        p = s.split("|||", 1)
        return p[0].strip(), p[1].strip()
    return None, None

def dump_file(path: str, hop: str, start_id: int, f_out) -> int:
    wrote = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q, a = parse_line(line)
            if not q:
                continue
            rec = {
                "id": start_id + wrote,
                "question": q,
                "answer": [a] if a else [],
                "source_split": hop,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    return wrote

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    total = 0
    cur_id = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for path, hop in [(PATH_1H, "1-hop"), (PATH_2H, "2-hop"), (PATH_3H, "3-hop")]:
            if not os.path.exists(path):
                print(f"[WARN] missing: {path}")
                continue
            n = dump_file(path, hop, cur_id, out)
            print(f"[INFO] {hop}: wrote {n} lines from {os.path.basename(path)}")
            cur_id += n
            total += n
    print(f"[DONE] Wrote {total} lines to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
