"""Convert a MetaQA QA file (question<TAB>answers) to the jsonl format in data/.

Example, for the 1-hop test split used in the paper:
    python tools/convert.py MetaQA/1-hop/vanilla/qa_test.txt data/metaqa.jsonl --hop 1-hop
"""

import argparse
import json
import os


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="MetaQA QA file, e.g. MetaQA/1-hop/vanilla/qa_test.txt")
    parser.add_argument("output", help="output jsonl file")
    parser.add_argument("--hop", default="1-hop", help="value for the source_split field (default: 1-hop)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        n = dump_file(args.input, args.hop, 0, out)

    print(f"[DONE] Wrote {n} lines to {args.output}")


if __name__ == "__main__":
    main()
