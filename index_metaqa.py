import os
import sys
import argparse

# === ΣΗΜΑΝΤΙΚΟ ===
# Βάλε εδώ το path προς τον φάκελο "src" που περιέχει το KG_RAG.
# Παράδειγμα:
#   C:\\Users\\alexp\\OneDrive\\Υπολογιστής\\KG_RAG\\src
KG_RAG_SRC_PATH = r"C:\Users\alexp\OneDrive\Υπολογιστής\KG-RAG_Comparison_System\KG-RAG-master\src"

# Προσθέτουμε το KG_RAG στο sys.path για να κάνουμε import
if KG_RAG_SRC_PATH not in sys.path:
    sys.path.append(KG_RAG_SRC_PATH)

from KG_RAG.pipeline import RAGAgent  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Build / update RAG index for MetaQA using KG_RAG.RAGAgent"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Φάκελος με τα documents/KB για MetaQA (όπως το περιμένει ο Ingestor).",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        required=True,
        help="Φάκελος όπου θα αποθηκευτεί το index (vectorstore κλπ).",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    persist_dir = os.path.abspath(args.persist_dir)

    print(f"[INFO] Using dataset_dir = {dataset_dir}")
    print(f"[INFO] Using persist_dir = {persist_dir}")

    os.makedirs(persist_dir, exist_ok=True)

    agent = RAGAgent()
    agent.index_documents(dataset_dir=dataset_dir, persist_dir=persist_dir)

    print("[DONE] Indexing finished.")


if __name__ == "__main__":
    main()
