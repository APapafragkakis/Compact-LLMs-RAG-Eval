import bm25s
import Stemmer
from typing import List
from langchain_core.documents import Document


class BM25Indexer:
    def __init__(self, config):
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")
    
    def index(self, documents: List[Document], persist_dir):
        corpus_records = [ ({"id": doc.metadata["id"], "source": doc.metadata["source"], "page":  doc.metadata.get("page",0), "row":  doc.metadata.get("row",0), "text": doc.page_content}) for doc in documents]
        corpus_lst = [ r["text"] for r in corpus_records]      
        corpus_tokens = bm25s.tokenize(corpus_lst, stopwords="en", stemmer=self.stemmer)
        retriever = bm25s.BM25(corpus=corpus_records)
        retriever.index(corpus_tokens)
        if persist_dir: retriever.save(persist_dir)
        return retriever

