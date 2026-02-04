import bm25s
import Stemmer
from typing import List
from langchain_core.documents import Document


class BM25Retriever:
    def __init__(self, config):
        self.config = config
        self.stemmer = Stemmer.Stemmer("english")

    def retrieve(self, query, index):
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = index.retrieve(query_tokens, k=self.config.get('candidate-pool-size', 5))
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]

        return [
            Document(result["text"], metadata={
                    "source": result.get("source", result["id"]), "page": result.get("page",0), "row": result.get("row",0), "id": result["id"]})
            for result, score in zip(results[0, :], scores[0, :]) if score > 1
        ]

    def load(self, persist_dir):
        return bm25s.BM25.load(persist_dir, load_corpus=True)
    
    def retrieve_persist(self, query, persist_dir):
        try: 
            index = bm25s.BM25.load(persist_dir, load_corpus=True)
            return self.retrieve(query, index)
        except FileNotFoundError:
            print(f'BM25 index not found in {persist_dir} , ensure the correct persist_dir is provided')
            return []