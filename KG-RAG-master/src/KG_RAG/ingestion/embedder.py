from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..utils import call_in_batches
import logging


class Embedder:
    def __init__(self, model: str):
        self.model = model

    def index_embeddings(self, text_chunks, persist_dir):
        logging.info("Indexing embeddings")
        
        kwargs = {"persist_directory": persist_dir} if persist_dir else {}

        vectorstore = Chroma(
            collection_metadata={"hnsw:space": "cosine", "hnsw:search_ef": 1000, "hnsw:construction_ef": 500},
            embedding_function=OllamaEmbeddings(model=self.model),
            **kwargs
        )

        def index_func(documents): return vectorstore.add_documents(
            documents=documents)
        call_in_batches(index_func, text_chunks, 500)
        
        return vectorstore

    def get_vectorstore(self):
        return self.vectorstore
