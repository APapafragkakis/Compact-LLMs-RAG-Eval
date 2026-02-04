from .data_loader import DataLoader
from .text_splitter import Splitter
from .embedder import Embedder
from .fulltext import BM25Indexer
import logging
from langchain_core.documents import BaseDocumentTransformer, Document
import os
from typing import Iterable
from KG_RAG.utils import benchmark

class Ingestor:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.splitter = Splitter()
        self.embedder = Embedder(config["embedding-model"])
        self.fulltextIndexer = BM25Indexer(config)
        
    @benchmark
    def ingest_documents(self, dataset_dir: str, persist_dir: str):
        supported_ext = ['pdf', 'csv']
        if self.config['separate-filetypes']:
            vectorstores = []
            for ext in supported_ext:
                documents = self.data_loader.load(dataset_dir, [ext])
                for doc in documents:
                    doc.metadata["id"] = doc.metadata.get("source", "")
                if len(documents) > 0:
                    vectorstores.append(self.ingest(documents, os.path.join(persist_dir,ext)))
            return vectorstores
        else:
            documents = self.data_loader.load(dataset_dir)
            for doc in documents:
                doc.metadata["id"] = doc.metadata.get("source", "")
            return self.ingest(documents, persist_dir)
    
    """
    :param text_chunks: {'id':..., 'text':...}
    """
    def ingest_text(self, text_chunks: Iterable[str], persist_dir: str):
        documents = [Document(chunk["text"], metadata={"id": chunk["id"], "source": chunk["id"]}) for chunk in text_chunks]
        return self.ingest(documents, persist_dir)
        
    @benchmark
    def ingest(self, documents: list, persist_dir: str):
        document_chunks = self.splitter.split_documents(documents)
        if persist_dir:
            vectorstore_dir = os.path.join(persist_dir, self.config["vectorstore-dir"])
            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            os.makedirs(vectorstore_dir, exist_ok=True)
            os.makedirs(bm25_dir, exist_ok=True)
        else:
            bm25_dir = None
            vectorstore_dir = None
            logging.warn("In-Memory indexing")
        
        logging.info("Full-text Indexing")
        bm25Index = self.fulltextIndexer.index(document_chunks, bm25_dir)

        logging.info("Vector Indexing")
        vectorstore = self.embedder.index_embeddings(document_chunks, vectorstore_dir)
        return vectorstore, bm25Index

    def get_vectorstore(self):
        return self.embedder.get_vectorstore()