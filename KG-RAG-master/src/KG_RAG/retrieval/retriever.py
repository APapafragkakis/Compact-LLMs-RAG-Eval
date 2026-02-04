import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
import torch
from sklearn.metrics.pairwise import cosine_similarity




from .fulltext import BM25Retriever
from KG_RAG.utils import benchmark


class Retriever:
    def __init__(self, config):
        self.config = config
        self.bm25Retriever = BM25Retriever(self.config)

    def load(self, persist_dir):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]

        vectorstore_dir = os.path.join(
            persist_dir, self.config["vectorstore-dir"])
        vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
            # for custom directory. TODO Improve it in the future
            persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

        bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
        bm25Index = self.bm25Retriever.load(bm25_dir)

        return vectorstore, bm25Index

    def retrieve_persist(self, prompt: str, persist_dir=None, retrieval_config={}):
        if persist_dir is None:
            persist_dir = self.config["persist-dir"]

        topk = retrieval_config.get('topk', self.config["topk"])
        score_threshold = retrieval_config.get(
            'score_threshold', self.config["score-threshold"])

        if self.config['separate-filetypes']:
            extensions = retrieval_config.get("extensions", ['pdf', 'csv'])
            topk = retrieval_config.get('topk', self.config["topk"])
            documents = {}
            for i, ext in enumerate(extensions):
                vectorstore_dir = os.path.join(
                    persist_dir, ext, self.config["vectorstore-dir"])
                if not os.path.isdir(vectorstore_dir):
                    continue
                vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                    # for custom directory. TODO Improve it in the future
                    persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

                bm25_dir = os.path.join(
                    persist_dir, ext, self.config["bm25-dir"])
                bm25Index = self.bm25Retriever.load(bm25_dir)
                documents[ext] = self.retrieve(
                    prompt, vectorstore, bm25Index, retrieval_config)
            merged_docs = merge_docs(documents)
            return rerank_docs(prompt, merged_docs, topk, score_threshold)
        else:
            vectorstore_dir = os.path.join(
                persist_dir, self.config["vectorstore-dir"])
            vectorstore = Chroma(                                                                                            # DB is reloaded every time to be up to date and allow
                # for custom directory. TODO Improve it in the future
                persist_directory=vectorstore_dir, embedding_function=OllamaEmbeddings(model=self.config["embedding-model"]))

            bm25_dir = os.path.join(persist_dir, self.config["bm25-dir"])
            bm25Index = self.bm25Retriever.load(bm25_dir)

            return rerank_docs(prompt, self.retrieve(prompt, vectorstore, bm25Index, retrieval_config, True), topk, score_threshold)

    def retrieve(self, prompt: str, vectorstore, bm25Index, retrieval_config={}):
        candidate_pool_size = retrieval_config.get(
            'candidate-pool-size', self.config["candidate-pool-size"])
        
        score_threshold = retrieval_config.get(
            'score_threshold', self.config["score-threshold"])

        vector_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": candidate_pool_size, "score_threshold": score_threshold})

        documents = {}
        documents["BM25"] = self.bm25Retriever.retrieve(prompt, bm25Index)[
            :candidate_pool_size]
        
        documents["DPR"] = vector_retriever.invoke(prompt)
        # get_relevant_documents
        return merge_docs(documents)

    def get_similarity(self, text1, text2):
        embeddings_model = OllamaEmbeddings(
            model=self.config["embedding-model"])
        embedding1 = embeddings_model.embed_query(text1)
        embedding2 = embeddings_model.embed_query(text2)
        similarity = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        return similarity


# Merge and rerank
@benchmark
def merge_docs(docsDict):
    merged_docs = []
    for key, docs in docsDict.items():
        merged_docs.extend(docs)

    if len(merged_docs) == 0:
        return []
    # Filter duplicate docs
    unique_merged_docs = []
    seen_combinations = set()
    for doc in merged_docs:
        source = doc.metadata.get("source")
        # page = doc.metadata.get("page")
        uniqueness_key = (doc.page_content, source)

        if uniqueness_key not in seen_combinations:
            unique_merged_docs.append(doc)
            seen_combinations.add(uniqueness_key)

    return unique_merged_docs


hf_ce = HuggingFaceCrossEncoder(
    model_name="Alibaba-NLP/gte-reranker-modernbert-base",
    model_kwargs={
        "automodel_args": {
            "torch_dtype": torch.float32,          # force fp32
            "attn_implementation": "eager",        # avoid FlashAttention path
        },
        "trust_remote_code": True,                 # often required for ModernBERT
        # "device": "cuda"  # or "cpu" if you want to force CPU
    },
)

@benchmark
def rerank_docs(query, docs, topk, score_threshold=0):
    if len(docs) == 0:
        return []
    
    texts = [(query, doc.page_content) for doc in docs]
    scores = hf_ce.score(texts)
    
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    doc_score_pairs = doc_score_pairs[:topk]
    
    final_docs = [doc for doc, score in doc_score_pairs]
    
    max_score = doc_score_pairs[0][1]
    min_score = doc_score_pairs[-1][1]
    
    def normalize_score(score):
        return (score - min_score) / (max_score - min_score) if max_score != min_score else 1.0
    
    final_docs_threshold = [doc for doc, score in doc_score_pairs 
                           if normalize_score(score) >= score_threshold]
    
    final_docs_scored = [(doc, score) for doc, score in doc_score_pairs]
    
    return final_docs_threshold