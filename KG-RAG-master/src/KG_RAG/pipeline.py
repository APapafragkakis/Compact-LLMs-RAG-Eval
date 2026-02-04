from .ingestion.ingest import Ingestor
from .retrieval.retriever import Retriever
from .generation.generator import Generator
from .config import Config
from .utils import benchmark

import json
import yaml
import os

    
default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(default_config_path, "r") as f:
    config_yaml = yaml.safe_load(f)

class RAGAgent:
    def __init__(self, config=config_yaml):
        self.config = Config(config_yaml)
        self.config.update(config)
        self._initialize_components()
        
    def _initialize_components(self):
        ingestion_config = self.config.config["ingestion"]
        retrieval_config = self.config.config["retrieval"]
        generation_config = self.config.config["generation"]

        self.ingestor = Ingestor(ingestion_config)
        self.retriever = Retriever(retrieval_config)
        self.generator = Generator(generation_config)
        
    def index_documents(self, dataset_dir, persist_dir=None):
        return self.ingestor.ingest_documents(dataset_dir, persist_dir)
    
    def index_text(self, text_chunks, persist_dir=None):
        return self.ingestor.ingest_text(text_chunks, persist_dir)
    
    @benchmark
    def retrieve_persist(self, prompt, persist_dir, retrieval_config={}):
        return self.retriever.retrieve_persist(prompt, persist_dir, retrieval_config)
    
    def retrieve(self, prompt, vectorstore, bm25Index, retrieval_config={}):
        return self.retriever.retrieve(prompt, vectorstore, bm25Index, retrieval_config)

    @benchmark
    def generate(self, prompt):
        return self.generator.generate(prompt)
        
    @benchmark
    def generate_rag(self, question, rag_prompt, documents):
        if not documents:
            return self.generator.generate(question), []

        rag_prompt = config_yaml["generation"]["prompts"].get("rag_prompt", None)
        retrieval_text = rag_prompt + ".\n Snippets \n " + "\n ".join(f"Snippet {i+1}: {doc.page_content}" for i, doc in enumerate(documents))
        return self.generator.generate(retrieval_text + "\n Question: \n" + question), documents
    
    def generate_rag_persist(self, prompt, persist_dir=None, retrieval_config={}):
        documents = self.retrieve_persist(prompt, persist_dir, retrieval_config)
        rag_prompt = config_yaml["generation"]["prompts"].get("rag_prompt", None)
        return self.generate_rag(prompt, rag_prompt, documents)
    
    def generate_kgrag_persist(self, prompt, persist_dir=None, retrieval_config={}):
        documents = self.retrieve_persist(prompt, persist_dir, retrieval_config)
        rag_prompt = config_yaml["generation"]["prompts"].get("kgrag_prompt", None)
        return self.generate_rag(prompt, rag_prompt, documents)
            
        
    '''
    {"id": "2hop__460946_294723", "predicted_answer": "Jennifer Garner", "predicted_support_idxs": [0, 10], "predicted_answerable": true}
    '''
    def generate_to_file_with_facts(self, id, generationFunc, filePath):
        answer,docs = generationFunc()
        with open(filePath, 'a+') as outfile:
            predObj = {
                'id': id,
                'predicted_answer': answer,
                'predicted_support_idxs': [],
                'predicted_answerable': True
            }
            outfile.write(json.dumps(predObj) + "\n")

    
    def generate_triples(self, text):
        named_entities_prompt = config_yaml["generation"]["prompts"].get("ner", None)
        named_entities = self.generator.generate(named_entities_prompt + "\n" + text)
        named_entities_triples_prompt = config_yaml["generation"]["prompts"].get("ner_triples", None)
        raw_triples = self.generator.generate(named_entities_triples_prompt + "\n" + text +"\n named entities: " + named_entities)
        return json.loads(raw_triples)["triples"]
        # triples_ner_prompt = config_yaml["generation"]["prompts"].get("ner_triples", None)
        # triples_prompt = config_yaml["generation"]["prompts"].get("triples", None)
        # return self.generator.generate(triples_prompt + "\n" + text)
    
        

