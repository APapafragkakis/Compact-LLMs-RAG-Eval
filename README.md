# Compact Language Models in Retrieval-Augmented Generation

This repository accompanies the paper:

**Compact Language Models in Retrieval-Augmented Generation:  
Accuracy–Latency Trade-offs under a Unified Pipeline**  
Alexandros Papafragkakis, Yannis Tzitzikas  
University of Crete & FORTH-ICS


## Overview

Retrieval-Augmented Generation (RAG) systems are increasingly deployed in environments where **latency, cost, and hardware limits** matter as much as answer quality.

This work provides a **systematic evaluation of compact open-source language models (1B–8B parameters)** within a **controlled RAG pipeline**, isolating the effect of the generator model while keeping retrieval fixed.

We evaluate six models across two knowledge-intensive QA benchmarks and analyze:

- **Accuracy:** Hits@1, F1, Exact Match (EM)
- **Latency:** retrieval, generation, end-to-end
- **Bottlenecks:** retrieval-bound vs generation-bound behavior

For full details, see the paper.

---

## Key Findings

- Model size alone is a **poor predictor** of RAG performance in the 1B–8B range.
- **Mid-size models (≈3B)** can approach **8B accuracy** when retrieval quality is strong.
- Smaller models benefit **disproportionately** from retrieval augmentation.
- Latency bottlenecks differ by model size:
  - **1B–3B:** mostly **retrieval-bound**
  - **7B–8B:** increasingly **generation-bound**
- **Mistral 7B** lies on the accuracy–latency efficiency frontier for factual QA in our setup.

---

## Evaluated Models

- Llama 3.2 (1B, 3B)
- Gemma 2 (2B)
- Phi-3 Mini (3.8B)
- Mistral 7B
- Llama 3.1 8B

All models are evaluated under the **same retrieval configuration**.

---

## Benchmarks

- **MetaQA (single-hop)** — movie-domain QA  
- **WC2014QA (single-hop)** — FIFA World Cup factual QA

---

## Experimental Setup (Summary)

- Fixed retrieval pipeline (embeddings, hybrid search, reranker, prompt template)
- Deterministic decoding (**temperature = 0**, `max_new_tokens=128`)
- 4-bit quantization (bitsandbytes **NF4**)
- Latency measured end-to-end via HTTP API round-trip
- GPU (server): AMD Radeon RX 9070XT (16GB VRAM)

See Section 3 of the paper for full methodological details.

---

## Reproducing Results

This repository assumes access to the **SemanticRAG HTTP API** used in our experiments.

### High-level workflow

1. Configure the API endpoint and model identifiers  
2. Run the evaluation scripts (per dataset, per model)  
3. Aggregate metrics and generate plots/tables  

Detailed configurations (retrieval settings, prompt template, normalization, and metrics) are described in the paper.

---

## Notes on Fair Comparison

To isolate generator effects, we keep retrieval and prompting fixed across models:

- Same embedder, hybrid retrieval parameters, reranker, and target top-k  
- Same decoding settings (`temperature = 0`) and maximum generation length  
- Same evaluation normalization applied to all model predictions  

