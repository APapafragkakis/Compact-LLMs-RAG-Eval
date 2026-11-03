# Beyond Parameter Count: Selecting Compact Language Models for Retrieval-Augmented Generation

Code and results for the paper:

**Beyond Parameter Count: Selecting Compact Language Models for Retrieval-Augmented Generation**  
Alexandros Papafragkakis, Yannis Tzitzikas  
University of Crete & FORTH-ICS

Published in the Proceedings of the 14th EETN Conference on Artificial Intelligence (SETN 2026), Chania, Greece, September 9–11, 2026. ACM.  
DOI: https://doi.org/10.1145/3860954.3860972

## Overview

We evaluate six compact open-source language models (1B–8B parameters) as the generator in a controlled RAG pipeline. Retrieval and prompting are kept fixed, so differences come from the generator.

We report:

- **Accuracy:** Hits@1, F1, Exact Match (EM), with McNemar significance tests for all pairwise comparisons
- **Latency:** retrieval, generation, end-to-end

## Key Findings

- Parameter count is an unreliable predictor of RAG accuracy across the compact model families we evaluate.
- MetaQA: Gemma 2 2B and Mistral 7B are statistically tied (81.66% vs 81.45% Hits@1, p = 0.588). Llama 3.2 3B is within 0.8pp of Llama 3.1 8B.
- WC2014QA: Mistral 7B beats Llama 3.1 8B (92.70% vs 89.28%, p < 0.001) with about 24% lower end-to-end latency (0.61s vs 0.80s).
- Retrieval latency is roughly the same for all generators. Generation latency depends on architecture, not size: Phi-3 Mini and Mistral 7B are the fastest.
- RAG improves Hits@1 by +49.6pp to +84.5pp for all models.

## Evaluated Models

- Llama 3.2 (1B, 3B)
- Gemma 2 (2B)
- Phi-3 Mini (3.8B)
- Mistral 7B
- Llama 3.1 8B

## Benchmarks

- **MetaQA (1-hop)**: 9,947 test questions, movie domain
- **WC2014QA**: 6,482 test questions, 2014 FIFA World Cup

## Experimental Setup

- Fixed retrieval pipeline (embeddings, hybrid search, reranker, prompt template)
- Deterministic decoding (`temperature=0`, `max_new_tokens=128`)
- 4-bit quantization (bitsandbytes NF4)
- Latency measured client-side as HTTP API round-trip
- Server GPU: AMD Radeon RX 9070XT (16GB VRAM)

See Section 3 of the paper for details.

## Reproducing Results

The scripts use the SemanticRAG HTTP API, which is not publicly available.

1. Set the API endpoint and model identifiers in the scripts under `src/`
2. Run the evaluation per dataset and per model
3. Aggregate metrics with the scripts under `tools/`

Per-question outputs for all runs are in `out/`.

## Citation

```bibtex
@inproceedings{papafragkakis2026beyond,
  author    = {Papafragkakis, Alexandros and Tzitzikas, Yannis},
  title     = {Beyond Parameter Count: Selecting Compact Language Models for Retrieval-Augmented Generation},
  booktitle = {Proceedings of the 14th EETN Conference on Artificial Intelligence (SETN 2026)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3860954.3860972}
}
```
