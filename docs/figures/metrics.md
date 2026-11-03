# Metrics

Both benchmarks are scored on normalized text, but the answer formats differ, so the details differ too.

## WC2014QA

Each question has a list of valid answers and the model returns one entity. The prediction is normalized by stripping trailing whitespace, replacing spaces with underscores, lowercasing and cutting at the first comma. Gold answers are lowercased.

- **Hits@1:** correct if the normalized prediction is exactly equal to any valid answer (exact membership, not substring).
- **EM:** correct only if the prediction equals the first listed valid answer.
- **F1:** token-level F1 between the prediction and the best-matching valid answer, averaged over questions.

## MetaQA (1-hop)

Answers are entity sets. Predictions and gold answers are lowercased, punctuation is removed, and both are split on `|` into sets.

- **Hits@1:** correct if the first predicted entity is in the gold set.
- **EM:** correct if the predicted set equals the gold set.
- **F1:** set-level F1 between predicted and gold entities, averaged over questions.

## Notes

- EM ≤ Hits@1 by construction. The gap is largest on set-valued questions, where several answers are valid (about 33% of MetaQA and 36% of WC2014QA). For those questions Hits@1 and F1 are the meaningful metrics.
- Latency is measured on the client with `perf_counter()` around each HTTP request: retrieval (embedding, hybrid search, reranking), generation (forward pass and decoding), and end-to-end as their sum.

## Datasets

- **MetaQA:** Zhang et al., *Variational Reasoning for Question Answering with Knowledge Graph*, AAAI 2018. Only the 1-hop test split is used.
- **WC2014QA:** Zhang et al., *Gaussian Attention Model and Its Application to Knowledge Base Embedding and Question Answering*, arXiv:1611.02266, 2016.
