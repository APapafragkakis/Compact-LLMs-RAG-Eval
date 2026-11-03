# Full Results

All numbers are computed from the per-question outputs in `out/` and match Tables 1–4 and Figure 3 of the paper. H@1 = Hits@1, EM = Exact Match, all accuracy values in %. Metric definitions are in [metrics.md](metrics.md).

McNemar tests use continuity correction on per-question Hits@1 correctness (RAG runs). Δ is Hits@1 of the second model minus the first. With 15 comparisons per dataset, p < 0.0033 is significant after Bonferroni correction at α = 0.05.

![Baseline vs RAG accuracy](figures/accuracy.png)

![Latency breakdown](figures/latency.png)

### MetaQA

| Model | Baseline H@1 | Baseline F1 | Baseline EM | RAG H@1 | RAG F1 | RAG EM |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 5.49 | 5.77 | 0.80 | 72.56 | 59.42 | 29.96 |
| Gemma 2 2B | 18.53 | 15.73 | 6.39 | 81.66 | 79.91 | 67.96 |
| Llama 3.2 3B | 23.52 | 19.11 | 7.61 | 85.34 | 78.52 | 60.53 |
| Phi-3 Mini 3.8B | 7.96 | 6.07 | 0.69 | 79.42 | 71.89 | 50.34 |
| Mistral 7B | 31.83 | 26.62 | 8.01 | 81.45 | 79.10 | 68.32 |
| Llama 3.1 8B | 36.19 | 31.07 | 15.02 | 86.15 | 81.35 | 66.53 |

### MetaQA McNemar tests

| Comparison | Δ (pp) | p-value | Sig. |
|---|---:|---:|:---:|
| L3.2-1B vs G2-2B | +9.10 | < 0.001 | *** |
| L3.2-1B vs L3.2-3B | +12.78 | < 0.001 | *** |
| L3.2-1B vs Phi-3 | +6.86 | < 0.001 | *** |
| L3.2-1B vs M-7B | +8.89 | < 0.001 | *** |
| L3.2-1B vs L3.1-8B | +13.58 | < 0.001 | *** |
| G2-2B vs L3.2-3B | +3.68 | < 0.001 | *** |
| G2-2B vs Phi-3 | -2.24 | < 0.001 | *** |
| G2-2B vs M-7B | -0.21 | 0.588 | n.s. |
| G2-2B vs L3.1-8B | +4.48 | < 0.001 | *** |
| L3.2-3B vs Phi-3 | -5.92 | < 0.001 | *** |
| L3.2-3B vs M-7B | -3.89 | < 0.001 | *** |
| L3.2-3B vs L3.1-8B | +0.80 | < 0.001 | *** |
| Phi-3 vs M-7B | +2.03 | < 0.001 | *** |
| Phi-3 vs L3.1-8B | +6.73 | < 0.001 | *** |
| M-7B vs L3.1-8B | +4.69 | < 0.001 | *** |

### MetaQA latency (s per query)

| Model | Retrieval | Generation | End-to-end |
|---|---:|---:|---:|
| Llama 3.2 1B | 0.434 | 0.457 | 0.891 |
| Gemma 2 2B | 0.456 | 0.459 | 0.915 |
| Llama 3.2 3B | 0.440 | 0.467 | 0.907 |
| Phi-3 Mini 3.8B | 0.453 | 0.236 | 0.689 |
| Mistral 7B | 0.446 | 0.275 | 0.721 |
| Llama 3.1 8B | 0.434 | 0.485 | 0.919 |

### WC2014QA

| Model | Baseline H@1 | Baseline F1 | Baseline EM | RAG H@1 | RAG F1 | RAG EM |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 6.03 | 9.63 | 1.50 | 75.35 | 83.02 | 51.08 |
| Gemma 2 2B | 12.90 | 16.90 | 8.69 | 69.38 | 76.00 | 63.73 |
| Llama 3.2 3B | 13.11 | 17.44 | 8.81 | 73.76 | 80.87 | 60.94 |
| Phi-3 Mini 3.8B | 11.34 | 16.33 | 6.08 | 84.85 | 90.43 | 58.87 |
| Mistral 7B | 8.21 | 15.65 | 7.16 | 92.70 | 94.19 | 64.12 |
| Llama 3.1 8B | 20.66 | 25.18 | 11.60 | 89.28 | 92.20 | 64.89 |

### WC2014QA McNemar tests

| Comparison | Δ (pp) | p-value | Sig. |
|---|---:|---:|:---:|
| L3.2-1B vs G2-2B | -5.97 | < 0.001 | *** |
| L3.2-1B vs L3.2-3B | -1.59 | 0.033 | * |
| L3.2-1B vs Phi-3 | +9.50 | < 0.001 | *** |
| L3.2-1B vs M-7B | +17.36 | < 0.001 | *** |
| L3.2-1B vs L3.1-8B | +13.93 | < 0.001 | *** |
| G2-2B vs L3.2-3B | +4.38 | < 0.001 | *** |
| G2-2B vs Phi-3 | +15.47 | < 0.001 | *** |
| G2-2B vs M-7B | +23.33 | < 0.001 | *** |
| G2-2B vs L3.1-8B | +19.90 | < 0.001 | *** |
| L3.2-3B vs Phi-3 | +11.09 | < 0.001 | *** |
| L3.2-3B vs M-7B | +18.94 | < 0.001 | *** |
| L3.2-3B vs L3.1-8B | +15.52 | < 0.001 | *** |
| Phi-3 vs M-7B | +7.85 | < 0.001 | *** |
| Phi-3 vs L3.1-8B | +4.43 | < 0.001 | *** |
| M-7B vs L3.1-8B | -3.42 | < 0.001 | *** |

### WC2014QA latency (s per query)

| Model | Retrieval | Generation | End-to-end |
|---|---:|---:|---:|
| Llama 3.2 1B | 0.294 | 0.429 | 0.723 |
| Gemma 2 2B | 0.291 | 0.464 | 0.755 |
| Llama 3.2 3B | 0.289 | 0.480 | 0.769 |
| Phi-3 Mini 3.8B | 0.338 | 0.280 | 0.618 |
| Mistral 7B | 0.289 | 0.326 | 0.614 |
| Llama 3.1 8B | 0.299 | 0.505 | 0.804 |

