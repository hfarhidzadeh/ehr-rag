# Retriever comparison: compare_minilm

Retrieval only (no answer generation) on the synthetic gold questions; one run per mode with the same questions and k.

## Configuration

- **embedder**: minilm
- **k**: 6
- **n_questions**: 210
- **n_questions_available**: 210
- **rrf_k**: 60
- **candidate_pool**: 30
- **index**: embedder=minilm, dim=384, model=sentence-transformers/all-MiniLM-L6-v2, collection=ehr_chunks_minilm, n_docs=623, n_chunks=2210, n_patients=30, data_seed=42, chunk_max_chars=900, chunk_overlap_sentences=1, created_at=2026-10-21T21:11:18+00:00
- **modes**: ['random', 'bm25', 'dense', 'hybrid']
- **generate**: False
- **llm**: none
- **runtime_s**: 10.76
- **created_at**: 2026-10-21T21:31:27+00:00
- **synthetic**: True

## Control checks

| status | check | value | threshold | detail |
|---|---|---|---|---|
| **PASS** | leakage_rate == 0 (random) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | leakage_rate == 0 (bm25) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | verbatim recall@1 >= 0.9 (bm25) | 1.000 | >= 0.900 | n=30 |
| **PASS** | leakage_rate == 0 (dense) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | leakage_rate == 0 (hybrid) | 0.000 | == 0.000 | leaked chunks: 0 |
| **FAIL** | verbatim recall@1 >= 0.9 (hybrid) | 0.800 | >= 0.900 | n=30 |
| **PASS** | random recall@k <= best real recall@k - 0.2 | 0.713 | <= 0.780 | best real retriever: hybrid (0.980); positives only |
| **PASS** | random primary_recall@k <= best real primary_recall@k - 0.2 | 0.127 | <= 0.473 | best real retriever: bm25 (0.673); positives only |
| **PASS** | random evidence_recall@k <= best real evidence_recall@k - 0.2 | 0.253 | <= 0.747 | best real retriever: hybrid (0.947); positives only |

## Positives (all positive categories)

| mode | n | R@1 | R@3 | R@k | pR@1 | pR@k | eR@1 | eR@k | MRR |
|---|---|---|---|---|---|---|---|---|---|
| random | 150 | 0.340 | 0.553 | 0.713 | 0.040 | 0.127 | 0.027 | 0.253 | 0.467 |
| bm25 | 150 | 0.720 | 0.867 | 0.960 | 0.447 | 0.673 | 0.680 | 0.920 | 0.807 |
| dense | 150 | 0.607 | 0.807 | 0.933 | 0.173 | 0.580 | 0.480 | 0.867 | 0.720 |
| hybrid | 150 | 0.700 | 0.867 | 0.980 | 0.313 | 0.673 | 0.653 | 0.947 | 0.795 |

## Recall@1 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.367 | 0.200 | 0.633 | 0.133 |
| bm25 | 0.883 | 0.200 | 0.633 | 1.000 |
| dense | 0.717 | 0.367 | 0.767 | 0.467 |
| hybrid | 0.850 | 0.233 | 0.767 | 0.800 |

## Recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.783 | 0.633 | 0.900 | 0.467 |
| bm25 | 1.000 | 0.967 | 0.833 | 1.000 |
| dense | 0.950 | 1.000 | 0.900 | 0.867 |
| hybrid | 1.000 | 0.967 | 0.967 | 0.967 |

## Primary recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.067 | 0.000 | 0.033 | 0.467 |
| bm25 | 0.750 | 0.800 | 0.067 | 1.000 |
| dense | 0.550 | 0.800 | 0.133 | 0.867 |
| hybrid | 0.717 | 0.867 | 0.100 | 0.967 |

## Evidence recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.317 | 0.100 | 0.533 | 0.000 |
| bm25 | 1.000 | 0.967 | 0.633 | 1.000 |
| dense | 0.950 | 0.900 | 0.733 | 0.800 |
| hybrid | 1.000 | 0.967 | 0.833 | 0.933 |

## MRR by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.505 | 0.323 | 0.753 | 0.251 |
| bm25 | 0.932 | 0.446 | 0.723 | 1.000 |
| dense | 0.795 | 0.562 | 0.825 | 0.623 |
| hybrid | 0.904 | 0.483 | 0.836 | 0.849 |

## Leakage and latency

| mode | n | Leak | ms mean | ms p95 |
|---|---|---|---|---|
| random | 210 | 0.000 | 0 | 0 |
| bm25 | 210 | 0.000 | 0 | 1 |
| dense | 210 | 0.000 | 33 | 43 |
| hybrid | 210 | 0.000 | 17 | 23 |

Metric key: R@1/R@3/R@k = any gold document in the top 1/3/k chunks (positives only); pR@1/pR@k = the same for the canonical gold document only (structured list, lab panel or quoted note); eR@1/eR@k = a retrieved chunk of a gold document whose own text contains an expected keyword (chunk-level); MRR = mean reciprocal rank of the first gold document; Acc = positives answered without abstaining and containing an expected keyword; Acc|answered = Acc over non-abstained positives; FalseAbst = positives that abstained; CorrectAbst = negatives that abstained; CiteValid = answers whose citations all point at retrieved chunks of the asked patient; Leak = questions with any retrieved chunk of another patient. '-' = not applicable.
