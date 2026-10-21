# Retriever comparison: compare_tfidf

Retrieval only (no answer generation) on the synthetic gold questions; one run per mode with the same questions and k.

## Configuration

- **embedder**: tfidf
- **k**: 6
- **n_questions**: 210
- **n_questions_available**: 210
- **rrf_k**: 60
- **candidate_pool**: 30
- **index**: embedder=tfidf, dim=256, model=tfidf-svd(dim=256), collection=ehr_chunks_tfidf, n_docs=623, n_chunks=2210, n_patients=30, data_seed=42, chunk_max_chars=900, chunk_overlap_sentences=1, created_at=2026-10-21T20:19:50+00:00
- **modes**: ['random', 'bm25', 'dense', 'hybrid']
- **generate**: False
- **llm**: none
- **runtime_s**: 7.58
- **created_at**: 2026-10-21T21:30:59+00:00
- **synthetic**: True

## Control checks

| status | check | value | threshold | detail |
|---|---|---|---|---|
| **PASS** | leakage_rate == 0 (random) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | leakage_rate == 0 (bm25) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | verbatim recall@1 >= 0.9 (bm25) | 1.000 | >= 0.900 | n=30 |
| **PASS** | leakage_rate == 0 (dense) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | leakage_rate == 0 (hybrid) | 0.000 | == 0.000 | leaked chunks: 0 |
| **PASS** | verbatim recall@1 >= 0.9 (hybrid) | 0.967 | >= 0.900 | n=30 |
| **PASS** | random recall@k <= best real recall@k - 0.2 | 0.713 | <= 0.767 | best real retriever: hybrid (0.967); positives only |
| **PASS** | random primary_recall@k <= best real primary_recall@k - 0.2 | 0.127 | <= 0.513 | best real retriever: hybrid (0.713); positives only |
| **PASS** | random evidence_recall@k <= best real evidence_recall@k - 0.2 | 0.253 | <= 0.720 | best real retriever: bm25 (0.920); positives only |

## Positives (all positive categories)

| mode | n | R@1 | R@3 | R@k | pR@1 | pR@k | eR@1 | eR@k | MRR |
|---|---|---|---|---|---|---|---|---|---|
| random | 150 | 0.340 | 0.553 | 0.713 | 0.040 | 0.127 | 0.027 | 0.253 | 0.467 |
| bm25 | 150 | 0.720 | 0.867 | 0.960 | 0.447 | 0.673 | 0.680 | 0.920 | 0.807 |
| dense | 150 | 0.527 | 0.747 | 0.880 | 0.253 | 0.580 | 0.440 | 0.820 | 0.648 |
| hybrid | 150 | 0.613 | 0.847 | 0.967 | 0.293 | 0.713 | 0.527 | 0.913 | 0.740 |

## Recall@1 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.367 | 0.200 | 0.633 | 0.133 |
| bm25 | 0.883 | 0.200 | 0.633 | 1.000 |
| dense | 0.617 | 0.100 | 0.433 | 0.867 |
| hybrid | 0.683 | 0.133 | 0.600 | 0.967 |

## Recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.783 | 0.633 | 0.900 | 0.467 |
| bm25 | 1.000 | 0.967 | 0.833 | 1.000 |
| dense | 0.967 | 0.633 | 0.833 | 1.000 |
| hybrid | 1.000 | 0.967 | 0.867 | 1.000 |

## Primary recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.067 | 0.000 | 0.033 | 0.467 |
| bm25 | 0.750 | 0.800 | 0.067 | 1.000 |
| dense | 0.567 | 0.567 | 0.200 | 1.000 |
| hybrid | 0.817 | 0.900 | 0.033 | 1.000 |

## Evidence recall@6 by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.317 | 0.100 | 0.533 | 0.000 |
| bm25 | 1.000 | 0.967 | 0.633 | 1.000 |
| dense | 0.950 | 0.567 | 0.633 | 1.000 |
| hybrid | 1.000 | 0.967 | 0.600 | 1.000 |

## MRR by category

| mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.505 | 0.323 | 0.753 | 0.251 |
| bm25 | 0.932 | 0.446 | 0.723 | 1.000 |
| dense | 0.726 | 0.286 | 0.569 | 0.933 |
| hybrid | 0.806 | 0.398 | 0.707 | 0.983 |

## Leakage and latency

| mode | n | Leak | ms mean | ms p95 |
|---|---|---|---|---|
| random | 210 | 0.000 | 0 | 0 |
| bm25 | 210 | 0.000 | 0 | 0 |
| dense | 210 | 0.000 | 17 | 24 |
| hybrid | 210 | 0.000 | 19 | 25 |

Metric key: R@1/R@3/R@k = any gold document in the top 1/3/k chunks (positives only); pR@1/pR@k = the same for the canonical gold document only (structured list, lab panel or quoted note); eR@1/eR@k = a retrieved chunk of a gold document whose own text contains an expected keyword (chunk-level); MRR = mean reciprocal rank of the first gold document; Acc = positives answered without abstaining and containing an expected keyword; Acc|answered = Acc over non-abstained positives; FalseAbst = positives that abstained; CorrectAbst = negatives that abstained; CiteValid = answers whose citations all point at retrieved chunks of the asked patient; Leak = questions with any retrieved chunk of another patient. '-' = not applicable.
