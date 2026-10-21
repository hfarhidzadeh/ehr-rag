# EHR-RAG — Plan

A retrieval-augmented question-answering system over **synthetic** electronic health records.
A clinician-style user picks a patient, asks a question ("What was the most recent HbA1c?",
"Why was metoprolol started?"), and gets an answer grounded in that patient's chart, with a citation to
every supporting source.

> **Data policy.** This project uses only synthetic data produced by `ehr_rag.synth`. Every record carries
> `synthetic: true`. The generation layer refuses to send any context to an external LLM unless every chunk
> is flagged synthetic. Do not load real patient data into this system.

## 1. Decisions (confirmed with user)

| Area | Choice |
|---|---|
| LLM | Claude API (`claude-opus-5-5`, configurable) when credentials exist; **extractive fallback** (no LLM) otherwise |
| Embeddings | Pluggable: `minilm` (sentence-transformers `all-MiniLM-L6-v2`, default) and `tfidf` (TF-IDF → TruncatedSVD, no downloads) |
| Vector store | ChromaDB (persistent, cosine, metadata filter on `patient_id`, `doc_type`) |
| Lexical | BM25 (`rank_bm25`), fused with dense results by Reciprocal Rank Fusion → `hybrid` mode |
| Interfaces | Streamlit web UI + Jupyter walkthrough notebook (a small `python -m ehr_rag` CLI drives both) |
| Runtime | Python 3.11 env at `./.venv` (Intel Mac → PyTorch 2.2.2, `numpy<2`) |

## 2. Architecture

```
synth.generator ──► data/synthetic/*.jsonl  (raw tables + documents.jsonl)
                         │
ingest.phi_guard  ◄──────┤  (blocks non-synthetic / PHI-looking records)
ingest.chunking   ◄──────┘  section-aware chunks with header context
        │
embeddings (minilm | tfidf) ──► vectorstore (Chroma, per-embedder collection)
        │                                  │
        └── BM25 index (per patient) ──────┤
                                           ▼
retrieval: dense | bm25 | hybrid(RRF) | random(control) — ALWAYS filtered to one patient_id
                                           ▼
generation: ClaudeGenerator (cited JSON answer, abstains when evidence missing)
            ExtractiveGenerator (offline: top sentences + citations)
                                           ▼
pipeline.EHRRagPipeline.ask(question, patient_id) → Answer
        ▲                      ▲                       ▲
 Streamlit app          notebook walkthrough     evaluation.runner (gold Q&A + controls)
```

## 3. Synthetic data model

Deterministic (seeded) generator; ~30 patients by default, each with 2–6 chronic conditions drawn from
coherent clinical profiles (T2DM, HTN, CKD, CHF, COPD, asthma, CAD/prior MI, AFib, hypothyroidism,
depression, hyperlipidemia, …).

Raw tables (`data/synthetic/`): `patients`, `encounters`, `conditions`, `medications`, `allergies`,
`labs`, `vitals`, `notes` — all JSONL. Notes (progress note, discharge summary, ED note, radiology report,
consult) are written in clinical style with **abbreviations** (T2DM, HTN, SOB, MI, CKD3) so that
lay-language questions ("heart attack", "high blood pressure") test semantic retrieval.
Facts in the notes agree with the structured tables.

Every source becomes a `Document` (see `types.py`) with a deterministic `doc_id`:

| doc_type | doc_id | content |
|---|---|---|
| `demographics` | `{pid}-demographics` | age, sex, PCP, etc. |
| `problem_list` | `{pid}-problems` | conditions with ICD-10, onset, status |
| `medication_list` | `{pid}-medications` | drug, dose, route, frequency, start/stop, indication |
| `allergy_list` | `{pid}-allergies` | substance, reaction, severity (or NKDA) |
| `lab_panel` | `{pid}-labs-{date}` | one panel per draw date, values + ref ranges + H/L flags |
| `vitals` | `{pid}-vitals-{date}` | BP, HR, temp, SpO2, weight, BMI |
| `clinical_note` | `{note_id}` | sectioned note text (`SECTION:` headers) |

## 4. Retrieval & safety properties

* **Patient isolation is a hard filter**, not a ranking hint: Chroma `where={"patient_id": pid}` and a
  per-patient BM25 index. Cross-patient leakage must be exactly 0 in evaluation.
* Chunk text carries a header (`[Patient P007 | Discharge Summary | 2025-03-14 | Section: Hospital Course]`)
  so the date and source survive into the LLM context.
* The LLM must cite chunk labels (`S1…Sk`). Cited labels are validated against the retrieved set;
  invalid citations are dropped and reported as warnings.
* Abstention: when the record does not contain the answer, the system must say so
  (`abstained=True`) instead of guessing.

## 5. Evaluation with positive and negative controls

Gold questions are generated from the synthetic ground truth (`evaluation/questions.py`):

| Category | Control type | Expected behaviour |
|---|---|---|
| `fact_lookup` | positive | retrieve gold doc; answer contains the value (dose, lab value, allergy…) |
| `temporal` | positive | "most recent X" → the latest panel's value |
| `semantic_synonym` | positive | lay wording vs. abbreviated note text |
| `verbatim` | **positive control** | question quotes chunk text → Recall@1 ≈ 1.0 (pipeline sanity check) |
| `absent_fact` | **negative control** | fact not in this patient's chart → abstain |
| `cross_patient` | **negative control** | another patient's distinctive fact asked under this patient → abstain, 0 leaked chunks |
| `random` retriever | **baseline control** | real retrievers must clearly beat it |

Metrics: Recall@k, MRR, answer accuracy (value/keyword match), abstention rate on negatives,
false-abstention rate on positives, citation validity, leakage rate, latency. Reports go to
`reports/eval_<embedder>_<mode>_<llm>.json` plus a markdown summary.

## 6. Build order

1. Contracts: `pyproject.toml`, `config.py`, `types.py` (this step).
2. `synth` (vocab + generator) ‖ `ingest` (chunking, phi_guard) + `embeddings`.
3. `vectorstore` + `retrieval` ‖ `generation`.
4. `pipeline` + CLI ‖ `evaluation`.
5. Integration: end-to-end ingest + eval for both embedders; fix defects.
6. Streamlit app ‖ notebook (vector-space visualization, controls).
7. Independent review (correctness, healthcare safety, eval validity) → verified fixes.

## 7. Out of scope (for now)

Cohort queries across patients, real FHIR server ingestion, cross-encoder reranking, de-identification
of real notes, auth/audit logging for multi-user deployment.
