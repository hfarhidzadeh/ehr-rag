# EHR-RAG

Retrieval-augmented question answering (RAG) over **synthetic** electronic health records, built as a
teaching and research sandbox. You pick a patient, ask a question ("What is the most recent HbA1c?",
"Why was sertraline 50 mg stopped?") and get an answer that quotes or summarises that patient's chart,
with a citation to every excerpt it used. A gold-question evaluation with positive and negative controls
measures how well each part works.

## Data policy (read first)

* **Synthetic data only.** Every record comes from `ehr_rag.synth` (fictional names, `SYN-` MRNs, fixed seed)
  and carries `synthetic: true`. Indexing refuses any document without that flag, and a PHI tripwire
  (SSN, phone, email, URL, street address, non-`SYN-` MRN) blocks indexing if anything PHI-like appears.
* **Never load real patient data** into this system, not even "just to try it". It has no de-identification,
  access control or audit logging, and the Claude generator sends excerpts to an external API.
* **AI output must be checked by a person.** The answers come from heuristics or an LLM and can be wrong
  (see the measured failure rates below). Use the controls to judge whether a run behaved properly, and
  read the cited excerpts rather than trusting the answer text.

## What RAG means here, in one paragraph

An LLM answering from memory can invent facts. RAG first **retrieves** a few relevant excerpts (chunks) of
the patient's chart, then asks a **generator** to answer *only* from those excerpts and to cite them
(`[S1]`, `[S2]`, ...). So answer quality depends on two separate steps: did retrieval find the right
excerpt (measured by recall), and did the generator use it correctly (measured by answer accuracy,
abstention and citation validity)? This project measures both steps separately.

## Setup

Python 3.11 in a project-local environment (`.venv`, or a conda env if you prefer):

```bash
# either a venv ...
python3.11 -m venv .venv && source .venv/bin/activate
# ... or conda
conda create -n ehr-rag python=3.11 && conda activate ehr-rag

pip install -e ".[minilm,app,notebook,dev]"
```

Why some versions are pinned (`pyproject.toml`, full lock in `requirements.lock.txt`): **PyTorch 2.2.2 is the
last release with wheels for Intel (x86_64) Macs**, so `torch==2.2.2` is pinned. `sentence-transformers==3.4.1`
and `transformers==4.46.3` are the last versions that work with that torch, and `numpy<2` is required because the
torch 2.2 wheels were built against NumPy 1.x. On Apple Silicon or Linux you could use newer versions, but then
your numbers may differ slightly from the tables below. The `tfidf` embedder needs none of this: it never
imports torch. The MiniLM model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) is downloaded once on the
first `ingest --embedder minilm`.

Run the tests: `python -m pytest -q` (fast, ~40 s). Add `-m 'slow or not slow'` to include the MiniLM tests.

## Quickstart

```bash
python -m ehr_rag generate                      # 30 synthetic patients, seed 42 -> data/synthetic/
python -m ehr_rag ingest --embedder tfidf       # chunk + embed + index -> data/index/tfidf/   (~25 s)
python -m ehr_rag ingest --embedder minilm      # same with MiniLM      -> data/index/minilm/  (a few min on CPU)

python -m ehr_rag ask "What is the most recent HbA1c?" --patient P003 --embedder tfidf --llm extractive
python -m ehr_rag ask "Why was sertraline 50 mg stopped?" -p P001 --show-retrieved
python -m ehr_rag ask "allergies" -p P002 --json            # full Answer object as JSON

python -m ehr_rag eval --embedder tfidf --compare             # random/bm25/dense/hybrid, retrieval only
python -m ehr_rag eval --embedder tfidf --mode hybrid --llm extractive   # full eval incl. answers
python -m ehr_rag eval --embedder minilm --vocab-expansions   # separate "vocab-expanded" configuration
python -m ehr_rag eval --retrieval-only --limit 20            # quick retrieval check

python -m ehr_rag app                                         # Streamlit UI (app/streamlit_app.py)
jupyter notebook notebooks/                                   # walkthrough notebook
```

Global options go *before* the command: `python -m ehr_rag --data-dir X --index-dir Y --reports-dir Z -v ask ...`.
Every setting can also be set with an `EHR_RAG_<FIELD>` environment variable (see `src/ehr_rag/config.py`),
e.g. `EHR_RAG_EMBEDDER=tfidf`, `EHR_RAG_TOP_K=8`, `EHR_RAG_LLM=extractive`. CLI exit codes: 0 ok, 1 data or
index missing (the message says how to build it), 2 unknown patient, 3 PHI or non-synthetic data refused.

The web UI is `app/streamlit_app.py` (tabs: Ask, Vector space, Evaluation). The walkthrough notebook is
`notebooks/01_ehr_rag_walkthrough.ipynb`; edit `notebooks/build_walkthrough.py` and regenerate rather than editing
the `.ipynb` by hand:

```bash
.venv/bin/python notebooks/build_walkthrough.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_ehr_rag_walkthrough.ipynb
```

## Architecture

```
synth.generator ──► data/synthetic/*.jsonl  (raw tables + documents.jsonl + eval_questions.jsonl)
                         │
ingest.phi_guard  ◄──────┤  refuses non-synthetic or PHI-looking documents (nothing is indexed)
ingest.chunking   ◄──────┘  notes split by SECTION:, structured docs by line; every chunk starts with
        │                   "[Patient P007 | Discharge Summary | 2025-03-14 | Section: Hospital Course]"
        ▼
embeddings (tfidf | minilm) ──► vectorstore: Chroma, one collection per embedder, cosine, patient_id filter
        │                                   │
        └── BM25 index (one per patient) ───┤
                                            ▼
retrieval: dense | bm25 | hybrid (RRF) | random (control)   ── ALWAYS restricted to one patient_id
                                            ▼
generation: ClaudeGenerator     (JSON answer + cited labels; abstains when evidence is missing)
            ExtractiveGenerator (offline: quotes matching sentences verbatim; abstains when terms are missing)
                                            ▼
pipeline.EHRRagPipeline.ask(question, patient_id) -> Answer(text, citations, retrieved, abstained, ...)
        ▲                         ▲                          ▲
  Streamlit app            notebook walkthrough       evaluation.runner (gold questions + controls -> reports/)
```

Synthetic chart per patient: demographics, problem list (ICD-10, onset), medication list (active and
discontinued meds with reasons), allergy list, one lab panel per draw date, vital signs and 4-10 clinical
notes (progress, consult, ED, discharge, radiology). Notes use clinical abbreviations (T2DM, HTN, s/p MI,
CKD3a, A1c) and never the lay terms ("heart attack", "high blood pressure"), so lay-language questions test
semantic matching. Index: 623 documents -> 2,210 chunks for 30 patients.

## Retrieval modes

| mode | how it ranks the patient's chunks | good at | weak at |
|---|---|---|---|
| `bm25` | word overlap (BM25 Okapi) on a tokenizer that keeps `7.2`, `2025-07-10`, drug names | exact names, values, quotes | lay wording ("heart attack" vs "s/p MI") |
| `dense` | cosine similarity of embeddings: `tfidf` (TF-IDF -> 256-d SVD, fitted on this corpus) or `minilm` (pretrained 384-d sentence model) | paraphrase (MiniLM only) | exact quotes; tfidf is basically lexical too |
| `hybrid` | Reciprocal Rank Fusion of the top-30 of both: score = sum 1/(60 + rank) | usually the best overall | a weak or misleading retriever can outvote a strong one |
| `random` | deterministic random sample of the patient's chunks (seeded by the query) | nothing: it is the **baseline control** | everything |

**Patient isolation** is a hard filter, not a ranking hint: Chroma `where={"patient_id": pid}`, one BM25 index
per patient, a re-check of every returned chunk (wrong-patient chunks are dropped and counted), and
generators that refuse contexts of another patient.

## Evaluation and controls

`evaluation/questions.py` builds 210 gold questions from the synthetic ground truth, 7 per patient. The
expected answers are computed from the raw tables; they are never written by hand or by an LLM:

| category | n | control type | example | expected behaviour |
|---|---|---|---|---|
| `fact_lookup` | 60 | positive | "What reaction does the patient have to ibuprofen?" | retrieve the allergy list / notes; answer contains "bronchospasm" |
| `temporal` | 30 | positive | "What is the most recent creatinine result?" | latest panel's value only |
| `semantic_synonym` | 30 | positive | "What medications is the patient taking for their heartburn?" | lay term vs chart abbreviation (GERD) |
| `verbatim` | 30 | **positive control** | 'Where in the chart does it say: "..."?' | the quoted chunk is retrieved first (pipeline sanity check) |
| `absent_fact` | 30 | **negative control** | "What dose of famotidine is the patient currently taking?" (not in chart) | abstain |
| `cross_patient` | 30 | **negative control** | another patient's distinctive drug/allergen | abstain, and 0 chunks from another patient |
| `random` retriever | - | **baseline control** | - | real retrievers must clearly beat it |

Metrics (`evaluation/metrics.py`; k = 6):

* **R@1 / R@6** - a gold *document* is among the top 1 / 6 retrieved chunks.
* **primary R@6** - the same, for the canonical document only (the structured list, lab panel or quoted note).
* **evidence R@1 / R@6** - stricter, chunk-level: the retrieved chunk is from a gold document *and its own text
  contains an expected keyword*. Gold is document-level, so plain R@k also credits, e.g., the PHYSICAL EXAM chunk
  of a note whose plan holds the answer; evidence recall does not. Use it to compare retrievers.
* **MRR** - mean of 1/rank of the first gold document.
* **answer accuracy** - positives answered (not abstained) whose text contains an expected keyword;
  **false abstention** - positives that abstained; **correct abstention** - negatives that abstained;
  **citation validity** - every citation points at a retrieved chunk of the asked patient; **leakage** -
  questions with any retrieved chunk of another patient (must be 0).

Control checks printed as PASS/FAIL in every report: verbatim R@1 >= 0.9 for `bm25` and `hybrid`;
leakage == 0 for every mode; `random` R@6, primary R@6 and evidence R@6 at least 0.2 below the best real
retriever; citation validity == 1; correct-abstention and false-abstention rates reported. The checks have their
own negative controls in `tests/test_runner.py`: a leaking retriever, mislabelled citations and a "random"
retriever as good as the real one each turn a check to FAIL.

## Results (measured 2026-10-21, seed 42, 30 patients, 210 questions, k = 6)

Reports: `reports/compare_<embedder>.{md,json}` and `reports/eval_<embedder>_hybrid_<generator>.{md,json}`.
These are numbers on a **synthetic, templated benchmark** built by the same project; they say nothing about
clinical performance.

### Retrieval (positives, n = 150)

TF-IDF embedder (`reports/compare_tfidf.md`):

| mode | R@1 | R@6 | primary R@6 | evidence R@1 | evidence R@6 | MRR | verbatim R@1 | leakage | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.34 | 0.71 | 0.13 | 0.03 | 0.25 | 0.47 | 0.13 | 0.00 | <1 |
| bm25 | **0.72** | 0.96 | 0.67 | **0.68** | **0.92** | **0.81** | **1.00** | 0.00 | <1 |
| dense | 0.53 | 0.88 | 0.58 | 0.44 | 0.82 | 0.65 | 0.87 | 0.00 | 17 |
| hybrid | 0.61 | **0.97** | **0.71** | 0.53 | 0.91 | 0.74 | 0.97 | 0.00 | 16 |

MiniLM embedder (`reports/compare_minilm.md`; `random` and `bm25` do not use the embedder, so they match the
table above):

| mode | R@1 | R@6 | primary R@6 | evidence R@1 | evidence R@6 | MRR | verbatim R@1 | leakage | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.34 | 0.71 | 0.13 | 0.03 | 0.25 | 0.47 | 0.13 | 0.00 | <1 |
| bm25 | **0.72** | 0.96 | 0.67 | **0.68** | 0.92 | **0.81** | **1.00** | 0.00 | <1 |
| dense | 0.61 | 0.93 | 0.58 | 0.48 | 0.87 | 0.72 | 0.47 | 0.00 | 40 |
| hybrid | 0.70 | **0.98** | 0.67 | 0.65 | **0.95** | 0.80 | **0.80 (FAIL)** | 0.00 | 20 |

Evidence R@6 by category:

| embedder / mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.32 | 0.10 | 0.53 | 0.00 |
| bm25 | 1.00 | 0.97 | 0.63 | 1.00 |
| tfidf dense | 0.95 | 0.57 | 0.63 | 1.00 |
| tfidf hybrid | 1.00 | 0.97 | 0.60 | 1.00 |
| minilm dense | 0.95 | 0.90 | 0.73 | 0.80 |
| minilm hybrid | 1.00 | 0.97 | **0.83** | 0.93 |

R@1 by category (how often the *first* chunk is right):

| embedder / mode | fact_lookup | temporal | semantic_synonym | verbatim |
|---|---|---|---|---|
| random | 0.37 | 0.20 | 0.63 | 0.13 |
| bm25 | 0.88 | 0.20 | 0.63 | 1.00 |
| tfidf hybrid | 0.68 | 0.13 | 0.60 | 0.97 |
| minilm dense | 0.72 | 0.37 | 0.77 | 0.47 |
| minilm hybrid | 0.85 | 0.23 | 0.77 | 0.80 |

Control checks: leakage 0 in every run (0 wrong-patient chunks dropped by the guard); the random baseline is
well below the real retrievers (evidence R@6 0.25 vs 0.95); verbatim R@1 passes for `bm25` (1.00, both
embedders) and for tfidf `hybrid` (0.97), and **fails for MiniLM `hybrid` (0.80 < 0.90)**.

### Answer generation (full eval, hybrid retrieval, all 210 questions)

No Claude evaluation was run for these tables (no API key in the build environment), so every answer below
comes from the offline extractive generator. `-vocab` is the separate configuration in which the extractive
generator gets lay-term -> abbreviation expansions from the same vocabulary that the semantic questions were
built from, so read it as an optimistic upper bound, not a fair score.

| embedder | generator | accuracy (positives) | accuracy when answered | false abstention | correct abstention (negatives) | fact_lookup | temporal | semantic | verbatim | citation validity | leakage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf | extractive | 0.79 | 0.98 | 0.19 | 1.00 | 1.00 | 0.97 | 0.00 | 1.00 | 1.00 | 0.00 |
| tfidf | extractive-vocab | 0.87 | 0.96 | 0.09 | 1.00 | 1.00 | 0.97 | 0.40 | 1.00 | 1.00 | 0.00 |
| minilm | extractive | 0.78 | 0.97 | 0.20 | 1.00 | 1.00 | 0.93 | 0.00 | 0.97 | 1.00 | 0.00 |
| minilm | extractive-vocab | 0.89 | 0.95 | 0.06 | 1.00 | 1.00 | 0.93 | 0.57 | 0.97 | 1.00 | 0.00 |

Runtime on a 4-core Intel Mac: a full 210-question eval takes 6-12 s, a 4-mode comparison 7-13 s, about
20-40 ms per question. Building the tfidf index takes about 25 s and the MiniLM index a few minutes of CPU. The
MiniLM build took 34 min of wall-clock time here because the machine was swapping.

### How to read these numbers (honest interpretation)

* **BM25 is the strongest single retriever here.** Questions and chart text share exact tokens (drug names,
  lab names, values), because both come from the same templates. Real clinical text is messier, so do not
  expect this ranking to transfer.
* **Plain R@6 flatters the random baseline (0.71).** Medications and allergies are repeated in every note, and
  "has condition X" questions have many gold notes, so a random chunk often belongs to *some* gold document.
  Evidence R@6 (random 0.25) is the fairer comparison. This is why that metric was added.
* **MiniLM helps on meaning, hurts on exact text.** MiniLM dense is best at lay-language questions (semantic
  evidence R@6 0.73 vs 0.63 for BM25; hybrid 0.83) and at "most recent" questions. But it often ranks a quoted
  sentence's chunk below chunks *about* the same event (verbatim R@1 0.47; 0.53 even with the bare sentence
  as query). Equal-weight RRF then lets dense outvote BM25's clear exact match, and **MiniLM hybrid fails the
  verbatim control (0.80)**. We did not re-weight the fusion to make this pass: tuning weights on the same 210
  questions would just fit the evaluation. A held-out question set would be needed for that.
* **"Most recent" is not something retrievers understand.** Temporal R@1 is 0.13-0.37 in every mode, because
  similarity has no notion of date. The answers are still mostly right (0.93-0.97): the extractive generator
  picks the most recently dated excerpt, *provided the latest panel is among the 6 retrieved chunks*. The
  misses (P015 sodium, P012 WBC) are cases where it is not, and then the answer reports an older value as
  "most recent in the retrieved excerpts".
* **The extractive generator cannot do semantic questions** (0.00): it is purely lexical and abstains when the lay
  term is not in the excerpts, which is the safe failure. With vocab expansions it reaches 0.40-0.57. The rest
  fail because note MEDICATIONS sections list drugs without the condition, and because of lexical false leads
  ("smoker's lung" matches "former smoker" in a radiology report).
* **Correct abstention 1.00 on negatives is by construction for the extractive generator.** The negative
  questions name a drug or allergen that appears nowhere in the patient's chart, and the extractive generator
  abstains whenever a key term is missing. This shows the system does not invent an answer from another
  patient's data, but it is not evidence of reasoning. These controls become informative with Claude.
* **Keyword accuracy is lenient.** An answer counts as correct when an expected keyword (e.g. "20 mg") appears
  anywhere in it, and the extractive generator quotes up to 3 excerpts. We spot-checked about 15 answers by hand
  and they were right, but an answer that lists several values can be credited when one of them is the
  expected one. Read the worst-examples section of each report.

### Defects found and fixed during integration (before -> after, tfidf hybrid + extractive)

| change | why | effect |
|---|---|---|
| extractive: "reaction(s)" treated as a generic question word | allergy records read "ibuprofen — bronchospasm (severe)" and never contain the word "reaction", so all 10 allergy questions abstained | fact_lookup accuracy 0.83 -> 1.00; overall accuracy 0.73 -> 0.79; false abstention 0.27 -> 0.19 |
| extractive "most recent": on equal dates prefer a value line over a lab panel's title line, and a sentence with a number over prose | a split panel's title line ("panels: BMP, HbA1c") or a note line ("...albumin-to-creatinine ratio...") was quoted instead of the value | minilm temporal 0.90 -> 0.93, overall 0.767 -> 0.780 (together with the next row) |
| extractive: a note line ending in ":" ("Home medications:", "Drawn today (date):") quotes the lines it introduces; "say" and "first/diagnosed" treated as question words | the lead-in alone carried no answer; "Where does it say" made "say" a required term | small gains in the vocab configuration (tfidf 0.85 -> 0.87) |
| question builder: verbatim near-duplicate filter keeps ISO dates as one token | "Last seen 2026-01-24; no ED visits or hospitalizations since." differed from other notes only by its date, but the date split into 3 tokens hid the near-duplicate. 8/30 verbatim questions changed (3 such sentences plus 5 re-draws from the changed candidate pool) | verbatim R@1 bm25 0.93 -> 1.00, tfidf hybrid 0.93 -> 0.97. The old set already passed the control |
| metrics: chunk-level evidence recall added | document-level recall credits unrelated chunks of a gold note | new column; stricter, not more lenient |
| CLI: `eval` reads the runner's report structure; `config.with_overrides` turns str paths into `Path` | `eval` previously had no runner; `load_settings(index_dir="x")` left a str | - |

No question was removed or reworded because the system failed it, and no control threshold was changed.

## Using Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...            # or ANTHROPIC_AUTH_TOKEN
python -m ehr_rag ask "What is the most recent HbA1c?" -p P003 --llm claude
python -m ehr_rag eval --embedder minilm --llm claude          # ~210 API requests: this costs money
```

* `--llm auto` (the default) uses Claude when credentials exist, otherwise the extractive generator.
* Model and effort: `EHR_RAG_CLAUDE_MODEL` (default `claude-opus-5-5`), `EHR_RAG_CLAUDE_EFFORT`
  (`low|medium|high|xhigh|max`, default `medium`), `EHR_RAG_CLAUDE_MAX_TOKENS`.
* Server-side refusal fallback is **on by default**: a declined request is retried on Anthropic's recommended
  fallback model, and the answer's warnings say so. Turn it off with `EHR_RAG_CLAUDE_FALLBACKS=false`.
* Every excerpt is checked for `synthetic: true` before any network call. Claude must answer in JSON
  `{answer, cited_labels, insufficient_evidence}`, invalid citation labels are dropped with a warning, and a
  refusal becomes an abstention. If the API call fails, the pipeline falls back to the extractive generator and
  records a warning; the report counts these fallbacks (`llm_fallbacks_to_extractive`).
* A full Claude eval sends about 210 requests (one per question) with up to 6 excerpts each. Try
  `--limit 20` first.

## Known limitations

* Synthetic, templated data: results are optimistic for lexical methods and are not clinical validation.
* No recency awareness in retrieval; "most recent" works only if the latest document is retrieved.
* MiniLM truncates input at 256 word pieces, so 78 of 2,210 chunks (67 lab panels, 7 medication lists, 4 notes;
  max 416 pieces) have tails that the dense vector never sees. BM25 still sees the whole chunk.
* RRF fusion uses equal weights and is not tuned. MiniLM hybrid fails the verbatim control.
* The extractive generator is lexical: no lay-term understanding without expansions, no arithmetic, no
  comparison. Its "most recent" means the latest date among the retrieved excerpts.
* Answer accuracy is keyword-based and lenient (see above); gold recall is document-level (use evidence recall).
* The evaluation has no held-out split: do not tune parameters on it and then quote the same numbers.
* Single-patient questions only (no cohort queries); no auth, audit log or de-identification. Not for real data.

## Project layout

```
pyproject.toml, requirements.lock.txt   dependencies (torch/transformers pinned for Intel Macs)
src/ehr_rag/
  config.py           Settings + EHR_RAG_* environment overrides
  types.py            Document, Chunk, RetrievedChunk, Citation, Answer, EvalQuestion, doc-id helpers
  synth/vocab.py      clinical vocabulary (conditions, lay terms, meds, labs, allergens, note phrases)
  synth/generator.py  deterministic synthetic patients, tables, notes, documents
  ingest/phi_guard.py synthetic-flag gate + PHI tripwire
  ingest/chunking.py  section-aware chunking with header lines
  embeddings.py       TfidfSvdEmbedder, MiniLMEmbedder (lazy torch)
  vectorstore.py      ChromaStore (persistent, cosine, patient filter)
  retrieval.py        tokenizer, BM25Index, RRF, Retriever (dense|bm25|hybrid|random)
  generation.py       ClaudeGenerator, ExtractiveGenerator
  pipeline.py         EHRRagPipeline: build_index / load / retrieve / ask / patient_summary
  evaluation/questions.py  gold questions from ground truth (+ validation)
  evaluation/metrics.py    recall, evidence recall, MRR, accuracy, abstention, citations, leakage
  evaluation/runner.py     run_eval, compare_retrievers, write_report (+ control checks)
  cli.py, __main__.py      python -m ehr_rag ...
  app_support.py      helpers for the Streamlit app (projections, report tables, example questions)
app/streamlit_app.py  web UI: Ask / Vector space / Evaluation tabs
notebooks/            01_ehr_rag_walkthrough.ipynb + build_walkthrough.py (regenerates it)
tests/                pytest suite (fast by default; -m 'slow or not slow' for MiniLM)
data/synthetic/       generated dataset + eval_questions.jsonl (gitignored)
data/index/<embedder>/ chunks.jsonl, embedder file, chroma/, index_manifest.json (gitignored)
reports/              evaluation reports (*.md tracked, *.json gitignored)
```
