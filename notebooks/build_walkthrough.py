"""Build ``notebooks/01_ehr_rag_walkthrough.ipynb`` from the cell sources below.

Regenerate and execute (from the project root)::

    .venv/bin/python notebooks/build_walkthrough.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_ehr_rag_walkthrough.ipynb

Edit the cells here, not in the notebook, so the notebook can always be rebuilt from source.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK = Path(__file__).resolve().parent / "01_ehr_rag_walkthrough.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", dedent(text).strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", dedent(text).strip("\n")))


# =========================================================================== 0. policy
md("""
# EHR-RAG walkthrough: retrieval-augmented QA over synthetic health records

This notebook walks through every stage of the EHR-RAG pipeline, from the raw synthetic tables to the
evaluation report. It is written for lab members who are learning how RAG works. Each step has a short
explanation, a small piece of code you can change, and at least one **control**: a check whose correct
result we know in advance.

**Contents**
0. Data policy and why we use controls
1. Setup and settings
2. Synthetic data
3. Documents and chunking
4. PHI guard
5. Embeddings: do lay words land near clinical abbreviations?
6. The vector space (PCA and t-SNE)
7. Retrieval: four retrievers side by side, and patient isolation
8. Generation: extractive answers, Claude (optional), abstention
9. Evaluation: reports, recall per category, control checks
10. Discussion questions for the lab
""")

md("""
## 0. Data policy, and why we use positive and negative controls

**Synthetic data only.** Every record in this project comes from `ehr_rag.synth`: the names are made up, the
MRNs start with `SYN-`, and every record carries `synthetic: true`. The pipeline refuses to index any
document without that flag. **Never load real patient data into this notebook or this system**, not even to
"just try it". It has no de-identification, access control or audit log, and the optional Claude step
sends excerpts to an external API.

**Why controls?** A RAG system will always give you *an* answer, and an answer that sounds fluent can still
be wrong. We judge AI output the same way we judge an assay: we run it on cases where we already know
the right result.

* A **positive control** is a case where the system *should* find or flag something. Examples: a quoted
  sentence must be retrieved first; a fake SSN must be flagged. If a positive control fails, the step
  is broken, and you should not trust its other results.
* A **negative control** is a case where the system *should not* find anything. Examples: a drug the
  patient never took must lead to "not documented"; another patient's chunks must never appear. If a
  negative control fails, the system is producing false positives, such as hallucinated answers or leaked
  data.
* A **baseline control** (here, a random retriever) tells you what "no skill" looks like. A real method
  must clearly beat it.

Every control in this notebook is recorded with `control(...)` and printed as `[PASS]` or `[FAIL]`. A summary
table at the end lists them all. A FAIL does not stop the notebook. It is a result you need to look at.

Some cells deliberately feed the system bad input (an unflagged document, a leaking store), so you will see
red `ERROR:ehr_rag...` log lines in sections 4, 7 and 9. Those lines are **expected**: they are the safety
guards firing, and the control next to them checks that they did.

The controls tell you whether the machinery behaved. They do **not** tell you whether an answer is clinically
right. That judgement stays with a person who reads the cited excerpts.
""")

# =========================================================================== 1. setup
md("""
## 1. Setup and settings

We import the package, quiet the logs, and load the settings. Every setting can be overridden with an
`EHR_RAG_<FIELD>` environment variable (for example `EHR_RAG_TOP_K=8`). We fix `llm="extractive"`, so
the notebook never calls an external API unless you explicitly run the Claude cell in section 8.

The indexes are reused when present (`data/index/tfidf/`, `data/index/minilm/`). If the TF-IDF index is
missing, it is built here (about 25 s). The MiniLM index is optional, because building it takes several minutes. To
build it, run `python -m ehr_rag ingest --embedder minilm` in a terminal.
""")

code("""
import json
import logging
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # never download in the notebook; use the cached model
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("ehr_rag").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display, Markdown

# Explicit inline backend: an inherited MPLBACKEND=Agg (Streamlit sets it on import) would hide every figure.
get_ipython().run_line_magic("matplotlib", "inline")

from ehr_rag.config import PROJECT_ROOT, load_settings
from ehr_rag.pipeline import EHRRagPipeline, IndexNotBuiltError

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

# ---- the control registry used throughout the notebook
CONTROLS: list[dict] = []

def control(section: str, name: str, kind: str, expected, observed, passed: bool) -> None:
    \"\"\"Record and print one control check (kind: positive | negative | baseline).\"\"\"
    status = "PASS" if passed else "FAIL"
    CONTROLS.append({"section": section, "control": name, "type": kind,
                     "expected": str(expected), "observed": str(observed), "status": status})
    print(f"[{status}] ({kind} control) {name}\\n        expected: {expected}\\n        observed: {observed}")

settings = load_settings(llm="extractive")
print("project root:", PROJECT_ROOT)
pd.Series({k: getattr(settings, k) for k in (
    "data_dir", "index_dir", "reports_dir", "seed", "n_patients", "chunk_max_chars",
    "chunk_overlap_sentences", "top_k", "rrf_k", "candidate_pool", "retrieval_mode", "llm", "claude_model")},
    name="value").to_frame()
""")

md("""
**Figure style.** The charts below use one fixed colour per retriever across the whole notebook:
**random** is gray because it is the baseline, **bm25** is blue, **dense** is orange and **hybrid** is aqua.
Scatter plots highlight one group at a time against gray instead of using 30 colours.
""")

code("""
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # categorical slots 1-4 (fixed order)
BACKGROUND_POINTS = "#c3c2b7"
MODE_COLORS = {"random": "#898781", "bm25": SERIES[0], "dense": SERIES[1], "hybrid": SERIES[2]}
MODES = ("random", "bm25", "dense", "hybrid")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "font.family": "sans-serif", "figure.dpi": 110, "lines.linewidth": 2,
})
""")

code("""
from ehr_rag.synth.generator import generate_dataset

# Generate the synthetic dataset only if it is missing (deterministic: same seed -> same data).
if not (settings.data_dir / "documents.jsonl").exists():
    print("No synthetic data found; generating", settings.n_patients, "patients with seed", settings.seed)
    generate_dataset(n_patients=settings.n_patients, seed=settings.seed).write(settings.data_dir)

PIPES: dict[str, EHRRagPipeline] = {}
t0 = time.perf_counter()
try:
    PIPES["tfidf"] = EHRRagPipeline.load(settings.with_overrides(embedder="tfidf"))
except IndexNotBuiltError as exc:
    print("TF-IDF index not usable, building it now (~25 s):", exc)
    PIPES["tfidf"] = EHRRagPipeline.build_index(settings.with_overrides(embedder="tfidf"))
try:
    PIPES["minilm"] = EHRRagPipeline.load(settings.with_overrides(embedder="minilm"))
except Exception as exc:  # index missing, or model not in the local cache
    print("MiniLM not available, continuing with TF-IDF only:", type(exc).__name__, str(exc)[:300])

HAVE_MINILM = "minilm" in PIPES
MAIN = "minilm" if HAVE_MINILM else "tfidf"     # the embedder used for the main examples
pipe = PIPES[MAIN]
print(f"loaded {list(PIPES)} in {time.perf_counter() - t0:.1f} s; main embedder = {MAIN}")
pd.DataFrame({name: {k: p.manifest.get(k) for k in ("embedder", "model", "dim", "n_docs", "n_chunks",
                                                    "n_patients", "data_seed", "created_at")}
              for name, p in PIPES.items()})
""")

# =========================================================================== 2. data
md("""
## 2. Synthetic data

The generator writes eight raw tables as JSON Lines: one record per line. These tables are the
**ground truth**. The evaluation questions and their expected answers are computed from them, never
written by hand or by an LLM.

Below we load every table with pandas, check that every record is flagged synthetic, and then look at one
patient.
""")

code("""
from ehr_rag.synth.generator import TABLES

# convert_dates=False keeps dates as ISO strings ("2025-07-10"), as in the raw files and the chunks.
tables = {name: pd.read_json(settings.data_dir / f"{name}.jsonl", lines=True, convert_dates=False,
                             keep_default_dates=False) for name in TABLES}
manifest = json.loads((settings.data_dir / "manifest.json").read_text())
display(pd.DataFrame({"rows": {n: len(t) for n, t in tables.items()}}).T)

all_flagged = all(bool(t["synthetic"].all()) for t in tables.values())
control("2 data", "every raw record carries synthetic=True", "positive", "True for all 8 tables",
        all_flagged, all_flagged)
""")

code("""
PID = "P003"   # change me: any id from P001 to P030
summary = pipe.patient_summary(PID)
demo = summary["demographics"]
print(f"{PID}: {demo['name']} ({demo['sex']}, {demo['age']} y), MRN {demo['mrn']}, PCP {demo['pcp']}")
print("allergies:", "NKDA" if summary["nkda"] else
      "; ".join(f"{a['substance']} ({a['reaction']}, {a['severity']})" for a in summary["allergies"]))

cond = tables["conditions"].query("patient_id == @PID")
display(cond[["abbreviation", "display", "icd10", "onset_date", "status"]].sort_values("onset_date"))

meds = tables["medications"].query("patient_id == @PID")
display(meds[["med_id", "name", "dose", "frequency", "indication", "start_date", "end_date", "status",
              "discontinue_reason"]].sort_values(["status", "start_date"]))
""")

code("""
labs = tables["labs"].query("patient_id == @PID")
# One row per test, one column per draw date: the "trend" view clinicians use.
trend_table = labs.pivot_table(index="name", columns="date", values="value", aggfunc="first")
display(trend_table)

# Pick the most often repeated test for this patient (HbA1c for P003).
counts = labs.groupby("test_key")["date"].nunique().sort_values(ascending=False)
LAB = "hba1c" if counts.get("hba1c", 0) >= 2 else counts.index[0]
series = labs.query("test_key == @LAB").sort_values("date")
print(f"plotting {LAB}: {len(series)} draws")
""")

code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
dates = pd.to_datetime(series["date"])
lo, hi = series["ref_low"].iloc[0], series["ref_high"].iloc[0]
if pd.notna(lo) and pd.notna(hi):
    ax.axhspan(lo, hi, color="#cde2fb", alpha=0.6, lw=0, label=f"reference range {lo:g}-{hi:g}")
ax.plot(dates, series["value"], color=SERIES[0], marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label=series["name"].iloc[0])
for i in (0, len(series) - 1):   # direct-label the first and last values only
    ax.annotate(f"{series['value'].iloc[i]:g} {series['unit'].iloc[i]}", (dates.iloc[i], series["value"].iloc[i]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
ax.set_title(f"{series['name'].iloc[0]} over time, patient {PID} (synthetic)", loc="left", pad=26)
ax.set_ylabel(series["unit"].iloc[0])
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.margins(y=0.25)
fig.autofmt_xdate()
plt.show()
""")

md("""
*Look at it:* is the trend what you would expect for this patient's problems and medications? This
plot comes straight from the table and involves no AI. Later, when the system answers "What is the most recent
HbA1c?", you can check the answer against the last point here.
""")

# =========================================================================== 3. chunking
md("""
## 3. Documents and chunking

Each table is rendered into **documents**: one per note, one per lab panel, one medication list, and so on.
Documents are split into **chunks**, the units the retriever ranks. Two rules matter:

* Notes are split at their section headers (`ASSESSMENT AND PLAN:`), and long sections are split at sentence
  boundaries. Structured lists are split between lines, never inside a line.
* Every chunk starts with a **header line** such as `[Patient P001 | Primary Care Progress Note | 2025-07-10 |
  Section: Assessment and Plan]`. That way the patient, source and date are still attached to the text when the
  chunk is shown to the generator out of context.
""")

code("""
from ehr_rag.synth.generator import load_documents
from ehr_rag.ingest.chunking import chunk_document

documents = load_documents(settings.data_dir)
docs_by_id = {d.doc_id: d for d in documents}
display(pd.Series([d.doc_type for d in documents]).value_counts().rename("documents").to_frame())

note = docs_by_id["P001-N04"]
print(f"{note.doc_id} | {note.title} | {note.date} | {len(note.text)} characters\\n")
print(note.text)
""")

code("""
note_chunks = chunk_document(note, max_chars=settings.chunk_max_chars,
                             overlap_sentences=settings.chunk_overlap_sentences)
display(pd.DataFrame([{"chunk_id": c.chunk_id, "section": c.section, "chars": len(c.text),
                       "first line (header)": c.text.splitlines()[0]} for c in note_chunks]))
for c in note_chunks[-2:]:
    print("-" * 100)
    print(c.text)
""")

code("""
# Controls for the chunker.
indexed = [c for c in pipe.chunks if c.doc_id == note.doc_id]
same = [c.to_dict() for c in note_chunks] == [c.to_dict() for c in indexed]
control("3 chunking", "re-chunking the note reproduces the indexed chunks exactly", "positive",
        f"{len(indexed)} identical chunks", f"{len(note_chunks)} chunks, identical={same}", same)

headers_ok = all(c.text.startswith(f"[Patient {c.patient_id} | ") for c in pipe.chunks)
control("3 chunking", "every indexed chunk starts with its own patient header", "positive",
        f"{len(pipe.chunks)} / {len(pipe.chunks)}",
        f"{sum(c.text.startswith(f'[Patient {c.patient_id} | ') for c in pipe.chunks)} / {len(pipe.chunks)}",
        headers_ok)

# A structured list that needed more than one chunk: the summary line is repeated on each continuation.
multi = [d for d in documents if d.doc_type in ("medication_list", "lab_panel")
         and len([c for c in pipe.chunks if c.doc_id == d.doc_id]) > 1]
if multi:
    for c in [c for c in pipe.chunks if c.doc_id == multi[0].doc_id][:2]:
        print("-" * 100)
        print(c.text)
""")

md("""
*Things to notice:* the overlap sentence carried over between chunks of a long section, and how a structured list
repeats its "as of" line on each continuation chunk. Ask yourself: would *you* be able to answer a question
from one chunk alone, without the rest of the note?
""")

# =========================================================================== 4. PHI guard
md("""
## 4. PHI guard: a tripwire, tested with both kinds of control

Before anything is indexed, two gates run. `assert_synthetic` refuses any document without
`synthetic: true`. `scan_phi` looks for things that resemble protected health information: SSNs, phone
numbers, e-mail addresses, URLs, street addresses and MRNs that are not `SYN-` MRNs. If anything is found,
the whole index build is refused.

A tripwire that never fires looks exactly like a tripwire that is broken. So we test it in both directions:
* **Positive control.** A string we made up, containing a fake SSN (`123-45-6789`), a fictional `555-01xx`
  phone number, an `example.com` e-mail, an invented street address and a non-synthetic MRN, *must* be
  flagged.
* **Negative control.** The synthetic documents, which are full of numbers that are *not* PHI (dates, lab
  values, blood pressures, LOINC codes, `SYN-` MRNs), must produce *no* findings.
""")

code("""
from ehr_rag.ingest.phi_guard import scan_phi, check_documents, assert_synthetic, NonSyntheticDataError
from ehr_rag.types import Document

fake = ("MADE-UP TEST STRING, NOT A REAL PERSON: SSN 123-45-6789, call (713) 555-0142, "
        "email jane.doe@example.com, lives at 42 Imaginary Lane, MRN: 00482913.")
findings = scan_phi(fake)
display(pd.DataFrame([f.__dict__ for f in findings]))
expected_kinds = {"ssn", "phone", "email", "street_address", "mrn"}
found_kinds = {f.kind for f in findings}
control("4 PHI guard", "made-up PHI string is flagged (all five kinds)", "positive",
        sorted(expected_kinds), sorted(found_kinds), expected_kinds <= found_kinds)

near_miss = "BP 132/84, A1c 7.2 on 2025-07-10, LOINC 4548-4, ICD-10 E11.9, MRN SYN-100001, Plt 245 K/uL."
control("4 PHI guard", "clinical numbers that are not PHI are not flagged", "negative",
        "0 findings", f"{len(scan_phi(near_miss))} findings", not scan_phi(near_miss))
""")

code("""
t0 = time.perf_counter()
doc_findings = check_documents(documents)
control("4 PHI guard", f"all {len(documents)} synthetic documents produce no PHI findings", "negative",
        "0 documents with findings", f"{len(doc_findings)} documents with findings "
        f"({time.perf_counter() - t0:.2f} s)", not doc_findings)

unflagged = Document(doc_id="TEST-unflagged", patient_id="P999", doc_type="clinical_note", date="2026-01-01",
                     title="Test", text="Made-up text.", metadata={})          # no synthetic flag
try:
    assert_synthetic([unflagged])
    refused = False
except NonSyntheticDataError as exc:
    refused, msg = True, str(exc)
control("4 PHI guard", "a document without synthetic=True is refused", "positive",
        "NonSyntheticDataError", "raised: " + (msg if refused else "nothing"), refused)
""")

md("""
*Limits, stated honestly:* this is a pattern-based tripwire, not de-identification. It will miss a patient's
name in free text, an unusual date format or a rare identifier. Its job is to catch an accidental load of the
wrong file. It does **not** make it safe to use real data.
""")

# =========================================================================== 5. embeddings
md("""
## 5. Embeddings: do lay words land near clinical abbreviations?

An **embedder** turns text into a vector, so that texts with similar meaning have a high cosine similarity.
We compare two:

* **TF-IDF + SVD** (`tfidf`): fitted on this corpus only. It is essentially lexical, so it only knows words that
  appear in the charts.
* **MiniLM** (`all-MiniLM-L6-v2`): a small pretrained sentence model that has seen general English text.

The chart text uses abbreviations (`s/p MI`, `HTN`, `SOB`), while patients and new users ask in lay words. The
table compares each lay term with its abbreviation and with its spelled-out clinical term. **Unrelated
clinical terms are the negative controls**: a "match" only counts if it scores clearly higher than an unrelated
term from the same domain. Identical text is the positive control and must score 1.0.
""")

code("""
lay = {
    "heart attack":        {"abbreviation": "s/p MI", "clinical term": "myocardial infarction", "unrelated control": "knee OA"},
    "high blood pressure": {"abbreviation": "HTN",    "clinical term": "hypertension",          "unrelated control": "hypothyroidism"},
    "shortness of breath": {"abbreviation": "SOB",    "clinical term": "dyspnea",               "unrelated control": "GERD"},
}
embedders = {name: p.embedder for name, p in PIPES.items()}

def cos(embedder, a: str, b: str) -> float:
    return float(np.dot(embedder.embed_query(a), embedder.embed_query(b)))

rows = []
for term, targets in lay.items():
    for kind, other in targets.items():
        rows.append({"lay term": term, "compared with": other, "pair type": kind,
                     **{f"{n} cosine": round(cos(e, term, other), 3) for n, e in embedders.items()}})
rows.append({"lay term": "HTN", "compared with": "HTN", "pair type": "identical (positive control)",
             **{f"{n} cosine": round(cos(e, "HTN", "HTN"), 3) for n, e in embedders.items()}})
sim = pd.DataFrame(rows)
display(sim)

for n, e in embedders.items():
    control("5 embeddings", f"{n}: identical text has cosine 1.0", "positive", "1.000",
            f"{cos(e, 'HTN', 'HTN'):.3f}", abs(cos(e, "HTN", "HTN") - 1) < 1e-4)
""")

code("""
# Does the clinically matched text beat the unrelated control for the same lay term?
verdict = []
for n in embedders:
    for term, targets in lay.items():
        r = sim[(sim["lay term"] == term)].set_index("pair type")[f"{n} cosine"]
        verdict.append({"embedder": n, "lay term": term,
                        "abbreviation beats control": bool(r["abbreviation"] > r["unrelated control"]),
                        "clinical term beats control": bool(r["clinical term"] > r["unrelated control"]),
                        "margin (clinical term - control)": round(r["clinical term"] - r["unrelated control"], 3)})
display(pd.DataFrame(verdict))

# The same comparison inside short sentences, closer to what a chunk looks like.
sentences = [("Patient has a history of heart attack.", "Patient is s/p MI.", "Patient has knee OA."),
             ("Patient has high blood pressure.", "Patient has HTN.", "Patient has hypothyroidism.")]
display(pd.DataFrame([{"query": q, "match": m, "control": c,
                       **{f"{n} match": round(cos(e, q, m), 3) for n, e in embedders.items()},
                       **{f"{n} control": round(cos(e, q, c), 3) for n, e in embedders.items()}}
                      for q, m, c in sentences]))
""")

md("""
**What we observed (seed 42, cached `all-MiniLM-L6-v2`):**

* MiniLM links the lay terms to the *spelled-out* clinical words ("high blood pressure" vs "hypertension"
  about 0.73, well above the controls). It does **not** reliably link them to the *abbreviations*: "heart
  attack" scores higher against the unrelated control "knee OA" than against "s/p MI". In short sentences, the
  matched and control sentences score almost the same, because the shared words ("Patient has ...")
  dominate.
* TF-IDF gives near-zero similarity for the abbreviation pairs and for "high blood pressure" vs
  "hypertension". The lay terms never occur in the charts (by design), so it can only match on shared words.
  Its one clear "hit", "heart attack" vs "myocardial infarction" (about 0.42), shows that even a lexical
  model can pick up co-occurrence from the corpus (for example "heart" near "myocardial"). It is not
  understanding the lay term.

Without the unrelated controls, a MiniLM similarity of 0.15-0.25 for a lay/abbreviation pair would
look like "some semantic match". With the controls, you can see it is noise. Retrieval of lay-language questions
still works partly (section 9), because chunks such as the problem list spell out the full terms ("Essential
(primary) hypertension (HTN)").
""")

# =========================================================================== 6. vector space
md("""
## 6. The vector space

Each chunk is a point in 256 dimensions (TF-IDF) or 384 dimensions (MiniLM). We load the stored vectors back from
Chroma and project them to 2D in two ways:

* **PCA** is linear. Distances along the axes are faithful but crowded, and the first two components keep only a
  small part of the variance.
* **t-SNE** is non-linear. It keeps *local neighbourhoods*, but the distances *between* clusters and the cluster
  sizes are **not** meaningful, and the picture changes with the seed and the perplexity.

Neither projection proves anything by itself. So we also compute one number in the full space: for each
chunk, does its nearest neighbour have the same document type? The same patient? We compare these with what
chance would give, which is the negative control for the metric.
""")

code("""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

vec_chunks, X = pipe.store.get_embeddings()
doc_type = np.array([c.doc_type for c in vec_chunks])
patient = np.array([c.patient_id for c in vec_chunks])
print(f"{MAIN}: {X.shape[0]} chunk vectors of dimension {X.shape[1]}")

t0 = time.perf_counter()
pca = PCA(n_components=2, random_state=settings.seed)
P2 = pca.fit_transform(X)
T2 = TSNE(n_components=2, perplexity=30, init="pca", metric="cosine",
          random_state=settings.seed).fit_transform(X)
print(f"PCA explains {pca.explained_variance_ratio_.sum():.1%} of the variance with 2 components; "
      f"projections took {time.perf_counter() - t0:.1f} s")
""")

code("""
GROUPS = {"clinical notes": ["clinical_note"], "lab panels": ["lab_panel"], "vital signs": ["vitals"],
          "lists (demographics, problems, meds, allergies)":
              ["demographics", "problem_list", "medication_list", "allergy_list"]}

fig, axes = plt.subplots(2, len(GROUPS), figsize=(13, 6.4), sharex="row", sharey="row")
for row, (proj, name) in enumerate([(P2, "PCA"), (T2, "t-SNE")]):
    for col, (label, types) in enumerate(GROUPS.items()):
        ax = axes[row, col]
        mask = np.isin(doc_type, types)
        ax.scatter(proj[~mask, 0], proj[~mask, 1], s=4, color=BACKGROUND_POINTS, alpha=0.35, lw=0)
        ax.scatter(proj[mask, 0], proj[mask, 1], s=7, color=SERIES[0], lw=0)
        ax.set_title(f"{name}: {label} (n={mask.sum()})" if row == 0 else f"{name}", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle(f"{MAIN} chunk vectors, coloured by document type (blue = group, gray = all other chunks)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()
""")

code("""
SHOW = ["P001", "P002", "P003"]     # at most 3 highlighted patients keeps the colours distinguishable
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (proj, name) in zip(axes, [(P2, "PCA"), (T2, "t-SNE")]):
    rest = ~np.isin(patient, SHOW)
    ax.scatter(proj[rest, 0], proj[rest, 1], s=4, color=BACKGROUND_POINTS, alpha=0.35, lw=0, label="other 27 patients")
    for color, pid in zip(SERIES, SHOW):
        m = patient == pid
        ax.scatter(proj[m, 0], proj[m, 1], s=14, color=color, edgecolors=SURFACE, linewidths=0.5, label=pid)
    ax.set_title(f"{name}: chunks of three patients", loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axes[0].legend(loc="best", markerscale=1.5)
fig.tight_layout()
plt.show()
""")

code("""
def neighbour_agreement(chunks, X):
    \"\"\"Share of chunks whose nearest neighbour (cosine, excluding itself) has the same doc_type / patient.\"\"\"
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    dt = np.array([c.doc_type for c in chunks]); pt = np.array([c.patient_id for c in chunks])
    chance = lambda labels: float(sum((np.sum(labels == u) / len(labels)) ** 2 for u in np.unique(labels)))
    return {"same doc_type": float(np.mean(dt == dt[nn])), "same doc_type (chance)": chance(dt),
            "same patient": float(np.mean(pt == pt[nn])), "same patient (chance)": chance(pt)}

nn_table = pd.DataFrame({name: neighbour_agreement(*p.store.get_embeddings()) for name, p in PIPES.items()}).round(3)
display(nn_table)
""")

md("""
**Interpretation prompts:**

1. Chunks group very strongly by *document type* (the nearest neighbour has the same type about 99% of the
   time). Why? Think about the templates: every vital-signs chunk has the same lines with different
   numbers.
2. The nearest neighbour is from the *same patient* far more often than chance (1 in 30). Part of the reason is
   that every chunk header contains the patient id (`[Patient P001 | ...`). Is that a feature or a confound?
   And why does it make it even more important that patient isolation is a hard **filter** rather than
   relying on similarity?
3. Rerun this section with `MAIN = "tfidf"` (and `pipe = PIPES["tfidf"]`). Which picture changes more:
   PCA or t-SNE?
4. t-SNE clusters look convincing. Which conclusions from the plots would you be willing to put in a paper,
   and which would need the full-dimensional number instead?
""")

# =========================================================================== 7. retrieval
md("""
## 7. Retrieval: four retrievers side by side

Retrieval always runs inside **one patient's** chunks. The four modes are:

| mode | how it ranks |
|---|---|
| `random` | a deterministic random sample: the **baseline control** |
| `bm25` | word overlap (BM25 Okapi) |
| `dense` | cosine similarity of embeddings |
| `hybrid` | Reciprocal Rank Fusion of the top 30 of `bm25` and `dense`: score = sum of 1/(60 + rank) |

We ask a lay-language question and mark retrieved chunks that name one of the patient's actual
blood-pressure medications (from the medications table: `indication == "htn"`). The marks come from the
ground truth, not from the system.
""")

code("""
QUESTION = "What medications is the patient taking for their high blood pressure?"
QPID = "P001"

def htn_meds(pid):
    m = tables["medications"].query("patient_id == @pid and indication == 'htn' and status == 'active'")
    return sorted(m["name"])

def side_by_side(question, pid, p=pipe, k=6):
    truth = htn_meds(pid)
    cols = {}
    for mode in MODES:
        hits = p.retrieve(question, pid, k=k, mode=mode)
        cols[mode] = [f"{'* ' if any(t in h.chunk.text for t in truth) else '  '}{h.chunk.chunk_id}"
                      f" [{h.chunk.section or h.chunk.doc_type}] {h.score:.3f}" for h in hits]
    return pd.DataFrame(cols, index=pd.RangeIndex(1, k + 1, name="rank")), truth

table, truth = side_by_side(QUESTION, QPID)
print(f"{QPID} | {QUESTION}\\nground truth (active HTN meds): {truth}   (* = chunk names one of them)")
display(table)
""")

md("""
*What to look for:*

In our run (MiniLM index, P001):

* `bm25` returns six **vital-signs** chunks. They contain the literal words "Blood pressure: .../... mmHg", so
  word overlap is high, but they say nothing about medications. This is a *lexical false lead*.
* `dense` returns ASSESSMENT AND PLAN sections, where the chart says "# HTN: ... Continue lisinopril 20 mg PO
  daily". That is the line that actually links the drug to the condition.
* `hybrid` returns MEDICATIONS sections. They name lisinopril, but they list every drug without saying what
  each one is *for*. Every `*` here is "correct" by the ground-truth mark, yet would you accept these chunks as
  evidence for "taking it for high blood pressure"? Fusing two retrievers does not always give the most
  *supporting* chunk.
* Compare with `random`: if a real retriever does not clearly beat the random baseline, it is not working.
""")

md("""
### Patient isolation

The same question asked under a different patient must return **only that patient's** chunks. We check this
in three ways:

1. Same question, two patients: every chunk must belong to the patient asked about (negative control: zero
   foreign chunks).
2. A deliberately **leaky** vector store that returns another patient's chunks. The retriever's guard must
   catch and drop them. This is a positive control *for the guard*: if the guard does not fire here, then
   seeing 0 leaks in normal runs proves nothing.
3. The generators refuse to answer from another patient's chunks.
""")

code("""
OTHER = "P002"
for pid in (QPID, OTHER):
    hits = pipe.retrieve(QUESTION, pid, mode="hybrid")
    foreign = [h.chunk.chunk_id for h in hits if h.chunk.patient_id != pid]
    print(f"{pid}: HTN meds in table = {htn_meds(pid)}; retrieved {[h.chunk.chunk_id for h in hits]}")
    control("7 isolation", f"hybrid retrieval for {pid} returns only {pid}'s chunks", "negative",
            "0 foreign chunks", f"{len(foreign)} foreign chunks", not foreign)
""")

code("""
from ehr_rag.retrieval import Retriever, PatientIsolationError
from ehr_rag.generation import ExtractiveGenerator

class LeakyStore:
    \"\"\"Deliberately broken store for a control: ignores the requested patient and returns OTHER's chunks.\"\"\"
    def __init__(self, store, leak_from):
        self.store, self.leak_from = store, leak_from
    def query(self, embedding, k, patient_id, doc_types=None):
        return self.store.query(embedding, k, self.leak_from, doc_types=doc_types)
    def get_chunks(self, patient_id=None):
        return self.store.get_chunks(self.leak_from)

leaky = Retriever(LeakyStore(pipe.store, OTHER), pipe.embedder, pipe.bm25)
hits = leaky.retrieve(QUESTION, QPID, k=6, mode="dense")
control("7 isolation", "guard drops chunks leaked by a broken store", "positive",
        "0 chunks returned, violations counted > 0",
        f"{len(hits)} returned, {leaky.isolation_violations} violations counted",
        not hits and leaky.isolation_violations > 0)

strict = Retriever(LeakyStore(pipe.store, OTHER), pipe.embedder, pipe.bm25, strict=True)
try:
    strict.retrieve(QUESTION, QPID, k=6, mode="dense"); raised = "nothing"
except PatientIsolationError as exc:
    raised = f"PatientIsolationError: {str(exc)[:90]}..."
control("7 isolation", "strict guard raises on a leak", "positive", "PatientIsolationError", raised,
        raised.startswith("PatientIsolationError"))

try:
    pipe.store.query(pipe.embedder.embed_query(QUESTION), 6, patient_id="")
    raised = "nothing"
except ValueError as exc:
    raised = f"ValueError: {exc}"
control("7 isolation", "vector store refuses a query without a patient id", "positive", "ValueError", raised,
        raised.startswith("ValueError"))

other_ctx = pipe.retrieve(QUESTION, OTHER)
try:
    ExtractiveGenerator().generate(QUESTION, QPID, other_ctx); raised = "nothing"
except ValueError as exc:
    raised = f"ValueError: {str(exc)[:90]}..."
control("7 isolation", f"generator refuses {OTHER}'s chunks for a {QPID} question", "positive", "ValueError",
        raised, raised.startswith("ValueError"))
print("pipeline isolation counter (real runs):", {n: p.retriever.isolation_violations for n, p in PIPES.items()})
""")

# =========================================================================== 8. generation
md("""
## 8. Generation: answers with citations

The generator receives the top-k chunks, labelled `[S1] ... [S6]`, and must answer *only* from them, citing the
labels it used. There are two generators:

* **Extractive** (offline, no LLM): it quotes the sentences that contain the question's key terms, and
  **abstains** ("does not appear to document") when a key term is missing from every excerpt.
* **Claude** (optional): returns JSON `{answer, cited_labels, insufficient_evidence}`. Invalid labels are
  dropped, every chunk is checked for `synthetic: true` before any network call, and a refusal or "insufficient
  evidence" becomes an abstention.

For every answer below, we print the ground truth from the raw table next to it. That comparison is your job
as the reader; the system cannot do it for itself.
""")

code("""
def show_answer(ans):
    print(f"Patient {ans.patient_id} | Q: {ans.question}\\n")
    print(ans.text)
    print("\\nCitations:")
    for c in ans.citations:
        print(f"  [{c.label}] {c.title} ({c.date}) - {c.chunk_id}")
    print(f"Abstained: {ans.abstained} | generator: {ans.mode} | {len(ans.retrieved)} chunks | {ans.latency_ms:.0f} ms")
    if ans.warnings:
        print("Warnings:", ans.warnings)

extractive = ExtractiveGenerator()
pipe.generator = extractive
ans = pipe.ask("Why was sertraline 50 mg stopped?", "P001")
show_answer(ans)

truth = tables["medications"].query("patient_id == 'P001' and name == 'sertraline' and dose == '50 mg'").iloc[0]
print(f"\\nGround truth (medications.jsonl {truth.med_id}): stopped {truth.end_date}, reason: {truth.discontinue_reason}")
control("8 generation", "stop reason from the table appears in the answer", "positive",
        truth.discontinue_reason, "found" if truth.discontinue_reason in ans.text else "missing",
        truth.discontinue_reason in ans.text and not ans.abstained)
""")

code("""
ans = pipe.ask("What is the most recent HbA1c?", "P003")
show_answer(ans)
latest = tables["labs"].query("patient_id == 'P003' and test_key == 'hba1c'").sort_values("date").iloc[-1]
first_line = ans.text.splitlines()[0]
print(f"\\nGround truth: {latest.value:g} {latest.unit} on {latest.date}")
control("8 generation", "'most recent' answer leads with the latest table value and date", "positive",
        f"{latest.value:g} and {latest.date} in the first line", first_line[:140],
        f"{latest.value:g}" in first_line and latest.date in first_line)
""")

md("""
**Check the citations yourself.** For each citation, print the cited chunk and decide: does it actually
support the sentence it is attached to? A *valid* citation (the label exists and belongs to this patient) is not
necessarily a *supporting* citation. The automated `citation_validity` metric only checks the first.
""")

code("""
by_label = {f"S{i}": rc for i, rc in enumerate(ans.retrieved, start=1)}
for c in ans.citations:
    print(f"===== [{c.label}] {c.chunk_id}")
    print(by_label[c.label].chunk.text[:600], "\\n")
""")

md("""
### Abstention: the negative controls

Questions about facts that are not in the chart must lead to "not documented", not to a guess. We use two
questions from the evaluation set. `absent_fact` asks about a drug that this patient never took.
`cross_patient` asks about another patient's distinctive allergen. Before trusting an abstention, we check
the absence independently with a plain text scan of all the patient's documents.
""")

code("""
from ehr_rag.evaluation.questions import read_questions, build_questions
from ehr_rag.synth.generator import SyntheticDataset

qpath = settings.data_dir / "eval_questions.jsonl"
questions = (read_questions(qpath) if qpath.exists() else
             build_questions(SyntheticDataset.load(settings.data_dir), pipe.chunks, seed=settings.seed))
qby = {q.qid: q for q in questions}
print(len(questions), "questions;", pd.Series([q.category for q in questions]).value_counts().to_dict())

neg = [next(q for q in questions if q.category == "absent_fact"),
       next(q for q in questions if q.category == "cross_patient")]
for q in neg:
    ans = pipe.ask(q.question, q.patient_id)
    show_answer(ans)
    term = q.metadata["key_term"].lower()
    in_chart = any(term in d.text.lower() for d in documents if d.patient_id == q.patient_id)
    src = q.metadata.get("source_patient_id")
    print(f"independent scan: '{term}' in {q.patient_id}'s documents? {in_chart}"
          + (f" (it is in {src}'s chart)" if src else ""), "\\n")
    control("8 generation", f"{q.category}: abstains on '{term}' ({q.patient_id})", "negative",
            "abstained=True", f"abstained={ans.abstained}", ans.abstained and not in_chart)
""")

md("""
### A safe failure, and an upper-bound configuration

Lay-language questions are hard for the lexical extractive generator. For P001 "high blood pressure", hybrid
retrieval *does* bring back chunks that name lisinopril (section 7). But the generator looks for the words
"high", "blood" and "pressure", and they are not in those chunks, so it abstains. That is a **safe** failure.

The "vocab-expanded" configuration gives the generator the lay-term-to-abbreviation map from the same vocabulary
that the evaluation questions were built from. It therefore **reuses the answer key**: treat its scores as an
upper bound, never as a fair result.

In our run, the expansions rescue P010 ("sugar diabetes" -> T2DM) but not P001: the MEDICATIONS chunks that hybrid
retrieved never mention HTN, so even the expanded terms are missing. Read the P010 answer carefully. It is
scored *correct*, because "metformin" appears in it, but it also quotes amlodipine and atorvastatin, which are not
diabetes drugs.
""")

code("""
from ehr_rag.evaluation.runner import vocab_term_expansions

for gen_name, gen in [("extractive", extractive),
                      ("extractive + vocab expansions (upper bound)", ExtractiveGenerator(term_expansions=vocab_term_expansions()))]:
    pipe.generator = gen
    for pid, q in [("P001", QUESTION), ("P010", "What medications is the patient taking for their sugar diabetes?")]:
        a = pipe.ask(q, pid)
        print(f"--- {gen_name} | {pid} | {q}\\n    abstained={a.abstained}\\n    " + a.text.replace("\\n", "\\n    ")[:500])
pipe.generator = extractive
print("\\nP010 ground truth (active T2DM meds):",
      sorted(tables["medications"].query("patient_id == 'P010' and indication == 't2dm' and status == 'active'")["name"]))
""")

md("""
### Claude (optional)

The next cell first shows exactly what *would* be sent: the fixed system prompt and the user message with the
labelled excerpts. Building this preview makes no network call. If `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`) is set, the cell after it sends **three** requests (one positive question, two
negative controls), which costs a small amount of money. Otherwise, it prints how to enable this.
""")

code("""
from ehr_rag.generation import ClaudeGenerator, GenerationError, has_claude_credentials

claude = ClaudeGenerator(model=settings.claude_model, effort=settings.claude_effort,
                         max_tokens=settings.claude_max_tokens, fallbacks=settings.claude_fallbacks)
ctx = pipe.retrieve("What is the most recent HbA1c?", "P003")
req = claude.build_request("What is the most recent HbA1c?", "P003", ctx)
print("model:", req["model"], "| effort:", req["output_config"]["effort"])
print("SYSTEM PROMPT:\\n" + req["system"])
user = req["messages"][0]["content"]
print("USER MESSAGE (first 1200 of", len(user), "characters):\\n" + user[:1200] + "\\n...")
""")

code("""
if has_claude_credentials():
    pipe.generator = claude
    try:
        for pid, q in [("P003", "What is the most recent HbA1c?"),
                       (neg[0].patient_id, neg[0].question), (neg[1].patient_id, neg[1].question)]:
            a = pipe.ask(q, pid)
            show_answer(a)
            print()
            if q != "What is the most recent HbA1c?":
                control("8 generation", f"Claude abstains on a negative question ({pid})", "negative",
                        "abstained=True", f"abstained={a.abstained}", a.abstained)
    finally:
        pipe.generator = extractive
else:
    print("No Claude credentials found, so this cell made no API call.\\n"
          "To try it: export ANTHROPIC_API_KEY=sk-ant-... in the shell that starts Jupyter, restart the kernel,\\n"
          "and rerun this notebook. The CLI equivalent is:\\n"
          "  python -m ehr_rag ask \\"What is the most recent HbA1c?\\" -p P003 --llm claude\\n"
          "Only synthetic chunks are ever sent; every chunk is checked for synthetic=True first.")
""")

# =========================================================================== 9. evaluation
md("""
## 9. Evaluation

The evaluation runner asks all 210 gold questions (7 per patient) and scores each step separately:

* **Retrieval:** `recall@k` (a gold document is among the top k), `evidence recall@k` (stricter: the retrieved
  *chunk* itself contains an answer keyword) and MRR.
* **Generation:** answer accuracy (an expected keyword appears in the answer), false abstention on positives,
  correct abstention on negatives, citation validity, and leakage (any chunk from another patient).

We load the reports already written by `python -m ehr_rag eval --compare` instead of recomputing them. If a
report is missing, the cell runs a retrieval-only comparison in memory, which takes a few seconds.
""")

code("""
from ehr_rag.evaluation.runner import compare_retrievers, run_eval, control_status

def load_report(name):
    path = settings.reports_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None

compare = {}
for name, p in PIPES.items():
    rep = load_report(f"compare_{name}")
    if rep is None:
        print(f"reports/compare_{name}.json not found; running compare_retrievers in memory")
        rep = compare_retrievers(p, questions)
    compare[name] = rep
    cfg = rep["config"]
    print(f"compare_{name}: {cfg['n_questions']} questions, k={cfg['k']}, created {cfg['created_at']}")

POS_CATS = ["fact_lookup", "temporal", "semantic_synonym", "verbatim"]
long = pd.DataFrame([{"embedder": e, "mode": m, "category": c, "n": s["by_category"][c]["n"],
                      "recall@k": s["by_category"][c]["recall@k"], "evidence recall@k": s["by_category"][c]["evidence_recall@k"],
                      "recall@1": s["by_category"][c]["recall@1"]}
                     for e, rep in compare.items() for m, s in rep["modes"].items() for c in POS_CATS])
""")

code("""
metrics = ["recall@k", "evidence recall@k"]
emb_names = list(compare)
fig, axes = plt.subplots(len(metrics), len(emb_names), figsize=(6.2 * len(emb_names), 6.4), sharey=True, squeeze=False)
width = 0.2
x = np.arange(len(POS_CATS))
for r, metric in enumerate(metrics):
    for c, emb in enumerate(emb_names):
        ax = axes[r, c]
        for i, mode in enumerate(MODES):
            vals = [long.query("embedder == @emb and mode == @mode and category == @cat")[metric].iloc[0] for cat in POS_CATS]
            ax.bar(x + (i - 1.5) * width, vals, width, color=MODE_COLORS[mode], edgecolor=SURFACE, linewidth=1.5,
                   label=mode)
        ax.set_xticks(x, [p.replace("_", "\\n") for p in POS_CATS])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{emb}: {metric.replace('k', str(compare[emb]['config']['k']))}", loc="left")
        ax.grid(axis="x", visible=False)
axes[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.1))
fig.suptitle("Retrieval by question category and retriever (positives only)", x=0.01, ha="left",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
plt.show()

# Table views: evidence recall@k (the bottom row of the chart) and recall@1 (is the FIRST chunk right?).
for metric in ("evidence recall@k", "recall@1"):
    print(metric)
    display(long.pivot_table(index=["embedder", "category"], columns="mode", values=metric)[list(MODES)].round(2))
""")

md("""
*Reading the chart:*

* **Plain recall@6 flatters the random baseline.** Medications and allergies are repeated in every note, so a
  random chunk often comes from *some* gold document. The chunk-level evidence recall is the fairer comparison.
* **Verbatim** is a positive control for the whole retrieval stack: a question that quotes a chunk must find
  it. BM25 gets it almost perfectly. MiniLM dense does not (meaning-based similarity is weak on exact
  quotes).
* On **semantic_synonym**, the random baseline's plain recall@6 (0.90) is *higher* than BM25's (0.83). "Has
  condition X" questions have many gold notes, so almost any chunk counts. Evidence recall drops random to
  0.53. A metric that a random retriever passes is not measuring what you think it measures.
* **Temporal** ("most recent X") recall@1 is low in every mode (second table), because
  similarity has no notion of date.
""")

code("""
# Every automated control check from the saved reports.
reports = {f"compare_{e}": rep for e, rep in compare.items()}
for name in ("eval_tfidf_hybrid_extractive", "eval_minilm_hybrid_extractive"):
    rep = load_report(name)
    if rep is not None:
        reports[name] = rep
ctl = pd.DataFrame([{"report": r, "check": c["name"], "value": None if c["value"] is None else round(c["value"], 3),
                     "threshold": c["threshold"], "status": c["status"]}
                    for r, rep in reports.items() for c in rep["controls"]])
display(ctl)
print({r: control_status(rep) for r, rep in reports.items()})
""")

md("""
**What a failing control would mean:**

* **`leakage_rate == 0` FAIL.** Some answer was built with another patient's chart. This is a stop-everything
  safety bug. No accuracy number from that run means anything, and the cause (filter, index, guard) must be
  found before anything else.
* **`verbatim recall@1 >= 0.9` FAIL** (a positive control). The retriever cannot find text that is literally
  in the question. Something is broken (chunking, index out of date, tokenizer), or, as for MiniLM
  `hybrid` in our reports, one fused retriever outvotes the other. Do not trust that configuration's other
  recall numbers until you understand why.
* **`random ... <= best real - 0.2` FAIL** (the baseline control). The real retrievers are no better than
  chance. This usually means the questions are too easy, or the gold labels are too generous.
* **`citation_validity == 1` FAIL.** The generator cited something it was not given. Treat that as
  fabrication.

A green report is not proof of correctness. It means that none of the problems we thought of in advance
showed up.
""")

code("""
# Reproducibility: the saved report should match what the code produces now (retrieval-only, same questions).
p = PIPES["tfidf"]
t0 = time.perf_counter()
live = run_eval(p, questions, mode="hybrid", generate=False)
live_val = live["summary"]["by_control"]["positive"]["evidence_recall@k"]
saved_val = compare["tfidf"]["modes"]["hybrid"]["by_control"]["positive"]["evidence_recall@k"]
print(f"live run: {len(questions)} questions in {time.perf_counter() - t0:.1f} s")
control("9 evaluation", "re-running tfidf hybrid reproduces the saved evidence recall@k", "positive",
        f"{saved_val:.4f} (saved report)", f"{live_val:.4f} (live)", abs(live_val - saved_val) < 1e-9)
""")

code("""
# A control for the control: a deliberately leaky pipeline must make the leakage check FAIL.
from ehr_rag.types import RetrievedChunk

class LeakyPipeline:
    \"\"\"Puts one chunk of another patient at rank 1 of every result (for this demonstration only).\"\"\"
    def __init__(self, base, leak_from):
        self.base, self.leak_from = base, leak_from
        self.settings, self.manifest, self.generator = base.settings, base.manifest, base.generator
    def retrieve(self, question, patient_id, k=None, mode=None, doc_types=None):
        k = k or self.settings.top_k
        foreign = self.base.retrieve(question, self.leak_from, k=1, mode="bm25")
        hits = foreign + self.base.retrieve(question, patient_id, k=k - 1, mode="bm25")
        return [RetrievedChunk(h.chunk, h.score, i, "bm25") for i, h in enumerate(hits, start=1)]

p001_qs = [q for q in questions if q.patient_id == "P001"]
bad = run_eval(LeakyPipeline(p, "P002"), p001_qs, mode="bm25", generate=False)
leak_check = next(c for c in bad["controls"] if c["metric"] == "leakage_rate")
print(f"leaky pipeline: leakage_rate={bad['summary']['overall']['leakage_rate']}, check status={leak_check['status']}")
control("9 evaluation", "the leakage check fails on a deliberately leaky pipeline", "positive",
        "leakage check = FAIL", f"leakage check = {leak_check['status']}", leak_check["status"] == "FAIL")
""")

md("""
### Looking at failures

Every report keeps the worst examples per category. Below are a real retrieval miss for a "most recent"
question, and answers in the vocab-expanded configuration that were *answered but wrong*. These are the cases
that a metric alone would hide inside an average.
""")

code("""
ev = load_report("eval_tfidf_hybrid_extractive")
if ev:
    for f in ev["failures"].get("temporal", [])[:2]:
        print(f"{f['qid']} | {f['question']} | expected: {f['expected_answer']} | why: {f['why']}")
        print("   answer:", (f["answer_text"] or "").splitlines()[0][:160])
vocab = load_report(f"eval_{MAIN}_hybrid_extractive-vocab")
if vocab:
    wrong = [r for r in vocab["rows"] if r["category"] == "semantic_synonym" and not r["abstained"] and not r["answer_correct"]]
    print(f"\\n{len(wrong)} semantic questions answered but wrong in eval_{MAIN}_hybrid_extractive-vocab:")
    for r in wrong[:3]:
        print(f"- {r['qid']} | {r['question']} | expected: {r['expected_answer']}")
        print("   answer:", (r["answer_text"] or "").replace("\\n", " ")[:230])
""")

# =========================================================================== summary
md("""
## Summary of the controls run in this notebook
""")

code("""
ctrl = pd.DataFrame(CONTROLS)
display(ctrl[["section", "type", "control", "status"]])
print(ctrl["status"].value_counts().to_dict())
""")

# =========================================================================== 10. discussion
md("""
## 10. Discussion questions for the lab

These are the points where **a person has to judge the output**. No metric in this notebook can answer them.

1. **Is the citation actually supporting the claim?** Take the sertraline answer in section 8. One cited
   sentence says "MDD on sertraline 50 mg PO daily". It is a valid citation, but does it support *why* the drug
   was stopped? How would you score an answer with one supporting and two irrelevant citations?
2. **Would you accept a MEDICATIONS section as evidence for "taking it for high blood pressure"?** It lists
   lisinopril but never says what it is for. What would a clinician need to see?
3. **"Most recent" in the retrieved excerpts is not "most recent in the chart."** The P015 sodium failure
   shows the latest panel was not retrieved. How would you design a check that catches this without
   knowing the answer?
4. **What would you change in the question set?** All questions come from the same templates as the chart
   text, so BM25 looks unrealistically strong. Which question types are missing (negation, "was it ever...",
   comparisons, dose changes, questions that need two documents)? How would you build a *held-out* set so that
   tuning does not overfit to these 210 questions?
5. **Is keyword accuracy too lenient?** An answer that quotes three excerpts with different values is
   counted as correct if one of them matches. The P010 "sugar diabetes" answer in section 8 is scored correct
   although it also lists amlodipine and atorvastatin. How would you score it instead, and who would do the
   grading?
6. **Abstention at 100% on negatives.** For the extractive generator this holds by construction: it abstains
   whenever the drug name is missing. What negative controls would be informative for Claude, which *can*
   reason (for example, a drug in the same class as one the patient takes)?
7. **The embedding controls in section 5** showed that a similarity of 0.2 can be noise. What controls would
   you require before claiming that an embedding "understands" clinical synonyms?
8. **Where should this system never be used?** List the safety properties that you have actually *seen*
   tested today, and the ones that you have only been told about.

Bring your answers, and any failure you find, to the next lab discussion.
""")


def build(path: Path = NOTEBOOK) -> Path:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    nbformat.validate(nb)
    path.write_text(nbformat.writes(nb), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({len(CELLS)} cells)")
