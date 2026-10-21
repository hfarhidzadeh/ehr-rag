# Module interfaces (build contract)

Shared types live in `src/ehr_rag/types.py`, settings in `src/ehr_rag/config.py` (`load_settings()`).
Those two files are frozen during the parallel build: do not edit them. If a module needs a change there,
say so in your final report.

Python: `/Users/hamidfarhid/Downloads/RAG/.venv/bin/python` (3.11, package installed editable).
Run tests with `.venv/bin/python -m pytest tests/<file> -q`. Tests marked `@pytest.mark.slow` (MiniLM/torch)
are skipped by default; run them with `-m slow`.

General rules
* Synthetic data only. Every raw record and every `Document.metadata` / `Chunk.metadata` has `"synthetic": True`.
* Deterministic: every random choice flows from an explicit `seed` (`random.Random(seed)` / `np.random.default_rng(seed)`),
  never global state, never wall-clock time. Dates are fixed relative to `REFERENCE_DATE = 2026-06-30`.
* The torch / sentence-transformers import must stay lazy (inside `MiniLMEmbedder`) so the `tfidf` path never imports torch.
* Library code logs with `logging.getLogger(__name__)`, no prints (CLI excepted).

---

## `ehr_rag.synth.vocab`
Clinical vocabulary as plain data: conditions (key, display, icd10, abbreviation, **lay_terms**, typical meds,
typical labs, note phrases), medications (generic name, dose options, route, frequency, indication key),
lab tests (key, name, LOINC, unit, ref_low, ref_high, panel, plausible normal/abnormal ranges per condition),
allergens (substance, reactions), providers, note templates.
**`lay_terms` (e.g. "heart attack", "high blood pressure", "sugar diabetes") must never appear in any generated
text** — they exist so evaluation can ask lay-language questions against abbreviated chart text.

## `ehr_rag.synth.generator`
```python
REFERENCE_DATE: datetime.date            # 2026-06-30
@dataclass
class SyntheticDataset:
    patients: list[dict]; encounters: list[dict]; conditions: list[dict]; medications: list[dict]
    allergies: list[dict]; labs: list[dict]; vitals: list[dict]; notes: list[dict]
    seed: int
    def to_documents(self) -> list[Document]
    def write(self, out_dir: Path) -> None   # <table>.jsonl for each table + documents.jsonl + manifest.json
    @classmethod
    def load(cls, data_dir: Path) -> "SyntheticDataset"
def generate_dataset(n_patients: int = 30, seed: int = 42) -> SyntheticDataset
def load_documents(data_dir: Path) -> list[Document]      # reads documents.jsonl
```
Record schemas (every record also has `"patient_id"` and `"synthetic": True`):
* patient: `patient_id` ("P001"), `mrn` ("SYN-100001"), `name` (obviously fictional), `sex`, `birth_date`, `age`, `pcp`
* encounter: `encounter_id` ("P001-E01"), `date`, `encounter_type` (outpatient|ed|inpatient|telehealth), `reason`, `provider`, `discharge_date` (inpatient only, else None)
* condition: `condition_key`, `display`, `icd10`, `abbreviation`, `onset_date`, `status` (active|resolved)
* medication: `med_id` ("P001-M03"), `name` (generic, lowercase), `dose` ("1000 mg"), `route`, `frequency`, `start_date`, `end_date` (None if active), `status` (active|discontinued), `indication` (condition_key), `discontinue_reason` (None if active)
* allergy: `substance`, `reaction`, `severity` (mild|moderate|severe). Patients with no allergies have no rows; their allergy document says "No known drug allergies (NKDA)."
* lab: `date`, `encounter_id`, `panel`, `test_key`, `name`, `loinc`, `value` (float), `unit`, `ref_low`, `ref_high`, `flag` ("H"|"L"|"")
* vital: `date`, `encounter_id`, `systolic`, `diastolic`, `heart_rate`, `resp_rate`, `temp_c`, `spo2`, `weight_kg`, `bmi`
* note: `note_id` ("P001-N02"), `encounter_id`, `date`, `note_type` (NOTE_TYPES), `author`, `title`, `sections` (ordered list of `[SECTION_NAME, text]`), `mentions` (`{"conditions": [keys], "medications": [med_ids], "labs": [[test_key, date]]}`)

Every patient: ≥3 encounters spread over ~2 years before REFERENCE_DATE, ≥2 dated lab panels including at least one
test that is repeated over time (trend), a medication list with ≥1 discontinued med (with reason), and ≥2 notes.
Notes agree with the structured tables (doses, values, dates) and favour abbreviations (T2DM, HTN, CKD3, SOB, s/p MI, A1c).
Patients should differ enough to support "distinctive fact" questions (unique allergy, unique med).

`to_documents()` produces exactly the doc types and ids in `PLAN.md` §3 using the id helpers in `types.py`.
Structured documents are rendered one record per line (e.g. `- metformin 1000 mg PO BID — started 2024-05-02 — for T2DM (active)`).
Note documents render sections as `SECTION NAME:\n<text>` separated by blank lines.

## `ehr_rag.ingest.chunking`
```python
def header_for(doc: Document, section: str | None) -> str
    # "[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]"  (section part omitted if None)
def chunk_document(doc: Document, max_chars: int = 900, overlap_sentences: int = 1) -> list[Chunk]
def chunk_documents(docs: Iterable[Document], max_chars: int = 900, overlap_sentences: int = 1) -> list[Chunk]
```
Notes: split on section header lines (`^[A-Z][A-Z0-9 /&(),'-]+:$`); one or more chunks per section; long sections split on
sentence boundaries with `overlap_sentences` carried over. Structured docs: group whole lines up to `max_chars`, never split a line.
Each chunk's `text` = header line + "\n" + body. Chunk ids via `chunk_id_for(doc_id, i)`; `metadata` copies the document metadata
(so `synthetic` propagates) and adds `note_type` when present.

## `ehr_rag.ingest.phi_guard`
```python
class NonSyntheticDataError(ValueError)
@dataclass
class PHIFinding: kind: str; match: str; start: int; end: int
def assert_synthetic(items: Iterable[Document | Chunk]) -> None       # raises NonSyntheticDataError naming offending ids
def scan_phi(text: str) -> list[PHIFinding]                           # SSN, phone, email, URL, street address, non-"SYN-" MRN
def check_documents(docs: Iterable[Document]) -> dict[str, list[PHIFinding]]   # only docs with findings
```
The synthetic generator must produce zero findings.

## `ehr_rag.embeddings`
```python
class Embedder(Protocol):
    name: str; dim: int
    def fit(self, texts: list[str]) -> None                      # no-op for pretrained models
    def embed_documents(self, texts: list[str]) -> np.ndarray     # (n, dim) float32, L2-normalised
    def embed_query(self, text: str) -> np.ndarray                # (dim,) float32, L2-normalised
    def save(self, directory: Path) -> None
class MiniLMEmbedder:   # __init__(model_name=..., batch_size=32, device="cpu"); lazy torch import
class TfidfSvdEmbedder: # __init__(dim=256, seed=42); TF-IDF (1-2 grams, sublinear_tf) -> TruncatedSVD; dim shrinks if vocab is small
def get_embedder(settings: Settings) -> Embedder                 # fresh, unfitted
def load_embedder(settings: Settings) -> Embedder                # fitted; tfidf loads from settings.embedder_index_dir
```

## `ehr_rag.vectorstore`
```python
class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str)   # PersistentClient, cosine space, telemetry off
    def reset(self) -> None
    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 512) -> None
    def query(self, embedding: np.ndarray, k: int, patient_id: str,
              doc_types: list[str] | None = None) -> list[tuple[Chunk, float]]   # similarity = 1 - cosine distance, desc
    def count(self) -> int
    def get_chunks(self, patient_id: str | None = None) -> list[Chunk]
    def get_embeddings(self, patient_id: str | None = None) -> tuple[list[Chunk], np.ndarray]
```
`query` raises `ValueError` if `patient_id` is falsy (patient isolation is mandatory). Chroma metadata must be scalar: store chunk
fields flat and `metadata` as a JSON string; reconstruct `Chunk` exactly on read.

## `ehr_rag.retrieval`
```python
def tokenize(text: str) -> list[str]      # lowercase; keeps alphanumerics, decimals ("7.2"), drug names; drops stopwords
def reciprocal_rank_fusion(ranked_ids: list[list[str]], k: int = 60) -> list[tuple[str, float]]
class BM25Index:
    def __init__(self, chunks: list[Chunk])            # one BM25Okapi per patient
    def search(self, query: str, patient_id: str, k: int) -> list[tuple[Chunk, float]]
class Retriever:
    def __init__(self, store: ChromaStore, embedder: Embedder, bm25: BM25Index,
                 rrf_k: int = 60, candidate_pool: int = 30)
    def retrieve(self, query: str, patient_id: str, k: int = 6, mode: str = "hybrid",
                 doc_types: list[str] | None = None) -> list[RetrievedChunk]
```
Modes: `dense`, `bm25`, `hybrid` (RRF over dense+bm25 candidate pools), `random` (baseline control: deterministic sample of the
patient's chunks seeded from the query text). Every returned chunk has `chunk.patient_id == patient_id` — enforce, don't assume.

## `ehr_rag.generation`
```python
@dataclass
class GenerationResult: text: str; citations: list[Citation]; abstained: bool; mode: str; model: str | None; warnings: list[str]
class Generator(Protocol):
    name: str
    def generate(self, question: str, patient_id: str, contexts: list[RetrievedChunk]) -> GenerationResult
def format_context(contexts: list[RetrievedChunk]) -> str          # "[S1] <chunk text>\n\n[S2] ..."
class ClaudeGenerator:     # __init__(model, effort, max_tokens, fallbacks=True, client=None)
class ExtractiveGenerator: # offline; cites the supporting sentences; abstains when evidence is weak
def has_claude_credentials() -> bool
def get_generator(settings: Settings) -> Generator   # llm="auto": Claude if credentials else extractive
```
ClaudeGenerator: calls `assert_synthetic` on the contexts **before** any network call; asks for JSON
`{answer, cited_labels, insufficient_evidence}`; validates labels against S1..Sk (drops invalid ones, adds a warning); maps
`stop_reason == "refusal"` to an abstention with a warning; never invents facts outside the context.

## `ehr_rag.pipeline`
```python
class IndexNotBuiltError(RuntimeError)
class EHRRagPipeline:
    settings: Settings; store: ChromaStore; embedder: Embedder; retriever: Retriever; generator: Generator
    @classmethod
    def build_index(cls, settings: Settings, documents: list[Document] | None = None) -> "EHRRagPipeline"
        # documents default: load_documents(settings.data_dir). assert_synthetic + PHI check -> chunk -> fit/embed
        # -> store.reset()+add -> write chunks.jsonl, embedder, index_manifest.json into settings.embedder_index_dir
    @classmethod
    def load(cls, settings: Settings) -> "EHRRagPipeline"     # IndexNotBuiltError with a how-to-fix message
    def retrieve(self, question: str, patient_id: str, k: int | None = None, mode: str | None = None,
                 doc_types: list[str] | None = None) -> list[RetrievedChunk]
    def ask(self, question: str, patient_id: str, k: int | None = None, mode: str | None = None,
            doc_types: list[str] | None = None) -> Answer
    def list_patients(self) -> list[dict]
    def patient_summary(self, patient_id: str) -> dict        # demographics, active problems, active meds, allergies
```
If Claude fails with an API/connection error, `ask` falls back to the extractive generator and records a warning.

## `ehr_rag.cli` (`python -m ehr_rag`)
`generate [--n-patients --seed]`, `ingest [--embedder]`, `ask QUESTION --patient PID [--mode --k --llm --embedder]`,
`eval [--embedder --mode --llm --limit --compare]`, `app` (runs `streamlit run app/streamlit_app.py`).

## `ehr_rag.evaluation`
```python
# questions.py
def build_questions(dataset: SyntheticDataset, chunks: list[Chunk] | None = None, seed: int = 42) -> list[EvalQuestion]
def write_questions(questions: list[EvalQuestion], path: Path) -> None
def read_questions(path: Path) -> list[EvalQuestion]
# metrics.py
def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float
def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float
def answer_correct(answer_text: str, keywords: list[str]) -> bool
def aggregate(rows: list[dict]) -> dict      # overall + per-category
# runner.py
def run_eval(pipeline: EHRRagPipeline, questions: list[EvalQuestion], k: int | None = None, mode: str | None = None,
             limit: int | None = None, generate: bool = True) -> dict
def compare_retrievers(pipeline: EHRRagPipeline, questions: list[EvalQuestion],
                       modes: tuple[str, ...] = ("random", "bm25", "dense", "hybrid")) -> dict
def write_report(report: dict, reports_dir: Path, name: str) -> tuple[Path, Path]   # JSON + Markdown
```
Categories and expected behaviour are in `PLAN.md` §5. Gold doc ids come from the deterministic id helpers plus note
`mentions`. Leakage = any retrieved chunk whose `patient_id` differs from the question's `patient_id` (must be 0).
