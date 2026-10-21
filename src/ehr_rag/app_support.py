"""Streamlit-free helpers for ``app/streamlit_app.py``: index discovery, LLM status, example questions,
query-term highlighting, 2D PCA projection of chunk embeddings and evaluation-report tables.

Everything here is plain Python / numpy / pandas so it can be unit-tested without a Streamlit runtime.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ehr_rag.config import Settings
from ehr_rag.types import Answer, Chunk

logger = logging.getLogger(__name__)

EMBEDDERS: tuple[str, ...] = ("tfidf", "minilm")
LLM_CHOICES: tuple[str, ...] = ("auto", "claude", "extractive")
MANIFEST_FILENAME = "index_manifest.json"     # mirrors pipeline.MANIFEST_FILENAME (kept import-light)

# Metric columns shown in the evaluation tab (key, header).
RETRIEVAL_METRICS: tuple[tuple[str, str], ...] = (
    ("recall@1", "R@1"), ("recall@k", "R@k"), ("primary_recall@k", "primaryR@k"),
    ("evidence_recall@1", "evidenceR@1"), ("evidence_recall@k", "evidenceR@k"), ("mrr", "MRR"),
)
ANSWER_METRICS: tuple[tuple[str, str], ...] = (
    ("answer_accuracy", "Accuracy"), ("answered_accuracy", "Acc | answered"),
    ("false_abstention_rate", "False abstention"), ("correct_abstention_rate", "Correct abstention"),
    ("citation_validity", "Citation validity"), ("leakage_rate", "Leakage"),
)
CATEGORY_ORDER: tuple[str, ...] = (
    "fact_lookup", "temporal", "semantic_synonym", "verbatim", "absent_fact", "cross_patient",
)
MODE_ORDER: tuple[str, ...] = ("random", "bm25", "dense", "hybrid")


# --------------------------------------------------------------------------- index / data discovery


def index_manifest(settings: Settings, embedder: str) -> Optional[dict[str, Any]]:
    """The ``index_manifest.json`` of ``<index_dir>/<embedder>/`` or None when the index is not built."""
    path = Path(settings.index_dir) / embedder / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable index manifest %s: %s", path, exc)
        return None


def built_embedders(settings: Settings) -> list[str]:
    """Embedders whose index manifest exists (the manifest is written last, so it marks a complete build)."""
    return [e for e in EMBEDDERS if index_manifest(settings, e) is not None]


def data_ready(settings: Settings) -> bool:
    """True when the synthetic dataset (``documents.jsonl``) exists in ``settings.data_dir``."""
    return (Path(settings.data_dir) / "documents.jsonl").exists()


def _env_prefix(settings: Settings, *, data: bool = True, index: bool = True) -> str:
    default = Settings()
    env = []
    if data and Path(settings.data_dir).resolve() != default.data_dir.resolve():
        env.append(f"EHR_RAG_DATA_DIR={settings.data_dir}")
    if index and Path(settings.index_dir).resolve() != default.index_dir.resolve():
        env.append(f"EHR_RAG_INDEX_DIR={settings.index_dir}")
    return " ".join(env) + (" " if env else "")


def setup_commands(settings: Settings, embedder: str = "tfidf") -> list[str]:
    """Shell commands that create what is missing: the dataset (if absent) and the index for ``embedder``."""
    cmds = []
    if not data_ready(settings):
        cmds.append(f"{_env_prefix(settings, index=False)}python -m ehr_rag generate")
    cmds.append(f"{_env_prefix(settings)}python -m ehr_rag ingest --embedder {embedder}")
    return cmds


def eval_commands(settings: Settings, embedders: Sequence[str] = ("tfidf",)) -> list[str]:
    """Commands that write the reports the evaluation tab reads."""
    prefix = _env_prefix(settings)
    if Path(settings.reports_dir).resolve() != Settings().reports_dir.resolve():
        prefix += f"EHR_RAG_REPORTS_DIR={settings.reports_dir} "
    out = []
    for emb in embedders or ("tfidf",):
        out.append(f"{prefix}python -m ehr_rag eval --embedder {emb} --compare")
        out.append(f"{prefix}python -m ehr_rag eval --embedder {emb} --llm extractive")
    return out


# --------------------------------------------------------------------------- LLM status / generators


def llm_status(llm: str, model: str) -> dict[str, Any]:
    """What ``llm`` resolves to and whether Claude credentials are visible (env vars only, never their values)."""
    from ehr_rag.generation import has_claude_credentials

    creds = has_claude_credentials()
    llm = (llm or "auto").strip().lower()
    resolved = ("claude" if creds else "extractive") if llm == "auto" else llm
    if resolved == "claude" and creds:
        message = f"Claude credentials detected; answers come from {model}."
        level = "ok"
    elif resolved == "claude":
        message = (f"No Claude credentials found (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN); calls to {model} "
                   "will fail and fall back to the offline extractive generator.")
        level = "warn"
    elif llm == "auto":
        message = "No Claude credentials found; using the offline extractive generator (no LLM)."
        level = "info"
    else:
        message = ("Offline extractive generator (no LLM)"
                   + ("; Claude credentials are available." if creds else "; no Claude credentials found."))
        level = "info"
    return {"requested": llm, "resolved": resolved, "credentials": creds, "model": model if resolved == "claude"
            else None, "message": message, "level": level}


def make_generator(settings: Settings, llm: str):
    """A fresh generator for ``llm`` (the Claude client is created lazily on first use, so no network here)."""
    from ehr_rag.generation import get_generator

    return get_generator(settings.with_overrides(llm=llm))


def ask_with_generator(pipeline: Any, generator: Any, question: str, patient_id: str, *, k: int, mode: str,
                       doc_types: Optional[list[str]] = None) -> Answer:
    """``pipeline.ask`` with ``generator``, without mutating the (shared, cached) pipeline.

    A shallow copy shares the store, embedder and BM25 index but has its own ``generator`` attribute, so
    concurrent Streamlit sessions with different LLM choices do not interfere.
    """
    view = copy.copy(pipeline)
    view.generator = generator
    return view.ask(question, patient_id, k=k, mode=mode, doc_types=doc_types)


# --------------------------------------------------------------------------- raw tables / examples


def read_table(data_dir: Path, name: str) -> list[dict]:
    """Rows of ``<data_dir>/<name>.jsonl`` ([] when the file is missing)."""
    path = Path(data_dir) / f"{name}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class ExampleQuestion:
    question: str
    kind: str                 # fact_lookup | temporal | stop_reason | allergy | semantic_synonym | absent
    expect: str               # "answer" | "abstain"
    hint: str = ""            # what the chart says (or why it should abstain)


@dataclass
class PatientFacts:
    """The rows of one patient used to tailor the example questions."""

    patient_id: str
    medications: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    allergies: list[dict] = field(default_factory=list)
    labs: list[dict] = field(default_factory=list)

    @classmethod
    def from_data_dir(cls, data_dir: Path, patient_id: str) -> "PatientFacts":
        def rows(name: str) -> list[dict]:
            return [r for r in read_table(data_dir, name) if r.get("patient_id") == patient_id]

        return cls(patient_id, rows("medications"), rows("conditions"), rows("allergies"), rows("labs"))


_LAB_PRIORITY = ("hba1c", "ldl", "creatinine", "egfr", "potassium", "tsh", "bnp", "inr", "hemoglobin", "sodium")


def _stable_index(key: str, n: int) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % max(1, n)


def _absent_from(term: str, text: str) -> bool:
    from ehr_rag.evaluation.questions import term_in_text

    parts = [term] + ([p for p in term.split("-") if len(p) >= 4] if "-" in term else [])
    return not any(term_in_text(p, text) for p in parts)


def example_questions(facts: PatientFacts, patient_text: str, max_n: int = 6) -> list[ExampleQuestion]:
    """Questions tailored to one patient's real (synthetic) data, including one that should abstain.

    ``patient_text`` is the concatenated text of the patient's indexed chunks; the abstain example names a
    medication that occurs nowhere in it.
    """
    from ehr_rag.evaluation import questions as Q
    from ehr_rag.synth import vocab as V

    out: list[ExampleQuestion] = []
    active = sorted((m for m in facts.medications if m.get("status") == "active"),
                    key=lambda m: (m.get("name", ""), m.get("med_id", "")))
    stopped = sorted((m for m in facts.medications if m.get("status") == "discontinued"),
                     key=lambda m: (m.get("end_date") or "", m.get("med_id", "")), reverse=True)
    if active:
        m = active[0]
        out.append(ExampleQuestion(Q.Q_MED_DOSE.format(name=m["name"]), "fact_lookup", "answer",
                                   f"{m['name']} {m['dose']} {m.get('route', '')} {m.get('frequency', '')}".strip()))

    by_test: dict[str, list[dict]] = {}
    for row in facts.labs:
        by_test.setdefault(row["test_key"], []).append(row)
    repeated = [k for k in _LAB_PRIORITY if len({r["date"] for r in by_test.get(k, [])}) >= 2]
    lab_key = repeated[0] if repeated else next((k for k in _LAB_PRIORITY if k in by_test), None)
    if lab_key and lab_key in Q.LAB_QUESTION_NAMES:
        latest = max(by_test[lab_key], key=lambda r: r["date"])
        from ehr_rag.synth.generator import format_lab_value

        unit = latest.get("unit") or ""
        out.append(ExampleQuestion(Q.Q_LAB_LATEST.format(lab=Q.LAB_QUESTION_NAMES[lab_key]), "temporal", "answer",
                                   f"{format_lab_value(lab_key, latest['value'])} {unit} on {latest['date']}"
                                   .replace("  ", " ")))
    if stopped:
        m = stopped[0]
        out.append(ExampleQuestion(Q.Q_STOP.format(name=m["name"], dose=m["dose"]), "stop_reason", "answer",
                                   m.get("discontinue_reason") or ""))
    if facts.allergies:
        a = sorted(facts.allergies, key=lambda r: r["substance"])[0]
        out.append(ExampleQuestion(Q.Q_ALLERGY.format(substance=a["substance"]), "allergy", "answer",
                                   f"{a['reaction']} ({a.get('severity', '')})"))
    else:
        out.append(ExampleQuestion("Does the patient have any drug allergies?", "allergy", "answer",
                                   "No known drug allergies (NKDA)"))

    active_conditions = [c for c in facts.conditions if c.get("status") == "active"]
    for c in sorted(active_conditions, key=lambda c: (c.get("onset_date") or "", c["condition_key"])):
        spec = V.CONDITIONS.get(c["condition_key"])
        if spec is None or not spec.lay_terms:
            continue
        lay = spec.lay_terms[0]
        meds = [m["name"] for m in active if m.get("indication") == c["condition_key"]]
        if meds:
            out.append(ExampleQuestion(Q.Q_SYN_MEDS.format(lay=lay), "semantic_synonym", "answer",
                                       f"{', '.join(meds)} (chart says {spec.abbreviation}; lay wording "
                                       "is never in the chart)"))
        else:
            out.append(ExampleQuestion(Q.Q_SYN_HAS.format(lay=lay), "semantic_synonym", "answer",
                                       f"yes: {spec.abbreviation} (lay wording is never in the chart)"))
        break

    candidates = [name for name in sorted(V.MEDICATIONS) if _absent_from(name, patient_text)]
    if candidates:
        name = candidates[_stable_index(facts.patient_id, len(candidates))]
        absent = ExampleQuestion(Q.Q_MED_DOSE.format(name=name), "absent", "abstain",
                                 f"{name} appears nowhere in this patient's chart; the system should abstain")
        out = out[: max(0, max_n - 1)] + [absent]
    return out[:max_n]


# --------------------------------------------------------------------------- highlighting / rendering


def query_terms(question: str) -> list[str]:
    """Content terms of ``question`` (the BM25 tokenizer, de-duplicated; 1-char non-digits dropped)."""
    from ehr_rag.retrieval import tokenize

    seen: list[str] = []
    for tok in tokenize(question):
        if (len(tok) >= 2 or tok.isdigit()) and tok not in seen:
            seen.append(tok)
    return seen


def _terms_regex(terms: Iterable[str]) -> Optional[re.Pattern[str]]:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return None
    alt = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?:e?s)?(?![A-Za-z0-9])", re.IGNORECASE)


def highlight_html(text: str, terms: Iterable[str]) -> str:
    """HTML-escaped ``text`` with every whole-word occurrence of ``terms`` wrapped in ``<mark>``."""
    pattern = _terms_regex(terms)
    if pattern is None:
        return html.escape(text)
    parts: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append(f"<mark>{html.escape(m.group(0))}</mark>")
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]<>#+|$~])")
_CITE_RE = re.compile(r"\\\[(S\d+(?:\s*,\s*S\d+)*)\\\]")


def answer_markdown(text: str) -> str:
    """Answer text as Markdown: special characters escaped (no accidental headers, lists or LaTeX), ``[S#]``
    markers in bold, ``- `` bullets kept and every other line ending in a hard line break."""
    out_lines = []
    for line in (text or "").splitlines():
        m = re.match(r"^(\s*)- (.*)$", line)
        bullet, body = (f"{m.group(1)}- ", m.group(2)) if m else ("", line)
        esc = _MD_SPECIAL_RE.sub(r"\\\1", body)
        esc = re.sub(r"^(\s*\d+)([.)])", r"\1\\\2", esc)        # "4. Pulmonary ..." is not an ordered list
        esc = _CITE_RE.sub(lambda mm: f"**\\[{mm.group(1)}\\]**", esc)
        out_lines.append(bullet + esc if bullet else esc + "  ")
    return "\n".join(out_lines).rstrip()


def chunk_body(chunk: Chunk) -> str:
    """Chunk text without its ``[Patient ... | Section: ...]`` header line."""
    first, _, rest = chunk.text.partition("\n")
    return rest if first.startswith("[") and first.endswith("]") else chunk.text


def snippet(chunk: Chunk, limit: int = 160) -> str:
    body = " ".join(chunk_body(chunk).split())
    return body if len(body) <= limit else body[: limit - 3].rstrip() + "..."


# --------------------------------------------------------------------------- vector-space projection


@dataclass
class Projection:
    chunks: list[Chunk]
    coords: np.ndarray            # (n, 2)
    explained: tuple[float, float]
    model: Any = None             # fitted sklearn PCA, or None when n < 2

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Project one embedding (e.g. the query) into the same 2D space."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self.model is None:
            return np.zeros(2, dtype=np.float32)
        xy = self.model.transform(vec)[0]
        return np.pad(xy, (0, 2 - len(xy))).astype(np.float32)


def project_chunks(chunks: list[Chunk], vectors: np.ndarray, seed: int = 0) -> Projection:
    """2D PCA (sklearn) of chunk embeddings; degenerate inputs (n < 2) give zero coordinates."""
    from sklearn.decomposition import PCA

    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(chunks)
    if n < 2 or vectors.ndim != 2 or vectors.shape[1] < 1:
        return Projection(list(chunks), np.zeros((n, 2), dtype=np.float32), (0.0, 0.0), None)
    n_comp = min(2, n, vectors.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed, svd_solver="full")
    xy = pca.fit_transform(vectors)
    if n_comp < 2:
        xy = np.hstack([xy, np.zeros((n, 2 - n_comp))])
    ratio = list(pca.explained_variance_ratio_) + [0.0] * (2 - n_comp)
    return Projection(list(chunks), xy.astype(np.float32), (float(ratio[0]), float(ratio[1])), pca)


def projection_frame(proj: Projection, retrieved_ids: Sequence[str] = (), selected_patient: str | None = None,
                     snippet_chars: int = 140) -> pd.DataFrame:
    """One row per chunk for the scatter: coordinates, doc_type, patient, hover text, retrieved flag/rank."""
    ranks = {cid: i + 1 for i, cid in enumerate(retrieved_ids)}
    rows = []
    for c, (x, y) in zip(proj.chunks, proj.coords):
        rows.append({
            "x": float(x), "y": float(y), "chunk_id": c.chunk_id, "doc_type": c.doc_type,
            "patient_id": c.patient_id,
            "patient_group": ("selected patient" if c.patient_id == selected_patient else "other patients"),
            "title": c.title, "date": c.date, "section": c.section or "",
            "snippet": snippet(c, snippet_chars), "retrieved_rank": ranks.get(c.chunk_id),
        })
    return pd.DataFrame(rows, columns=["x", "y", "chunk_id", "doc_type", "patient_id", "patient_group", "title",
                                       "date", "section", "snippet", "retrieved_rank"])


# --------------------------------------------------------------------------- evaluation reports


def list_reports(reports_dir: Path) -> list[dict[str, Any]]:
    """Summary of every ``reports/*.json`` written by the runner (name, path, kind, config, status)."""
    from ehr_rag.evaluation.runner import control_status

    out = []
    for path in sorted(Path(reports_dir).glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable report %s: %s", path, exc)
            continue
        if not isinstance(report, dict) or report.get("kind") not in ("eval", "compare"):
            continue
        cfg = report.get("config", {}) or {}
        out.append({
            "name": path.stem, "path": str(path), "kind": report["kind"], "embedder": cfg.get("embedder"),
            "mode": cfg.get("mode") if report["kind"] == "eval" else ", ".join(cfg.get("modes", []) or []),
            "llm": cfg.get("llm"), "term_expansions": bool(cfg.get("term_expansions")),
            "n_questions": cfg.get("n_questions"), "created_at": cfg.get("created_at"),
            "controls": control_status(report), "mtime": path.stat().st_mtime,
        })
    return out


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_groups(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``aggregate`` output as ordered (group, metrics) pairs: overall, positive, negative, categories."""
    groups: list[tuple[str, dict[str, Any]]] = [("overall", summary.get("overall", {}) or {})]
    for ctl in ("positive", "negative"):
        if (summary.get("by_control") or {}).get(ctl):
            groups.append((ctl, summary["by_control"][ctl]))
    by_cat = summary.get("by_category", {}) or {}
    for cat in list(CATEGORY_ORDER) + sorted(set(by_cat) - set(CATEGORY_ORDER)):
        if cat in by_cat:
            groups.append((cat, by_cat[cat]))
    return groups


def metrics_frame(summary: dict[str, Any], metrics: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Per-group metric table (rows: overall/positive/negative/categories; columns: n + ``metrics``)."""
    rows = []
    for group, m in summary_groups(summary):
        row: dict[str, Any] = {"group": group, "n": m.get("n")}
        for key, header in metrics:
            val = m.get(key)
            row[header] = float(val) if isinstance(val, (int, float)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["group", "n"] + [h for _, h in metrics])


def controls_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Control checks as a table (status PASS | FAIL | INFO | N/A)."""
    rows = []
    for c in report.get("controls", []) or []:
        val = c.get("value")
        rows.append({"status": c.get("status"), "check": c.get("name"), "mode": c.get("mode"),
                     "value": float(val) if isinstance(val, (int, float)) else np.nan,
                     "threshold": f"{c.get('op', '')} {c.get('threshold')}" if c.get("threshold") is not None
                     else c.get("op", ""), "detail": c.get("detail", "")})
    return pd.DataFrame(rows, columns=["status", "check", "mode", "value", "threshold", "detail"])


def compare_frame(reports: dict[str, dict[str, Any]], group: str = "overall",
                  metrics: Sequence[str] = ("recall@1", "recall@k", "evidence_recall@k", "mrr")) -> pd.DataFrame:
    """Long table (embedder, mode, metric, value, n) from compare reports keyed by embedder."""
    rows = []
    for emb, report in reports.items():
        for t in report.get("table", []) or []:
            if t.get("group") != group:
                continue
            for metric in metrics:
                val = t.get(metric)
                if isinstance(val, (int, float)):
                    rows.append({"embedder": emb, "mode": t.get("mode"), "metric": metric, "value": float(val),
                                 "n": t.get("n")})
    df = pd.DataFrame(rows, columns=["embedder", "mode", "metric", "value", "n"])
    if not df.empty:
        order = {m: i for i, m in enumerate(MODE_ORDER)}
        df = df.sort_values(["embedder", "mode"], key=lambda s: s.map(order) if s.name == "mode" else s)
        df = df.reset_index(drop=True)
    return df


STATUS_COLORS: dict[str, str] = {"PASS": "#008300", "FAIL": "#e34948", "INFO": "#52514e", "N/A": "#8a8983"}


def html_table(df: pd.DataFrame, status_cols: Sequence[str] = (), digits: int = 2,
               css_class: str = "ehr-table") -> str:
    """Escaped HTML ``<table>`` for ``df`` (floats to ``digits``, NaN/None as "n/a", status cells coloured).

    The app renders tables this way because ``st.dataframe`` needs pyarrow, and the pinned environment's
    pyarrow build requires NumPy 2 (the project pins ``numpy<2``).
    """
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return html.escape(str(value))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        tds = []
        for col in df.columns:
            val = row[col]
            numeric = isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)
            style = ' style="text-align:right"' if numeric else ""
            if col in status_cols and str(val) in STATUS_COLORS:
                style = f' style="color:{STATUS_COLORS[str(val)]};font-weight:600"'
            tds.append(f"<td{style}>{cell(val)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="{css_class}-wrap"><table class="{css_class}"><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def compare_groups(report: dict[str, Any]) -> list[str]:
    """Groups present in a compare report's table, in display order."""
    present = {t.get("group") for t in report.get("table", []) or []}
    order = ["overall", "positive", "negative", *CATEGORY_ORDER]
    return [g for g in order if g in present] + sorted(g for g in present - set(order) if g)


__all__ = [
    "EMBEDDERS", "LLM_CHOICES", "ExampleQuestion", "PatientFacts", "Projection", "answer_markdown",
    "ask_with_generator", "built_embedders", "chunk_body", "compare_frame", "compare_groups", "controls_frame",
    "data_ready", "eval_commands", "example_questions", "highlight_html", "html_table", "index_manifest", "list_reports",
    "llm_status", "load_report", "make_generator", "metrics_frame", "project_chunks", "projection_frame",
    "query_terms", "read_table", "setup_commands", "snippet", "summary_groups",
]
