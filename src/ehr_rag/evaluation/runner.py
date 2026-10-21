"""Evaluation runner: gold questions -> retrieval (+ generation) -> metric rows -> report with control checks.

Report structure (``run_eval``)::

    {"kind": "eval",
     "config":   {embedder, mode, k, llm, model, generate, n_questions, rrf_k, candidate_pool, index, ...},
     "summary":  metrics.aggregate(rows)          # overall / by_control / by_category
     "controls": [{name, metric, category, value, threshold, op, status, detail}]   # status PASS|FAIL|INFO|N/A
     "rows":     [one metrics row per question, plus question/expected/retrieved_chunk_ids/...]
     "failures": {category: [worst examples: qid, question, expected, gold, retrieved doc ids, answer, why]}}

``compare_retrievers`` runs ``run_eval(generate=False)`` once per retrieval mode and returns
``{"kind": "compare", "config", "modes": {mode: summary}, "table": [...], "controls": [...], "runs": {mode: report}}``.

Control checks (PLAN.md section 5):

* ``verbatim recall@1 >= 0.9`` for ``bm25`` and ``hybrid`` (the question quotes a chunk; lexical retrieval must
  find it first - a pipeline sanity check).
* ``leakage_rate == 0`` for every mode (patient isolation).
* random baseline: ``recall@k`` of ``random`` must be at least ``RANDOM_MARGIN`` below the best real retriever
  (also checked on ``primary_recall@k`` and ``evidence_recall@k``, which are not inflated by facts repeated in
  every note or by unrelated chunks of a gold note).
* ``correct_abstention_rate`` on negatives is reported (no threshold: it measures the generator).
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ehr_rag.evaluation.metrics import CATEGORY_ORDER, aggregate, answer_correct, make_row, reciprocal_rank
from ehr_rag.types import RETRIEVAL_MODES, EvalQuestion

logger = logging.getLogger(__name__)

VERBATIM_MIN_RECALL_AT_1 = 0.9
VERBATIM_CHECK_MODES: tuple[str, ...] = ("bm25", "hybrid")
RANDOM_MARGIN = 0.2          # random recall@k must be at least this far below the best real retriever
N_FAILURES = 5               # worst examples kept per category
_ANSWER_CHARS = 600          # answer text kept in a failure example

# Columns shown in the Markdown tables (metric key, header, kind).
_RETRIEVAL_COLS: tuple[tuple[str, str], ...] = (
    ("recall@1", "R@1"), ("recall@3", "R@3"), ("recall@k", "R@k"), ("primary_recall@1", "pR@1"),
    ("primary_recall@k", "pR@k"), ("evidence_recall@1", "eR@1"), ("evidence_recall@k", "eR@k"), ("mrr", "MRR"),
)
_ANSWER_COLS: tuple[tuple[str, str], ...] = (
    ("answer_accuracy", "Acc"), ("answered_accuracy", "Acc|answered"), ("false_abstention_rate", "FalseAbst"),
    ("correct_abstention_rate", "CorrectAbst"), ("citation_validity", "CiteValid"),
)
_OTHER_COLS: tuple[tuple[str, str], ...] = (
    ("leakage_rate", "Leak"), ("latency_mean_ms", "ms mean"), ("latency_p95_ms", "ms p95"),
)


# --------------------------------------------------------------------------- helpers


def _generator_info(pipeline: Any) -> tuple[str, Optional[str]]:
    gen = getattr(pipeline, "generator", None)
    name = getattr(gen, "name", None) or type(gen).__name__
    model = getattr(gen, "model", None) if name == "claude" else None
    return str(name), model


def _index_info(pipeline: Any) -> dict[str, Any]:
    manifest = getattr(pipeline, "manifest", None) or {}
    keys = ("embedder", "dim", "model", "collection", "n_docs", "n_chunks", "n_patients", "data_seed",
            "chunk_max_chars", "chunk_overlap_sentences", "created_at")
    return {k: manifest.get(k) for k in keys if k in manifest}


def _isolation_count(pipeline: Any) -> int:
    return int(getattr(getattr(pipeline, "retriever", None), "isolation_violations", 0) or 0)


def _cut(text: Optional[str], limit: int = _ANSWER_CHARS) -> Optional[str]:
    if text is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _question_row(q: EvalQuestion, retrieved, answer, latency_ms: float, k: int) -> dict[str, Any]:
    row = make_row(q, retrieved, answer, latency_ms=latency_ms, k=k)
    row["question"] = q.question
    row["expected_answer"] = q.expected_answer
    row["retrieved_chunk_ids"] = [rc.chunk.chunk_id for rc in retrieved]
    row["gold_rank"] = next((i for i, d in enumerate(row["retrieved_doc_ids"], start=1) if d in set(q.gold_doc_ids)),
                            None)
    for key in ("fact_kind", "variant", "absent_kind", "source_kind", "source_patient_id", "key_term"):
        if key in q.metadata:
            row[key] = q.metadata[key]
    if answer is not None:
        row["generator_mode"] = answer.mode
        row["model"] = answer.model
        row["n_citations"] = len(answer.citations)
        row["warnings"] = list(answer.warnings)
        if q.category == "cross_patient":
            # Informational: the other patient's value (e.g. "20 mg") can occur by coincidence, so this is a
            # hint for manual review, not a leakage metric (leakage is measured on retrieved chunks).
            src_kw = list(q.metadata.get("source_answer_keywords") or [])
            row["source_fact_in_answer"] = bool(src_kw) and not answer.abstained and answer_correct(answer.text,
                                                                                                    src_kw)
    return row


def _failure_reason(row: dict[str, Any]) -> Optional[str]:
    """Why a row counts as a failure (None if it does not)."""
    reasons = []
    leaked = sum(1 for p in row.get("retrieved_patient_ids") or [] if p != row["patient_id"])
    if leaked:
        reasons.append(f"{leaked} leaked chunk(s)")
    if row["expected_abstain"]:
        if row.get("abstained") is False:
            reasons.append("answered a negative-control question (should abstain)")
    else:
        if row.get("gold_rank") is None:
            reasons.append(f"no gold doc in top {row['k']}")
        if row.get("abstained") is True:
            reasons.append("abstained on a positive question")
        elif row.get("abstained") is False and not row.get("answer_correct"):
            reasons.append("answer lacks every expected keyword")
        if row.get("citations_valid") is False:
            reasons.append("invalid or missing citations")
    return "; ".join(reasons) or None


def _severity(row: dict[str, Any]) -> tuple:
    """Sort key: worst first (answer wrong before retrieval miss order, then lowest reciprocal rank)."""
    rr = reciprocal_rank(row["retrieved_doc_ids"], row["gold_doc_ids"]) if row["gold_doc_ids"] else 1.0
    leaked = sum(1 for p in row.get("retrieved_patient_ids") or [] if p != row["patient_id"])
    wrong = 0
    if row.get("abstained") is not None:
        wrong = int(row["abstained"] != row["expected_abstain"] or
                    (not row["expected_abstain"] and not row.get("answer_correct")))
    return (-leaked, -wrong, rr, row["qid"])


def _failures(rows: list[dict[str, Any]], n: int = N_FAILURES) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for cat in CATEGORY_ORDER:
        bad = [(r, _failure_reason(r)) for r in rows if r["category"] == cat]
        bad = [(r, why) for r, why in bad if why]
        if not bad:
            continue
        bad.sort(key=lambda t: _severity(t[0]))
        out[cat] = [{
            "qid": r["qid"], "patient_id": r["patient_id"], "question": r["question"],
            "expected_answer": r["expected_answer"], "answer_keywords": r["answer_keywords"],
            "gold_doc_ids": r["gold_doc_ids"], "retrieved_doc_ids": r["retrieved_doc_ids"],
            "retrieved_chunk_ids": r["retrieved_chunk_ids"], "gold_rank": r["gold_rank"],
            "answer_text": _cut(r.get("answer_text")), "abstained": r.get("abstained"), "why": why,
            "n_failures_in_category": len(bad),
        } for r, why in bad[:n]]
    return out


def _check(name: str, value: Optional[float], threshold: Optional[float], op: str, *, metric: str,
           category: Optional[str] = None, mode: Optional[str] = None, detail: str = "") -> dict[str, Any]:
    if op == "report":
        status = "INFO" if value is not None else "N/A"
    elif value is None or threshold is None:
        status = "N/A"
    else:
        ok = {">=": value >= threshold - 1e-12, "<=": value <= threshold + 1e-12, "==": abs(value - threshold) < 1e-12}[op]
        status = "PASS" if ok else "FAIL"
    return {"name": name, "mode": mode, "metric": metric, "category": category, "value": value,
            "threshold": threshold, "op": op, "status": status, "detail": detail}


def _mode_controls(summary: dict[str, Any], mode: str, generate: bool) -> list[dict[str, Any]]:
    checks = [_check(f"leakage_rate == 0 ({mode})", summary["overall"]["leakage_rate"], 0.0, "==",
                     metric="leakage_rate", mode=mode,
                     detail=f"leaked chunks: {summary['overall']['leaked_chunks']}")]
    if mode in VERBATIM_CHECK_MODES:
        verb = summary["by_category"].get("verbatim", {})
        checks.append(_check(f"verbatim recall@1 >= {VERBATIM_MIN_RECALL_AT_1} ({mode})", verb.get("recall@1"),
                             VERBATIM_MIN_RECALL_AT_1, ">=", metric="recall@1", category="verbatim", mode=mode,
                             detail=f"n={verb.get('n', 0)}"))
    if generate:
        neg = summary["by_control"]["negative"]
        checks.append(_check(f"correct_abstention_rate on negatives ({mode})", neg.get("correct_abstention_rate"),
                             None, "report", metric="correct_abstention_rate", mode=mode,
                             detail=f"n={neg.get('n', 0)} (absent_fact + cross_patient)"))
        pos = summary["by_control"]["positive"]
        checks.append(_check(f"false_abstention_rate on positives ({mode})", pos.get("false_abstention_rate"), None,
                             "report", metric="false_abstention_rate", mode=mode, detail=f"n={pos.get('n', 0)}"))
        checks.append(_check(f"citation_validity == 1 ({mode})", summary["overall"].get("citation_validity"), 1.0,
                             "==", metric="citation_validity", mode=mode,
                             detail="every non-abstained answer cites retrieved chunks of the asked patient"))
    return checks


# --------------------------------------------------------------------------- run_eval


def run_eval(
    pipeline: Any,
    questions: list[EvalQuestion],
    k: int | None = None,
    mode: str | None = None,
    limit: int | None = None,
    generate: bool = True,
    *,
    n_failures: int = N_FAILURES,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Evaluate ``questions`` against ``pipeline`` (``generate=False``: retrieval only, no generator call).

    Returns ``{"kind", "config", "summary", "controls", "rows", "failures"}`` (see module docstring).
    Patient-isolation and synthetic-data errors from the pipeline propagate; they are never swallowed.
    """
    settings = pipeline.settings
    k = int(settings.top_k if k is None else k)
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
    qs = list(questions)[: limit] if limit is not None else list(questions)
    llm, model = _generator_info(pipeline) if generate else ("none", None)
    violations_before = _isolation_count(pipeline)

    t_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(qs, start=1):
        t0 = time.perf_counter()
        if generate:
            answer = pipeline.ask(q.question, q.patient_id, k=k, mode=mode)
            retrieved, latency = answer.retrieved, float(answer.latency_ms)
        else:
            answer = None
            retrieved = pipeline.retrieve(q.question, q.patient_id, k=k, mode=mode)
            latency = (time.perf_counter() - t0) * 1000.0
        rows.append(_question_row(q, retrieved, answer, latency, k))
        if progress is not None:
            progress(i, len(qs))
    elapsed = time.perf_counter() - t_start

    summary = aggregate(rows)
    dropped = _isolation_count(pipeline) - violations_before
    if generate:
        fallbacks = sum(1 for r in rows if any("failed" in w and "extractive" in w for w in r.get("warnings", [])))
        served = sorted({r.get("model") for r in rows if r.get("model")})
    else:
        fallbacks, served = 0, []
    config = {
        "embedder": getattr(settings, "embedder", None),
        "mode": mode,
        "k": k,
        "llm": llm,
        "model": model,
        "served_models": served,
        "generate": bool(generate),
        "n_questions": len(qs),
        "n_questions_available": len(questions),
        "rrf_k": getattr(settings, "rrf_k", None),
        "candidate_pool": getattr(settings, "candidate_pool", None),
        "index": _index_info(pipeline),
        "isolation_violations_dropped": dropped,
        "llm_fallbacks_to_extractive": fallbacks,
        "runtime_s": round(elapsed, 2),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": True,
    }
    if generate and llm == "extractive":
        gen = getattr(pipeline, "generator", None)
        config["term_expansions"] = bool(getattr(gen, "term_expansions", None))
    controls = _mode_controls(summary, mode, generate)
    if dropped:
        controls.append(_check(f"retriever dropped wrong-patient chunks ({mode})", float(dropped), 0.0, "==",
                               metric="isolation_violations", mode=mode,
                               detail="the Retriever guard removed chunks of other patients before ranking"))
    logger.info("Evaluated %d questions (embedder=%s mode=%s k=%d llm=%s) in %.1fs", len(qs), config["embedder"],
                mode, k, llm, elapsed)
    return {
        "kind": "eval",
        "config": config,
        "summary": summary,
        "controls": controls,
        "rows": rows,
        "failures": _failures(rows, n_failures),
    }


# --------------------------------------------------------------------------- compare_retrievers


def compare_retrievers(
    pipeline: Any,
    questions: list[EvalQuestion],
    modes: tuple[str, ...] = ("random", "bm25", "dense", "hybrid"),
    *,
    k: int | None = None,
    limit: int | None = None,
) -> dict:
    """Retrieval-only evaluation of each mode, side by side, with the control checks."""
    k = int(pipeline.settings.top_k if k is None else k)
    runs = {m: run_eval(pipeline, questions, k=k, mode=m, limit=limit, generate=False) for m in modes}
    summaries = {m: r["summary"] for m, r in runs.items()}

    table: list[dict[str, Any]] = []
    for m, s in summaries.items():
        groups = [("overall", s["overall"]), ("positive", s["by_control"]["positive"])]
        groups += [(c, s["by_category"][c]) for c in s["by_category"]]
        for group, metrics in groups:
            table.append({"mode": m, "group": group, **{key: metrics.get(key) for key, _ in
                                                       _RETRIEVAL_COLS + _OTHER_COLS}, "n": metrics.get("n")})

    controls: list[dict[str, Any]] = []
    for m, s in summaries.items():
        controls += _mode_controls(s, m, generate=False)
    real = [m for m in modes if m != "random"]
    if "random" in modes and real:
        for metric in ("recall@k", "primary_recall@k", "evidence_recall@k"):
            best_mode = max(real, key=lambda m: summaries[m]["by_control"]["positive"].get(metric) or -1.0)
            best = summaries[best_mode]["by_control"]["positive"].get(metric)
            rand = summaries["random"]["by_control"]["positive"].get(metric)
            threshold = None if best is None else round(best - RANDOM_MARGIN, 6)
            controls.append(_check(
                f"random {metric} <= best real {metric} - {RANDOM_MARGIN}", rand, threshold, "<=", metric=metric,
                mode="random", detail=f"best real retriever: {best_mode} ({_fmt(best)}); positives only"))

    first = next(iter(runs.values()))
    config = {key: first["config"].get(key) for key in ("embedder", "k", "n_questions", "n_questions_available",
                                                        "rrf_k", "candidate_pool", "index")}
    config.update({"modes": list(modes), "generate": False, "llm": "none",
                   "runtime_s": round(sum(r["config"]["runtime_s"] for r in runs.values()), 2),
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "synthetic": True})
    return {"kind": "compare", "config": config, "modes": summaries, "table": table, "controls": controls,
            "runs": runs}


# --------------------------------------------------------------------------- configurations


def vocab_term_expansions() -> dict[str, list[str]]:
    """``{lay_term: [abbreviation, *note_terms]}`` from the synthetic vocabulary.

    For the separate "vocab-expanded" extractive configuration. It reuses the vocabulary the semantic_synonym
    questions are built from, so its scores on that category are an upper bound, not a fair comparison.
    """
    from ehr_rag.synth import vocab as V

    out: dict[str, list[str]] = {}
    for spec in V.CONDITIONS.values():
        terms = list(dict.fromkeys([spec.abbreviation, *spec.note_terms]))
        for lay in spec.lay_terms:
            out.setdefault(lay, [])
            out[lay] += [t for t in terms if t not in out[lay]]
    return out


# --------------------------------------------------------------------------- reports


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.0f}" if key.startswith("latency") else f"{value:.3f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _metric_table(groups: list[tuple[str, dict[str, Any]]], cols: tuple[tuple[str, str], ...],
                  label: str = "group") -> list[str]:
    headers = [label, "n"] + [h for _, h in cols]
    rows = [[g, _fmt(m.get("n"))] + [_fmt(m.get(key), key) for key, _ in cols] for g, m in groups]
    return _md_table(headers, rows)


def _controls_md(controls: list[dict[str, Any]]) -> list[str]:
    rows = []
    for c in controls:
        thr = "-" if c["threshold"] is None else f"{c['op']} {_fmt(c['threshold'])}"
        rows.append([f"**{c['status']}**", c["name"], _fmt(c["value"]), thr, c.get("detail") or ""])
    return _md_table(["status", "check", "value", "threshold", "detail"], rows)


def _summary_groups(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    groups = [("overall", summary["overall"]), ("positive", summary["by_control"]["positive"]),
              ("negative", summary["by_control"]["negative"])]
    return groups + [(c, m) for c, m in summary["by_category"].items()]


def _config_md(config: dict[str, Any]) -> list[str]:
    lines = []
    for key, value in config.items():
        if key == "index" and isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        lines.append(f"- **{key}**: {value}")
    return lines


_METRIC_HELP = (
    "Metric key: R@1/R@3/R@k = any gold document in the top 1/3/k chunks (positives only); pR@1/pR@k = the same "
    "for the canonical gold document only (structured list, lab panel or quoted note); eR@1/eR@k = a retrieved "
    "chunk of a gold document whose own text contains an expected keyword (chunk-level); MRR = mean reciprocal "
    "rank of the first gold document; Acc = positives answered without abstaining and containing an expected "
    "keyword; Acc|answered = Acc over non-abstained positives; FalseAbst = positives that abstained; "
    "CorrectAbst = negatives that abstained; CiteValid = answers whose citations all point at retrieved chunks "
    "of the asked patient; Leak = questions with any retrieved chunk of another patient. '-' = not applicable."
)


def _eval_md(report: dict[str, Any], name: str) -> list[str]:
    cfg, summary = report["config"], report["summary"]
    lines = [f"# Evaluation report: {name}", "",
             "Synthetic data only (generated by `ehr_rag.synth`). Every number below describes this synthetic "
             "benchmark, not clinical performance; a person must review AI output.", "", "## Configuration", ""]
    lines += _config_md(cfg)
    lines += ["", "## Control checks", ""] + _controls_md(report["controls"])
    lines += ["", "## Retrieval", ""] + _metric_table(_summary_groups(summary), _RETRIEVAL_COLS)
    if cfg.get("generate"):
        lines += ["", "## Generation", ""] + _metric_table(_summary_groups(summary), _ANSWER_COLS)
    lines += ["", "## Leakage and latency", ""] + _metric_table(_summary_groups(summary), _OTHER_COLS)
    lines += ["", _METRIC_HELP, "", "## Worst examples per category", ""]
    if not report.get("failures"):
        lines.append("No failures.")
    for cat, items in report.get("failures", {}).items():
        lines += [f"### {cat} ({items[0]['n_failures_in_category']} failing)", ""]
        for f in items:
            lines.append(f"- **{f['qid']}** - {f['question']}")
            lines.append(f"  - why: {f['why']}")
            if f.get("expected_answer") is not None:
                lines.append(f"  - expected: {f['expected_answer']} (keywords: {', '.join(f['answer_keywords'])})")
            lines.append(f"  - gold: {', '.join(f['gold_doc_ids'][:6]) or '- (negative control)'}"
                         f"{' ...' if len(f['gold_doc_ids']) > 6 else ''}; gold rank: {_fmt(f['gold_rank'])}")
            lines.append(f"  - retrieved: {', '.join(f['retrieved_chunk_ids'])}")
            if f.get("answer_text") is not None:
                ans = " / ".join(line.strip() for line in f["answer_text"].splitlines() if line.strip())
                lines.append(f"  - answer (abstained={f['abstained']}): {ans}")
        lines.append("")
    return lines


def _compare_md(report: dict[str, Any], name: str) -> list[str]:
    cfg, modes = report["config"], report["modes"]
    lines = [f"# Retriever comparison: {name}", "",
             "Retrieval only (no answer generation) on the synthetic gold questions; one run per mode with the "
             "same questions and k.", "", "## Configuration", ""]
    lines += _config_md(cfg)
    lines += ["", "## Control checks", ""] + _controls_md(report["controls"])
    lines += ["", "## Positives (all positive categories)", ""]
    lines += _metric_table([(m, s["by_control"]["positive"]) for m, s in modes.items()], _RETRIEVAL_COLS, "mode")
    cats = []
    for s in modes.values():
        cats += [c for c in s["by_category"] if c not in cats]
    for key, header in (("recall@1", "Recall@1"), ("recall@k", f"Recall@{cfg.get('k')}"),
                        ("primary_recall@k", f"Primary recall@{cfg.get('k')}"),
                        ("evidence_recall@k", f"Evidence recall@{cfg.get('k')}"), ("mrr", "MRR")):
        lines += ["", f"## {header} by category", ""]
        headers = ["mode"] + [c for c in cats if c in ("fact_lookup", "temporal", "semantic_synonym", "verbatim")]
        rows = [[m] + [_fmt(s["by_category"].get(c, {}).get(key)) for c in headers[1:]] for m, s in modes.items()]
        lines += _md_table(headers, rows)
    lines += ["", "## Leakage and latency", ""]
    lines += _metric_table([(m, s["overall"]) for m, s in modes.items()], _OTHER_COLS, "mode")
    lines += ["", _METRIC_HELP, ""]
    return lines


def write_report(report: dict, reports_dir: Path, name: str) -> tuple[Path, Path]:
    """Write ``<reports_dir>/<name>.json`` (full report) and ``<name>.md`` (readable summary)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{name}.json"
    md_path = reports_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = _compare_md(report, name) if report.get("kind") == "compare" else _eval_md(report, name)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote report %s and %s", json_path, md_path)
    return json_path, md_path


def control_status(report: dict) -> str:
    """``"PASS"`` if no control check failed, else ``"FAIL"`` (INFO / N/A checks do not count)."""
    return "FAIL" if any(c["status"] == "FAIL" for c in report.get("controls", [])) else "PASS"


__all__ = ["run_eval", "compare_retrievers", "write_report", "control_status", "vocab_term_expansions",
           "RANDOM_MARGIN", "VERBATIM_MIN_RECALL_AT_1"]
