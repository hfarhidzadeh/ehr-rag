"""Tests for ehr_rag.evaluation.runner on a small generated dataset (tfidf + extractive, no network).

Besides the happy path, the control checks themselves get negative controls: a leaking retriever, invalid
citations and a "random" retriever that is as good as the real ones must each turn a check to FAIL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehr_rag import cli
from ehr_rag.config import Settings
from ehr_rag.evaluation import runner
from ehr_rag.evaluation.questions import build_questions
from ehr_rag.generation import GenerationResult
from ehr_rag.pipeline import EHRRagPipeline
from ehr_rag.synth.generator import generate_dataset
from ehr_rag.types import Citation, EvalQuestion, RetrievedChunk


def _settings(root: Path) -> Settings:
    return Settings().with_overrides(
        data_dir=root / "data", index_dir=root / "index", reports_dir=root / "reports",
        embedder="tfidf", llm="extractive", tfidf_dim=64,
    )


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("runner")
    settings = _settings(root)
    dataset = generate_dataset(n_patients=4, seed=7)
    dataset.write(settings.data_dir)
    pipe = EHRRagPipeline.build_index(settings)
    questions = build_questions(dataset, pipe.chunks, seed=settings.seed)
    return settings, pipe, questions


@pytest.fixture(scope="module")
def full_report(setup):
    _, pipe, questions = setup
    return runner.run_eval(pipe, questions, k=6, mode="hybrid")


@pytest.fixture(scope="module")
def compare_report(setup):
    _, pipe, questions = setup
    return runner.compare_retrievers(pipe, questions)


def _status(report: dict, prefix: str) -> str:
    matches = [c["status"] for c in report["controls"] if c["name"].startswith(prefix)]
    assert matches, f"no control check starting with {prefix!r}"
    return matches[0]


# --------------------------------------------------------------------------- run_eval


def test_run_eval_structure_and_rows(full_report, setup):
    _, _, questions = setup
    r = full_report
    assert r["kind"] == "eval" and set(r) >= {"config", "summary", "rows", "failures", "controls"}
    cfg = r["config"]
    assert (cfg["embedder"], cfg["mode"], cfg["k"], cfg["llm"], cfg["generate"]) == ("tfidf", "hybrid", 6,
                                                                                     "extractive", True)
    assert cfg["n_questions"] == len(questions) and cfg["synthetic"] is True
    assert cfg["index"]["n_patients"] == 4
    assert len(r["rows"]) == len(questions)
    assert [row["qid"] for row in r["rows"]] == [q.qid for q in questions]
    for row, q in zip(r["rows"], questions):
        assert row["question"] == q.question and row["category"] == q.category
        assert 0 < len(row["retrieved_doc_ids"]) <= 6
        assert set(row["retrieved_patient_ids"]) == {q.patient_id}
        assert isinstance(row["abstained"], bool) and row["answer_text"]
    assert set(r["summary"]) == {"overall", "by_control", "by_category"}
    assert set(r["summary"]["by_category"]) == {q.category for q in questions}


def test_run_eval_controls_pass_on_real_pipeline(full_report):
    o = full_report["summary"]["overall"]
    assert o["leakage_rate"] == 0.0 and o["leaked_chunks"] == 0
    assert o["citation_validity"] == 1.0
    assert _status(full_report, "leakage_rate == 0") == "PASS"
    assert _status(full_report, "citation_validity") == "PASS"
    assert _status(full_report, "correct_abstention_rate") == "INFO"
    neg = full_report["summary"]["by_control"]["negative"]
    assert neg["correct_abstention_rate"] is not None
    assert runner.control_status(full_report) in {"PASS", "FAIL"}


def test_run_eval_failures_have_examples(full_report):
    fails = full_report["failures"]
    assert set(fails) <= {r["category"] for r in full_report["rows"]}
    for cat, items in fails.items():
        assert 0 < len(items) <= runner.N_FAILURES
        for f in items:
            assert f["why"] and f["question"] and "retrieved_doc_ids" in f and "gold_doc_ids" in f
            assert "answer_text" in f


def test_run_eval_retrieval_only_and_limit(setup):
    _, pipe, questions = setup
    r = runner.run_eval(pipe, questions, k=4, mode="bm25", limit=5, generate=False)
    assert r["config"]["llm"] == "none" and r["config"]["generate"] is False
    assert len(r["rows"]) == 5 and r["config"]["n_questions_available"] == len(questions)
    assert all(row["answer_text"] is None and row["abstained"] is None for row in r["rows"])
    o = r["summary"]["overall"]
    for key in ("answer_accuracy", "false_abstention_rate", "correct_abstention_rate", "citation_validity"):
        assert o[key] is None, key
    assert all(len(row["retrieved_doc_ids"]) <= 4 for row in r["rows"])
    assert not any(c["metric"] == "citation_validity" for c in r["controls"])


def test_run_eval_rejects_unknown_mode(setup):
    _, pipe, questions = setup
    with pytest.raises(ValueError):
        runner.run_eval(pipe, questions, mode="psychic", generate=False)


# --------------------------------------------------------------------------- compare_retrievers


def test_compare_retrievers(compare_report, setup):
    _, _, questions = setup
    r = compare_report
    assert r["kind"] == "compare" and list(r["modes"]) == ["random", "bm25", "dense", "hybrid"]
    assert r["config"]["llm"] == "none" and r["config"]["n_questions"] == len(questions)
    for mode, run in r["runs"].items():
        assert run["config"]["mode"] == mode and len(run["rows"]) == len(questions)
        assert r["modes"][mode]["overall"]["leakage_rate"] == 0.0
    names = [c["name"] for c in r["controls"]]
    for mode in ("random", "bm25", "dense", "hybrid"):
        assert f"leakage_rate == 0 ({mode})" in names
    assert any(n.startswith("verbatim recall@1") and "(bm25)" in n for n in names)
    assert any(n.startswith("verbatim recall@1") and "(hybrid)" in n for n in names)
    assert any(n.startswith("random recall@k") for n in names)
    assert {row["mode"] for row in r["table"]} == set(r["modes"])
    # the real retrievers clearly beat the random baseline even on this tiny dataset
    pos = {m: s["by_control"]["positive"] for m, s in r["modes"].items()}
    assert pos["bm25"]["evidence_recall@k"] > pos["random"]["evidence_recall@k"]
    assert r["modes"]["bm25"]["by_category"]["verbatim"]["recall@1"] >= 0.75


# --------------------------------------------------------------------------- negative controls of the controls


class _StubPipeline:
    """Returns the same fixed chunks for every query and mode (optionally another patient's)."""

    def __init__(self, base, leak_from: str | None = None, generator=None):
        self.settings = base.settings
        self.manifest = base.manifest
        self._base = base
        self._leak_from = leak_from
        self.generator = generator or base.generator

    def retrieve(self, question, patient_id, k=None, mode=None, doc_types=None):
        hits = self._base.retrieve(question, patient_id, k=k, mode="bm25")
        if self._leak_from:
            foreign = self._base.retrieve(question, self._leak_from, k=1, mode="bm25")
            hits = foreign + hits[: max(0, (k or 6) - 1)]
        return [RetrievedChunk(h.chunk, h.score, i, mode or "bm25") for i, h in enumerate(hits, start=1)]

    def ask(self, question, patient_id, k=None, mode=None, doc_types=None):
        from ehr_rag.types import Answer

        contexts = self.retrieve(question, patient_id, k=k, mode=mode)
        res = self.generator.generate(question, patient_id, contexts)
        return Answer(question, patient_id, res.text, res.citations, contexts, res.abstained, res.mode,
                      latency_ms=1.0)


def test_leakage_check_fails_when_retriever_leaks(setup):
    _, pipe, questions = setup
    qs = [q for q in questions if q.patient_id == "P001"][:4]
    r = runner.run_eval(_StubPipeline(pipe, leak_from="P002"), qs, k=4, mode="bm25", generate=False)
    assert r["summary"]["overall"]["leakage_rate"] == 1.0
    assert _status(r, "leakage_rate == 0") == "FAIL" and runner.control_status(r) == "FAIL"
    assert all("leaked" in f["why"] for items in r["failures"].values() for f in items)


class _BadCitationGenerator:
    name = "badcite"

    def generate(self, question, patient_id, contexts):
        c = contexts[-1].chunk
        cit = Citation("S1", c.chunk_id, c.doc_id, c.doc_type, c.date, c.title, "x")   # wrong label for chunk
        return GenerationResult(text="see [S1]", citations=[cit], abstained=False, mode="extractive")


def test_citation_check_fails_on_mislabelled_citations(setup):
    _, pipe, questions = setup
    qs = [q for q in questions if q.category == "fact_lookup"][:3]
    r = runner.run_eval(_StubPipeline(pipe, generator=_BadCitationGenerator()), qs, k=4, mode="bm25")
    assert r["config"]["llm"] == "badcite"
    assert r["summary"]["overall"]["citation_validity"] == 0.0
    assert _status(r, "citation_validity") == "FAIL"


def test_random_baseline_check_fails_when_random_is_as_good(setup):
    _, pipe, questions = setup
    r = runner.compare_retrievers(_StubPipeline(pipe), questions, modes=("random", "bm25"))
    assert _status(r, "random recall@k") == "FAIL"


def test_verbatim_check_fails_when_retrieval_misses(setup):
    _, pipe, questions = setup
    verbatim = [q for q in questions if q.category == "verbatim"]
    broken = [EvalQuestion(q.qid, q.patient_id, q.question, q.category, q.expected_answer, q.answer_keywords,
                           [f"{q.patient_id}-no-such-doc"], q.metadata) for q in verbatim]
    r = runner.run_eval(pipe, broken, mode="bm25", generate=False)
    assert _status(r, "verbatim recall@1") == "FAIL"


# --------------------------------------------------------------------------- reports + CLI


def test_write_report_eval_and_compare(full_report, compare_report, tmp_path):
    jp, mp = runner.write_report(full_report, tmp_path / "rep", "eval_tfidf_hybrid_extractive")
    assert jp == tmp_path / "rep" / "eval_tfidf_hybrid_extractive.json" and mp.suffix == ".md"
    data = json.loads(jp.read_text())
    assert data["summary"]["overall"]["n"] == len(full_report["rows"])
    md = mp.read_text()
    for needle in ("# Evaluation report", "## Control checks", "**PASS**", "## Retrieval", "## Generation",
                   "| fact_lookup |", "| absent_fact |", "Worst examples", "Synthetic data only"):
        assert needle in md, needle
    jp2, mp2 = runner.write_report(compare_report, tmp_path / "rep", "compare_tfidf")
    md2 = mp2.read_text()
    for needle in ("# Retriever comparison", "| random |", "| hybrid |", "Recall@1 by category",
                   "Evidence recall@6 by category", "random recall@k"):
        assert needle in md2, needle
    assert json.loads(jp2.read_text())["kind"] == "compare"


def test_vocab_term_expansions():
    exp = runner.vocab_term_expansions()
    assert "s/p MI" in exp["heart attack"] and "HTN" in exp["high blood pressure"]
    assert all(v for v in exp.values())


def test_cli_eval_writes_reports(setup, monkeypatch, capsys):
    settings, _, _ = setup
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(settings.index_dir))
    monkeypatch.setenv("EHR_RAG_REPORTS_DIR", str(settings.reports_dir))
    monkeypatch.setenv("EHR_RAG_LLM", "extractive")
    monkeypatch.setenv("EHR_RAG_TFIDF_DIM", "64")
    assert cli.main(["eval", "--embedder", "tfidf", "--retrieval-only", "--limit", "6"]) == 0
    assert (settings.reports_dir / "eval_tfidf_hybrid_none.json").exists()
    assert cli.main(["eval", "--embedder", "tfidf", "--mode", "bm25", "--limit", "6"]) == 0
    assert (settings.reports_dir / "eval_tfidf_bm25_extractive.md").exists()
    assert cli.main(["eval", "--embedder", "tfidf", "--compare", "--limit", "8"]) == 0
    assert (settings.reports_dir / "compare_tfidf.md").exists()
    out = capsys.readouterr().out
    assert "[PASS] leakage_rate == 0 (random)" in out and "random" in out
    assert cli.main(["eval", "--embedder", "tfidf", "--vocab-expansions", "--limit", "4"]) == 0
    assert (settings.reports_dir / "eval_tfidf_hybrid_extractive-vocab.json").exists()
