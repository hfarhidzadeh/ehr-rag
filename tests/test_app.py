"""Tests for the Streamlit app (``streamlit.testing.v1.AppTest``) and its helpers in ``ehr_rag.app_support``.

A small tfidf index (3 synthetic patients) and real runner reports are built once in a temp dir; the app is
pointed at them with ``EHR_RAG_*`` environment variables. Extractive generator only, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehr_rag import app_support as S
from ehr_rag.config import PROJECT_ROOT, Settings
from ehr_rag.evaluation.questions import build_questions
from ehr_rag.evaluation.runner import compare_retrievers, run_eval, write_report
from ehr_rag.generation import ExtractiveGenerator
from ehr_rag.pipeline import EHRRagPipeline
from ehr_rag.synth.generator import SyntheticDataset, generate_dataset
from ehr_rag.types import Chunk

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
TIMEOUT = 60


@pytest.fixture(scope="module")
def env_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    settings = Settings().with_overrides(
        data_dir=root / "data", index_dir=root / "index", reports_dir=root / "reports",
        embedder="tfidf", llm="extractive", tfidf_dim=32,
    )
    generate_dataset(n_patients=3, seed=11).write(settings.data_dir)
    pipe = EHRRagPipeline.build_index(settings, generator=ExtractiveGenerator())
    questions = build_questions(SyntheticDataset.load(settings.data_dir), pipe.chunks, seed=settings.seed)
    write_report(compare_retrievers(pipe, questions), settings.reports_dir, "compare_tfidf")
    write_report(run_eval(pipe, questions), settings.reports_dir, "eval_tfidf_hybrid_extractive")
    return {"root": root, "settings": settings, "pipe": pipe}


@pytest.fixture()
def app_env(env_root, monkeypatch):
    s = env_root["settings"]
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(s.data_dir))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(s.index_dir))
    monkeypatch.setenv("EHR_RAG_REPORTS_DIR", str(s.reports_dir))
    monkeypatch.setenv("EHR_RAG_EMBEDDER", "tfidf")
    monkeypatch.setenv("EHR_RAG_LLM", "extractive")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return env_root


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _texts(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


# --------------------------------------------------------------------------- AppTest


def test_app_renders_without_exceptions(app_env):
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    md = _texts(at.markdown)
    assert "SYNTHETIC DATA ONLY" in md and "NOT FOR CLINICAL USE" in md and "verify" in md
    assert [t.label for t in at.tabs] == ["Ask", "Vector space", "Evaluation"]
    pids = app_env["pipe"].patient_ids
    assert at.selectbox(key="patient_id").options == [f"{p} - {n}" for p, n in zip(
        pids, [r["name"] for r in app_env["pipe"].list_patients()])]
    assert at.selectbox(key="embedder").options == ["tfidf"]       # only built indexes
    modes = at.selectbox(key="mode").options
    assert len(modes) == 4 and any(m.startswith("random (baseline control") for m in modes)
    side = _texts(at.sidebar.markdown) + _texts(at.sidebar.info)
    summary = app_env["pipe"].patient_summary(pids[0])
    assert summary["demographics"]["name"] in side
    assert "Active problems" in side and "Active medications" in side and "Allergies" in side
    assert "extractive" in side                                     # LLM status line
    labels = [b.label for b in at.button]
    assert any(label.startswith("(should abstain)") for label in labels)
    assert "Ask" in labels
    # Evaluation tab read the runner reports
    assert "PASS" in md or "FAIL" in md
    assert "Retriever comparison" in _texts(at.subheader)


def test_ask_renders_answer_and_sources(app_env):
    pipe = app_env["pipe"]
    pid = pipe.patient_ids[0]
    med = pipe.patient_summary(pid)["active_medications"][0]
    at = _run_app()
    at.text_input(key="question").input(f"What dose of {med['name']} is the patient currently taking?")
    at.button(key="ask").click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = _texts(at.markdown)
    assert "**\\[S" in md                                            # inline citation markers
    assert med["dose"].split()[0] in md                              # the dose is quoted
    expanders = [e.label for e in at.expander if e.label.startswith("S")]
    assert len(expanders) == int(at.slider(key="top_k").value)
    assert any("cited" in label for label in expanders)
    assert "<mark>" in md                                            # query terms highlighted in sources
    assert "#about-extractive-mode" in md                            # extractive-mode note
    assert "Sources (retrieved excerpts)" in _texts(at.subheader)
    # the vector-space tab overlays the query and its retrieved chunks without errors
    at.radio(key="space_scope").set_value("All patients").run()
    at.radio(key="space_colour").set_value("Patient").run()
    assert not at.exception, [e.value for e in at.exception]


def test_abstain_example_shows_abstention_callout(app_env):
    at = _run_app()
    idx = next(i for i, b in enumerate(b for b in at.button if b.key and b.key.startswith("example_"))
               if b.label.startswith("(should abstain)"))
    at.button(key=f"example_{idx}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Abstained" in _texts(at.warning)


def test_missing_index_shows_setup_command(app_env, tmp_path, monkeypatch):
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "no_index"))
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    err = _texts(at.error)
    assert "No index is built yet" in err
    code = _texts(at.code)
    assert "python -m ehr_rag ingest --embedder tfidf" in code
    assert f"EHR_RAG_INDEX_DIR={tmp_path / 'no_index'}" in code
    assert "generate" not in code                                     # data exists, only the index is missing


def test_missing_data_and_index_shows_generate_command(app_env, tmp_path, monkeypatch):
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(tmp_path / "no_data"))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "no_index"))
    at = _run_app()
    assert not at.exception
    assert "No synthetic dataset and no index" in _texts(at.error)
    code = _texts(at.code)
    assert "python -m ehr_rag generate" in code and "python -m ehr_rag ingest" in code


def test_corrupt_index_shows_error_not_traceback(app_env, tmp_path, monkeypatch):
    bad = tmp_path / "bad_index" / "tfidf"
    bad.mkdir(parents=True)
    (bad / "index_manifest.json").write_text(json.dumps({"embedder": "tfidf", "created_at": "x"}))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "bad_index"))
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert "cannot be loaded" in _texts(at.error)
    assert "python -m ehr_rag ingest --embedder tfidf" in _texts(at.code)


def test_no_reports_shows_eval_command(app_env, tmp_path, monkeypatch):
    monkeypatch.setenv("EHR_RAG_REPORTS_DIR", str(tmp_path / "no_reports"))
    at = _run_app()
    assert not at.exception
    assert "No evaluation reports" in _texts(at.info)
    assert "python -m ehr_rag eval --embedder tfidf --compare" in _texts(at.code)


# --------------------------------------------------------------------------- helpers


def test_built_embedders_and_commands(env_root, tmp_path):
    s = env_root["settings"]
    assert S.built_embedders(s) == ["tfidf"]
    assert S.index_manifest(s, "minilm") is None
    empty = s.with_overrides(data_dir=tmp_path / "d", index_dir=tmp_path / "i")
    assert S.built_embedders(empty) == [] and not S.data_ready(empty)
    cmds = S.setup_commands(empty, "minilm")
    assert cmds[0].endswith("python -m ehr_rag generate") and f"EHR_RAG_DATA_DIR={tmp_path / 'd'}" in cmds[0]
    assert cmds[1].endswith("python -m ehr_rag ingest --embedder minilm")
    assert S.setup_commands(Settings(), "tfidf")[-1] == "python -m ehr_rag ingest --embedder tfidf"


def test_llm_status(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    st_ = S.llm_status("auto", "claude-opus-5-5")
    assert st_["resolved"] == "extractive" and not st_["credentials"] and st_["model"] is None
    st_ = S.llm_status("claude", "claude-opus-5-5")
    assert st_["level"] == "warn" and "fall back" in st_["message"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    st_ = S.llm_status("auto", "claude-opus-5-5")
    assert st_["resolved"] == "claude" and st_["model"] == "claude-opus-5-5" and st_["level"] == "ok"
    assert "test-not-a-real-key" not in st_["message"]
    assert S.llm_status("extractive", "m")["resolved"] == "extractive"


def test_ask_with_generator_does_not_mutate_pipeline(env_root):
    pipe = env_root["pipe"]
    before = pipe.generator
    gen = ExtractiveGenerator(max_sentences=1)
    pid = pipe.patient_ids[0]
    ans = S.ask_with_generator(pipe, gen, "What medications is the patient taking?", pid, k=3, mode="bm25")
    assert pipe.generator is before
    assert ans.patient_id == pid and len(ans.retrieved) == 3
    assert all(rc.chunk.patient_id == pid for rc in ans.retrieved)


def test_example_questions_are_tailored(env_root):
    s, pipe = env_root["settings"], env_root["pipe"]
    for pid in pipe.patient_ids:
        facts = S.PatientFacts.from_data_dir(s.data_dir, pid)
        text = "\n".join(c.text for c in pipe.bm25.chunks_for(pid))
        exs = S.example_questions(facts, text)
        assert 4 <= len(exs) <= 6
        abstain = [e for e in exs if e.expect == "abstain"]
        assert len(abstain) == 1 and abstain[0] is exs[-1]
        absent_med = abstain[0].question.split("dose of ")[1].split(" is the")[0]
        assert absent_med.lower() not in text.lower()
        active = {m["name"] for m in facts.medications if m["status"] == "active"}
        dose_q = next(e for e in exs if e.kind == "fact_lookup")
        assert any(f"dose of {name} " in dose_q.question for name in active)
        assert any(e.kind == "temporal" and "most recent" in e.question for e in exs)
        # deterministic
        assert [e.question for e in S.example_questions(facts, text)] == [e.question for e in exs]


def test_query_terms_and_highlight():
    terms = S.query_terms("What is the most recent HbA1c of 7.2 for s/p MI?")
    assert "hba1c" in terms and "7.2" in terms and "what" not in terms and "s" not in terms
    out = S.highlight_html("HbA1c 7.2 % <ref> on metformin; hba1cs", ["hba1c", "7.2", "metformin"])
    assert out.count("<mark>") == 4
    assert "&lt;ref&gt;" in out and "<ref>" not in out
    assert S.highlight_html("a < b", []) == "a &lt; b"
    assert "<mark>" not in S.highlight_html("metformin2", ["metformin"])        # whole words only


def test_answer_markdown_escapes_and_bolds_citations():
    md = S.answer_markdown("Relevant excerpts:\n- # MDD: $5 <7.0 [S2]\n4. Plan [S1, S3]")
    lines = md.splitlines()
    assert lines[0] == "Relevant excerpts:  "
    assert lines[1].startswith("- \\# MDD: \\$5 \\<7.0 **\\[S2\\]**")
    assert lines[2].startswith("4\\. Plan **\\[S1, S3\\]**")


def _chunk(i: int, pid: str = "P001", doc_type: str = "clinical_note") -> Chunk:
    return Chunk(chunk_id=f"{pid}-N01::{i:03d}", doc_id=f"{pid}-N01", patient_id=pid, doc_type=doc_type,
                 date="2025-01-01", title="Progress Note", section="PLAN" if i % 2 else None,
                 text=f"[Patient {pid} | Progress Note | 2025-01-01]\nbody text number {i} " + "x" * 200,
                 metadata={"synthetic": True})


def test_projection_roundtrip_and_frame():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(12, 8)).astype(np.float32)
    chunks = [_chunk(i, pid="P001" if i < 6 else "P002") for i in range(12)]
    proj = S.project_chunks(chunks, vecs)
    assert proj.coords.shape == (12, 2)
    assert 0 < proj.explained[0] <= 1 and proj.explained[0] >= proj.explained[1]
    np.testing.assert_allclose(proj.transform(vecs[3]), proj.coords[3], atol=1e-4)   # query uses same space
    df = S.projection_frame(proj, [chunks[4].chunk_id, chunks[1].chunk_id], selected_patient="P001")
    assert len(df) == 12
    assert df.set_index("chunk_id").loc[chunks[4].chunk_id, "retrieved_rank"] == 1
    assert df["retrieved_rank"].notna().sum() == 2
    assert set(df["patient_group"]) == {"selected patient", "other patients"}
    assert not df["snippet"].str.startswith("[Patient").any() and df["snippet"].str.len().max() <= 140
    # degenerate inputs
    one = S.project_chunks(chunks[:1], vecs[:1])
    assert one.coords.shape == (1, 2) and one.model is None and one.transform(vecs[0]).tolist() == [0.0, 0.0]
    assert S.project_chunks([], np.zeros((0, 0))).coords.shape == (0, 2)


def test_projection_on_store_embeddings(env_root):
    pipe = env_root["pipe"]
    pid = pipe.patient_ids[0]
    chunks, vecs = pipe.store.get_embeddings(pid)
    proj = S.project_chunks(chunks, vecs)
    assert len(proj.chunks) == len(chunks) == vecs.shape[0]
    q = proj.transform(pipe.embedder.embed_query("most recent creatinine"))
    assert q.shape == (2,) and np.isfinite(q).all()


def test_report_tables(env_root):
    s = env_root["settings"]
    reports = S.list_reports(s.reports_dir)
    assert {r["name"] for r in reports} == {"compare_tfidf", "eval_tfidf_hybrid_extractive"}
    assert all(r["controls"] in ("PASS", "FAIL") for r in reports)
    ev = S.load_report(Path(next(r["path"] for r in reports if r["kind"] == "eval")))
    mf = S.metrics_frame(ev["summary"], S.RETRIEVAL_METRICS)
    assert list(mf["group"][:3]) == ["overall", "positive", "negative"]
    assert "verbatim" in set(mf["group"]) and "R@1" in mf.columns
    af = S.metrics_frame(ev["summary"], S.ANSWER_METRICS)
    neg = af.set_index("group").loc["absent_fact"]
    assert np.isnan(neg["Accuracy"]) and not np.isnan(neg["Correct abstention"])
    cf = S.controls_frame(ev)
    assert len(cf) == len(ev["controls"]) and set(cf["status"]) <= {"PASS", "FAIL", "INFO", "N/A"}
    cmp_ = S.load_report(Path(next(r["path"] for r in reports if r["kind"] == "compare")))
    df = S.compare_frame({"tfidf": cmp_}, group="overall", metrics=("recall@1", "mrr"))
    assert list(dict.fromkeys(df["mode"])) == ["random", "bm25", "dense", "hybrid"]
    assert len(df) == 8 and df["value"].between(0, 1).all()
    assert S.compare_groups(cmp_)[0] == "overall"


def test_html_table_escapes_and_colours_status():
    df = pd.DataFrame([{"status": "FAIL", "check": "<b>x</b>", "value": 0.12345, "n": 3},
                       {"status": "PASS", "check": "y", "value": float("nan"), "n": 4}])
    out = S.html_table(df, status_cols=("status",))
    assert "&lt;b&gt;x&lt;/b&gt;" in out and "<b>x</b>" not in out
    assert "0.12" in out and "n/a" in out
    assert S.STATUS_COLORS["FAIL"] in out and S.STATUS_COLORS["PASS"] in out
