"""Tests for the ``python -m ehr_rag`` CLI on temporary directories (tfidf + extractive, no network)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ehr_rag import cli
from ehr_rag.config import PROJECT_ROOT


@pytest.fixture(scope="module")
def env_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    return {"data": root / "data", "index": root / "index", "reports": root / "reports"}


@pytest.fixture()
def cli_env(env_dirs, monkeypatch):
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(env_dirs["data"]))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(env_dirs["index"]))
    monkeypatch.setenv("EHR_RAG_REPORTS_DIR", str(env_dirs["reports"]))
    monkeypatch.setenv("EHR_RAG_LLM", "extractive")
    monkeypatch.delenv("EHR_RAG_EMBEDDER", raising=False)
    return env_dirs


@pytest.fixture(scope="module")
def ingested(env_dirs):
    mp = pytest.MonkeyPatch()
    mp.setenv("EHR_RAG_DATA_DIR", str(env_dirs["data"]))
    mp.setenv("EHR_RAG_INDEX_DIR", str(env_dirs["index"]))
    try:
        assert cli.main(["generate", "--n-patients", "4", "--seed", "7"]) == 0
        assert cli.main(["ingest", "--embedder", "tfidf"]) == 0
    finally:
        mp.undo()
    return env_dirs


def test_ask_before_ingest_explains_fix(cli_env, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "empty_index"))
    assert cli.main(["ask", "current meds", "--patient", "P001", "--embedder", "tfidf"]) == 1
    err = capsys.readouterr().err
    assert "python -m ehr_rag ingest --embedder tfidf" in err


def test_generate_and_ingest_outputs(ingested):
    data, index = ingested["data"], ingested["index"]
    assert (data / "documents.jsonl").exists() and (data / "manifest.json").exists()
    manifest = json.loads((index / "tfidf" / "index_manifest.json").read_text())
    assert manifest["n_patients"] == 4 and manifest["data_seed"] == 7


def test_ask_prints_answer_citations_and_status(ingested, cli_env, capsys):
    rc = cli.main(["ask", "What is the most recent creatinine?", "--patient", "p001", "--embedder", "tfidf",
                   "--mode", "hybrid", "--k", "4", "--show-retrieved"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Patient P001" in out
    assert "Abstained:" in out and "Generator: extractive" in out
    assert "Citations:" in out and "Retrieved:" in out
    assert "P001-" in out and "P002-" not in out


def test_ask_json(ingested, cli_env, capsys):
    rc = cli.main(["ask", "allergies", "--patient", "P002", "--embedder", "tfidf", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["patient_id"] == "P002"
    assert all(r["chunk"]["patient_id"] == "P002" for r in data["retrieved"])


def test_ask_unknown_patient(ingested, cli_env, capsys):
    assert cli.main(["ask", "allergies", "--patient", "P999", "--embedder", "tfidf"]) == 2
    err = capsys.readouterr().err
    assert "P999" in err and "P001" in err


def test_ingest_without_data(cli_env, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(tmp_path / "nothing"))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "ix"))
    assert cli.main(["ingest", "--embedder", "tfidf"]) == 1
    assert "python -m ehr_rag generate" in capsys.readouterr().err


def test_ingest_refuses_phi(ingested, cli_env, capsys, tmp_path, monkeypatch):
    src = ingested["data"] / "documents.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    rows[0]["text"] += "\nPhone 555-867-5309."
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "documents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("EHR_RAG_DATA_DIR", str(bad_dir))
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path / "ix"))
    assert cli.main(["ingest", "--embedder", "tfidf"]) == 3
    err = capsys.readouterr().err
    assert rows[0]["doc_id"] in err and "PHI" in err
    assert not (tmp_path / "ix" / "tfidf" / "index_manifest.json").exists()


def test_eval_without_runner_prints_message(cli_env, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "ehr_rag.evaluation.runner", None)  # makes the import fail
    assert cli.main(["eval", "--embedder", "tfidf"]) == 2
    assert "evaluation module is not available" in capsys.readouterr().err


def test_app_runs_streamlit_from_project_root(cli_env, monkeypatch, tmp_path):
    calls = {}

    class Done:
        returncode = 0

    def fake_run(cmd, cwd=None, env=None, **kw):
        calls.update(cmd=cmd, cwd=cwd, env=env)
        return Done()

    script = PROJECT_ROOT / "app" / "streamlit_app.py"
    if not script.exists():
        script = tmp_path / "streamlit_app.py"
        script.write_text("# placeholder\n")
        monkeypatch.setattr(cli, "APP_SCRIPT", script)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["app", "--port", "8599"]) == 0
    assert calls["cmd"][1:4] == ["-m", "streamlit", "run"]
    assert calls["cmd"][4].endswith("streamlit_app.py")
    assert calls["cmd"][-2:] == ["--server.port", "8599"]
    assert calls["cwd"] == str(PROJECT_ROOT)


def test_app_missing_script(cli_env, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "APP_SCRIPT", tmp_path / "missing.py")
    assert cli.main(["app"]) == 1
    assert "not found" in capsys.readouterr().err


def test_module_entry_point_help():
    proc = subprocess.run([sys.executable, "-m", "ehr_rag", "--help"], capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT), timeout=60)
    assert proc.returncode == 0
    for command in ("generate", "ingest", "ask", "eval", "app"):
        assert command in proc.stdout


def test_ask_requires_patient(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["ask", "anything"])
    assert exc.value.code == 2
