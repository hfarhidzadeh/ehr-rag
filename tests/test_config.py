"""Tests for ehr_rag.config."""
from __future__ import annotations

from pathlib import Path

from ehr_rag.config import Settings, load_settings


def test_str_path_overrides_become_paths(tmp_path):
    s = load_settings(index_dir=str(tmp_path / "ix"), data_dir=str(tmp_path / "d"), embedder="tfidf")
    assert isinstance(s.index_dir, Path) and isinstance(s.data_dir, Path)
    assert s.embedder_index_dir == tmp_path / "ix" / "tfidf"
    assert Settings().with_overrides(reports_dir="~/r").reports_dir == Path("~/r").expanduser()


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("EHR_RAG_TOP_K", "9")
    monkeypatch.setenv("EHR_RAG_CLAUDE_FALLBACKS", "false")
    monkeypatch.setenv("EHR_RAG_INDEX_DIR", str(tmp_path))
    s = load_settings(top_k=None)
    assert s.top_k == 9 and s.claude_fallbacks is False and s.index_dir == tmp_path
    assert load_settings(top_k=3).top_k == 3
