"""Tests for ehr_rag.embeddings. MiniLM tests are marked slow (run with ``-m slow``)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ehr_rag.config import load_settings
from ehr_rag.embeddings import (
    Embedder,
    MiniLMEmbedder,
    TfidfSvdEmbedder,
    get_embedder,
    load_embedder,
)

CORPUS = [
    "[Patient P001 | Progress Note | 2025-03-14 | Section: HPI]\nSOB and DOE x 3 days, orthopnea, HFrEF.",
    "[Patient P001 | Lab Panel 2025-03-14 | 2025-03-14]\n- Hemoglobin A1c (LOINC 4548-4): 7.2 % H",
    "[Patient P001 | Medication List | 2026-06-30]\n- metformin 1000 mg PO BID — for T2DM (active)",
    "[Patient P002 | Progress Note | 2025-01-02 | Section: Assessment]\nR knee OA, continue PT and NSAIDs.",
    "[Patient P002 | Vitals 2025-01-02 | 2025-01-02]\nBP 132/84, HR 72, SpO2 97%, BMI 31.2",
    "[Patient P003 | Allergy List | 2026-06-30]\n- penicillin: hives (moderate)",
    "[Patient P003 | Discharge Summary | 2025-05-05 | Section: Hospital Course]\nNSTEMI s/p PCI, started metoprolol.",
    "[Patient P003 | Problem List | 2026-06-30]\n- HTN (I10) since 2019-04-01 (active)",
]


@pytest.fixture()
def fitted() -> TfidfSvdEmbedder:
    emb = TfidfSvdEmbedder(dim=256, seed=7)
    emb.fit(CORPUS)
    return emb


def test_tfidf_shapes_dtype_and_norm(fitted):
    assert fitted.dim == len(CORPUS) - 1  # shrinks to n_samples - 1
    x = fitted.embed_documents(CORPUS)
    assert x.shape == (len(CORPUS), fitted.dim) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    q = fitted.embed_query("metformin dose")
    assert q.shape == (fitted.dim,) and q.dtype == np.float32
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
    assert fitted.embed_documents([]).shape == (0, fitted.dim)
    assert isinstance(fitted, Embedder)


def test_tfidf_dim_limited_by_features():
    emb = TfidfSvdEmbedder(dim=256)
    emb.fit(["alpha beta", "beta gamma", "gamma alpha", "alpha", "beta", "gamma"])
    # features: 3 unigrams + 3 bigrams = 6 -> min(256, 5, 5)
    assert emb.dim == 5
    emb = TfidfSvdEmbedder(dim=2)
    emb.fit(CORPUS)
    assert emb.dim == 2


def test_tfidf_retrieves_relevant_text(fitted):
    x = fitted.embed_documents(CORPUS)
    for query, expected in [("metformin 1000 mg", 2), ("hemoglobin a1c 7.2", 1), ("penicillin allergy hives", 5)]:
        sims = x @ fitted.embed_query(query)
        assert int(np.argmax(sims)) == expected, query


def test_tfidf_deterministic_with_seed():
    a, b = TfidfSvdEmbedder(dim=64, seed=3), TfidfSvdEmbedder(dim=64, seed=3)
    a.fit(CORPUS)
    b.fit(CORPUS)
    np.testing.assert_array_equal(a.embed_documents(CORPUS), b.embed_documents(CORPUS))


def test_tfidf_unknown_query_is_zero_vector(fitted):
    q = fitted.embed_query("zzzqqq xxyyzz")
    assert not np.any(np.isnan(q)) and np.allclose(q, 0.0)


def test_tfidf_save_load_roundtrip(fitted, tmp_path):
    fitted.save(tmp_path)
    loaded = TfidfSvdEmbedder.load(tmp_path)
    assert (loaded.dim, loaded.seed, loaded.requested_dim) == (fitted.dim, fitted.seed, fitted.requested_dim)
    np.testing.assert_array_equal(loaded.embed_documents(CORPUS), fitted.embed_documents(CORPUS))
    np.testing.assert_array_equal(loaded.embed_query("SOB"), fitted.embed_query("SOB"))


def test_tfidf_embed_before_fit_raises():
    emb = TfidfSvdEmbedder()
    with pytest.raises(RuntimeError, match="not fitted"):
        emb.embed_documents(["x"])
    with pytest.raises(RuntimeError, match="not fitted"):
        emb.embed_query("x")
    with pytest.raises(ValueError, match="at least 2"):
        emb.fit(["only one text"])


def test_get_embedder_factory():
    emb = get_embedder(load_settings(embedder="tfidf", tfidf_dim=32, seed=9))
    assert isinstance(emb, TfidfSvdEmbedder) and emb.requested_dim == 32 and emb.seed == 9
    assert not emb.is_fitted
    m = get_embedder(load_settings(embedder="minilm"))
    assert isinstance(m, MiniLMEmbedder) and m._model is None  # nothing loaded yet
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder(load_settings(embedder="word2vec"))


def test_load_embedder_tfidf(fitted, tmp_path):
    settings = load_settings(embedder="tfidf", index_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="ingest --embedder tfidf"):
        load_embedder(settings)
    fitted.save(settings.embedder_index_dir)
    loaded = load_embedder(settings)
    np.testing.assert_array_equal(loaded.embed_documents(CORPUS), fitted.embed_documents(CORPUS))


def test_minilm_save_writes_small_json_only(tmp_path):
    settings = load_settings(embedder="minilm", index_dir=tmp_path, minilm_model="some/model")
    emb = MiniLMEmbedder(model_name="some/model", batch_size=8)
    emb.save(settings.embedder_index_dir)
    files = list(settings.embedder_index_dir.iterdir())
    assert [f.name for f in files] == ["minilm.json"]
    assert json.loads(files[0].read_text())["model_name"] == "some/model"
    loaded = load_embedder(settings)
    assert isinstance(loaded, MiniLMEmbedder) and loaded.batch_size == 8 and loaded._model is None
    # no saved config -> fresh pretrained embedder from settings
    fresh = load_embedder(load_settings(embedder="minilm", index_dir=tmp_path / "empty"))
    assert isinstance(fresh, MiniLMEmbedder)


def test_tfidf_path_never_imports_torch(tmp_path):
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from ehr_rag.config import load_settings
        from ehr_rag.embeddings import get_embedder, load_embedder
        s = load_settings(embedder="tfidf", index_dir=Path(r"{tmp_path}"))
        e = get_embedder(s)
        e.fit({CORPUS!r})
        e.embed_documents({CORPUS!r}); e.embed_query("SOB")
        e.save(s.embedder_index_dir)
        load_embedder(s).embed_query("SOB")
        bad = [m for m in ("torch", "sentence_transformers", "transformers") if m in sys.modules]
        assert not bad, bad
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# --------------------------------------------------------------------------- MiniLM (slow)


@pytest.mark.slow
def test_minilm_real_model():
    emb = MiniLMEmbedder()
    assert emb.dim == 384
    texts = ["SOB, dyspnea on exertion", "knee osteoarthritis"]
    x = emb.embed_documents(texts)
    assert x.shape == (2, 384) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    q = emb.embed_query("shortness of breath")
    assert q.shape == (384,) and np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
    sim_sob, sim_knee = x @ q
    assert sim_sob > sim_knee
    assert emb.embed_documents([]).shape == (0, 384)
    assert "torch" in sys.modules  # control: the sys.modules check in the tfidf test can detect torch
