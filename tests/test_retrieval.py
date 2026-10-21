"""Tests for ehr_rag.retrieval: tokenizer, RRF, BM25, Retriever modes and patient isolation."""
from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from ehr_rag.config import PROJECT_ROOT
from ehr_rag.embeddings import TfidfSvdEmbedder
from ehr_rag.retrieval import BM25Index, PatientIsolationError, Retriever, reciprocal_rank_fusion, tokenize
from ehr_rag.types import RETRIEVAL_MODES, Chunk
from ehr_rag.vectorstore import ChromaStore

from tests.test_vectorstore import make_chunks

DATA_DIR: Path = PROJECT_ROOT / "data" / "synthetic"


@pytest.fixture()
def chunks() -> list[Chunk]:
    return make_chunks()


@pytest.fixture()
def retriever(tmp_path, chunks) -> Retriever:
    emb = TfidfSvdEmbedder(dim=32, seed=3)
    emb.fit([c.text for c in chunks])
    store = ChromaStore(tmp_path / "chroma", "ehr_chunks_test")
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return Retriever(store, emb, BM25Index(chunks), rrf_k=60, candidate_pool=5)


# ------------------------------------------------------------------ tokenize / RRF


def test_tokenize_keeps_clinical_tokens():
    toks = tokenize("What is the most recent HbA1c? A1c 7.2 on metformin 1000 mg; eGFR 52, losartan.")
    for t in ("hba1c", "a1c", "7.2", "1000", "mg", "egfr", "52", "losartan", "metformin", "recent"):
        assert t in toks
    assert "the" not in toks and "what" not in toks and "is" not in toks
    assert tokenize("Cr 0.84 on 2025-07-10.") == ["cr", "0.84", "2025-07-10"]
    assert "no" in tokenize("No known drug allergies")  # negation kept
    assert tokenize("the of and") == []


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == ["a", "b"]  # equal scores -> tie broken by id
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["d"] == pytest.approx(1 / 63)
    assert reciprocal_rank_fusion([]) == []


# ------------------------------------------------------------------ BM25


def test_bm25_per_patient(chunks):
    idx = BM25Index(chunks)
    hits = idx.search("why was lisinopril stopped", "P002", k=5)
    assert hits and hits[0][0].patient_id == "P002"
    assert all(c.patient_id == "P002" for c, _ in hits)
    assert "lisinopril" in hits[0][0].text
    assert idx.search("lisinopril", "P001", k=5) == []          # token only in another patient's chart
    assert idx.search("zzqx unknownword", "P002", k=5) == []    # no known tokens
    assert idx.search("the of", "P002", k=5) == []              # only stopwords
    assert idx.search("lisinopril", "P999", k=5) == []          # unknown patient
    assert idx.search("losartan", "P002", k=5, doc_types=["vitals"]) == []
    with pytest.raises(ValueError):
        idx.search("losartan", "", k=5)


# ------------------------------------------------------------------ Retriever


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_modes_return_ranked_patient_chunks(retriever, mode):
    res = retriever.retrieve("metformin A1c 7.2", "P001", k=3, mode=mode)
    assert 1 <= len(res) <= 3
    assert [r.rank for r in res] == list(range(1, len(res) + 1))
    assert all(r.chunk.patient_id == "P001" and r.source == mode for r in res)
    scores = [r.score for r in res]
    if mode != "random":
        assert scores == sorted(scores, reverse=True)
        assert any("metformin" in r.chunk.text for r in res[:2])


def test_k_larger_than_patient_chunks(retriever):
    for mode in RETRIEVAL_MODES:
        res = retriever.retrieve("hypertension HTN metoprolol eGFR", "P003", k=50, mode=mode)
        assert {r.chunk.patient_id for r in res} == {"P003"}
        assert len({r.chunk.chunk_id for r in res}) == len(res)
        assert len(res) <= 3
    assert len(retriever.retrieve("anything at all", "P003", k=50, mode="random")) == 3
    assert len(retriever.retrieve("metoprolol", "P003", k=50, mode="dense")) == 3


def test_hybrid_score_is_rrf(retriever):
    q = "lisinopril cough losartan"
    dense = [r.chunk.chunk_id for r in retriever.retrieve(q, "P002", k=5, mode="dense")]
    bm25 = [r.chunk.chunk_id for r in retriever.retrieve(q, "P002", k=5, mode="bm25")]
    expected = reciprocal_rank_fusion([dense, bm25], k=60)
    res = retriever.retrieve(q, "P002", k=5, mode="hybrid")
    assert [(r.chunk.chunk_id, pytest.approx(r.score)) for r in res] == [(i, pytest.approx(s)) for i, s in expected]


def test_random_is_deterministic_and_query_seeded(retriever):
    a = [r.chunk.chunk_id for r in retriever.retrieve("q1", "P001", k=2, mode="random")]
    b = [r.chunk.chunk_id for r in retriever.retrieve("q1", "P001", k=2, mode="random")]
    assert a == b
    seen = {tuple(r.chunk.chunk_id for r in retriever.retrieve(f"query {i}", "P001", k=4, mode="random"))
            for i in range(12)}
    assert len(seen) > 1  # different queries give different orders


def test_doc_types_filter(retriever):
    for mode in RETRIEVAL_MODES:
        res = retriever.retrieve("metformin A1c", "P001", k=5, mode=mode, doc_types=["lab_panel"])
        assert res and {r.chunk.doc_type for r in res} == {"lab_panel"}


def test_invalid_arguments(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("metformin", "", mode="dense")
    with pytest.raises(ValueError):
        retriever.retrieve("metformin", "P001", mode="nonsense")
    assert retriever.retrieve("metformin", "P001", k=0) == []
    assert retriever.retrieve("metformin", "P999", mode="hybrid") == []


class _LeakyStore:
    """Stub store that (wrongly) returns another patient's chunk first."""

    def __init__(self, real: ChromaStore, intruder: Chunk):
        self.real, self.intruder = real, intruder

    def query(self, embedding, k, patient_id, doc_types=None):
        return [(self.intruder, 0.99)] + self.real.query(embedding, k, patient_id, doc_types)

    def get_chunks(self, patient_id=None):
        return [self.intruder] + self.real.get_chunks(patient_id)


def test_wrong_patient_chunk_dropped_or_raised(retriever, chunks):
    intruder = next(c for c in chunks if c.patient_id == "P003")
    leaky = Retriever(_LeakyStore(retriever.store, intruder), retriever.embedder, retriever.bm25, candidate_pool=5)
    for mode in ("dense", "hybrid"):
        res = leaky.retrieve("metformin A1c", "P001", k=3, mode=mode)
        assert res and all(r.chunk.patient_id == "P001" for r in res)
        assert [r.rank for r in res] == list(range(1, len(res) + 1))
    assert leaky.isolation_violations == 2
    strict = Retriever(_LeakyStore(retriever.store, intruder), retriever.embedder, retriever.bm25, strict=True)
    with pytest.raises(PatientIsolationError):
        strict.retrieve("metformin A1c", "P001", k=3, mode="dense")
    # a mislabelled chunk inside the BM25 index (patient field forged after indexing) is also caught
    forged = BM25Index(chunks)
    forged._chunks["P001"][0] = dataclasses.replace(forged._chunks["P001"][0], patient_id="P002")
    r2 = Retriever(retriever.store, retriever.embedder, forged)
    for mode in ("bm25", "random"):
        res = r2.retrieve("metformin A1c penicillin T2DM", "P001", k=10, mode=mode)
        assert all(r.chunk.patient_id == "P001" for r in res)
    assert r2.isolation_violations == 2


# ------------------------------------------------------------------ generated synthetic data


@pytest.mark.skipif(not (DATA_DIR / "documents.jsonl").exists(), reason="run `python -m ehr_rag generate` first")
def test_patient_isolation_on_generated_data(tmp_path):
    from ehr_rag.ingest.chunking import chunk_documents
    from ehr_rag.synth.generator import load_documents

    chunks = chunk_documents(load_documents(DATA_DIR))
    assert all(c.metadata.get("synthetic") is True for c in chunks)
    emb = TfidfSvdEmbedder(dim=64, seed=42)
    emb.fit([c.text for c in chunks])
    store = ChromaStore(tmp_path / "chroma", "ehr_chunks_tfidf")
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    assert store.count() == len(chunks)
    retriever = Retriever(store, emb, BM25Index(chunks))
    per_patient = Counter(c.patient_id for c in chunks)
    queries = ["What was the most recent HbA1c?", "Why was lisinopril stopped?", "drug allergies reaction",
               "blood pressure today"]
    for pid in sorted(per_patient):
        for q in queries:
            for mode in RETRIEVAL_MODES:
                res = retriever.retrieve(q, pid, k=8, mode=mode)
                if mode != "bm25":  # bm25 may legitimately find nothing (e.g. no A1c in a non-diabetic chart)
                    assert len(res) == 8, (pid, q, mode)
                assert all(r.chunk.patient_id == pid for r in res), (pid, q, mode)
        # k beyond the patient's chunk count returns exactly that patient's chunks
        full = retriever.retrieve("medication list", pid, k=10_000, mode="dense")
        assert len(full) == per_patient[pid]
    assert retriever.isolation_violations == 0
    # the A1c question finds a lab panel or note quoting A1c for a diabetic patient
    t2dm = next(c.patient_id for c in chunks if "HbA1c" in c.text)
    top = retriever.retrieve("most recent HbA1c", t2dm, k=5, mode="hybrid")
    assert any("A1c" in r.chunk.text for r in top)
