"""Tests for ehr_rag.vectorstore (real Chroma in tmp_path, real TfidfSvdEmbedder, hand-made synthetic chunks)."""
from __future__ import annotations

import numpy as np
import pytest

from ehr_rag.embeddings import TfidfSvdEmbedder
from ehr_rag.types import Chunk, chunk_id_for
from ehr_rag.vectorstore import ChromaStore, build_where

# (patient, doc_type, date, title, section, body)
_ROWS = [
    ("P001", "medication_list", "2026-06-01", "Medication List", None,
     "Medication list as of 2026-06-01.\n- metformin 1000 mg PO BID — started 2024-05-02 — for T2DM (active)"),
    ("P001", "lab_panel", "2025-03-14", "Lab Panel 2025-03-14", None,
     "Lab results drawn 2025-03-14.\n- Hemoglobin A1c (HbA1c): 7.2 % (ref 4.0-5.6) [H]"),
    ("P001", "clinical_note", "2025-03-14", "Primary Care Progress Note", "ASSESSMENT AND PLAN",
     "# T2DM: A1c 7.2, above goal <7.0. Continue metformin 1000 mg PO BID."),
    ("P001", "allergy_list", "2026-06-01", "Allergy List", None, "Allergy list as of 2026-06-01.\n- penicillin — hives (moderate)"),
    ("P002", "medication_list", "2026-05-11", "Medication List", None,
     "Medication list as of 2026-05-11.\n- losartan 50 mg PO daily — started 2023-02-01 — for HTN (active)\n"
     "- lisinopril 20 mg PO daily — started 2021-01-10 — stopped 2023-02-01 — for HTN "
     "(discontinued: persistent dry cough (likely ACE inhibitor-related); switched to losartan)"),
    ("P002", "clinical_note", "2023-02-01", "Primary Care Progress Note", "HISTORY OF PRESENT ILLNESS",
     "Persistent dry cough on lisinopril; switched to losartan 50 mg PO daily."),
    ("P002", "vitals", "2025-01-02", "Vital Signs 2025-01-02", None,
     "Vital signs recorded 2025-01-02.\n- Blood pressure: 132/84 mmHg\n- Heart rate: 72 bpm"),
    ("P003", "clinical_note", "2025-05-05", "Discharge Summary", "HOSPITAL COURSE",
     "NSTEMI s/p PCI, started metoprolol succinate 25 mg PO daily and atorvastatin 80 mg."),
    ("P003", "problem_list", "2026-06-20", "Problem List", None,
     "Problem list as of 2026-06-20.\n- Essential (primary) hypertension (HTN) — ICD-10 I10 — onset 2019-04-01 — active"),
    ("P003", "lab_panel", "2026-06-20", "Lab Panel 2026-06-20", None,
     "Lab results drawn 2026-06-20.\n- eGFR (CKD-EPI 2021): 52 mL/min/1.73m2 (ref >=60) [L]"),
]


def make_chunks() -> list[Chunk]:
    chunks = []
    counters: dict[str, int] = {}
    for pid, doc_type, date, title, section, body in _ROWS:
        doc_id = f"{pid}-{doc_type}-{date}"
        i = counters.get(doc_id, 0)
        counters[doc_id] = i + 1
        sec = f" | Section: {section.title()}" if section else ""
        meta = {"synthetic": True, "patient_id": pid, "doc_type": doc_type, "source": "test",
                "mentions": {"conditions": ["t2dm"], "labs": [["hba1c", date]]}}
        if doc_type == "clinical_note":
            meta["note_type"] = "progress_note"
        chunks.append(Chunk(
            chunk_id=chunk_id_for(doc_id, i), doc_id=doc_id, patient_id=pid, doc_type=doc_type, date=date,
            title=title, section=section, text=f"[Patient {pid} | {title} | {date}{sec}]\n{body}", metadata=meta,
        ))
    return chunks


@pytest.fixture()
def chunks() -> list[Chunk]:
    return make_chunks()


@pytest.fixture()
def embedder(chunks) -> TfidfSvdEmbedder:
    emb = TfidfSvdEmbedder(dim=32, seed=3)
    emb.fit([c.text for c in chunks])
    return emb


@pytest.fixture()
def store(tmp_path, chunks, embedder) -> ChromaStore:
    s = ChromaStore(tmp_path / "chroma", "ehr_chunks_test")
    s.add(chunks, embedder.embed_documents([c.text for c in chunks]), batch_size=3)  # several batches
    return s


def test_collection_uses_cosine_space(store, chunks):
    assert store.space == "cosine"
    assert store.count() == len(chunks)


def test_roundtrip_chunks_equal_written(store, chunks):
    got = store.get_chunks()
    assert got == sorted(chunks, key=lambda c: c.chunk_id)
    assert any(c.section is None for c in got) and any(c.section == "HOSPITAL COURSE" for c in got)
    assert all(isinstance(c.metadata["mentions"], dict) for c in got)  # nested metadata survives via JSON
    assert store.get_chunks("P002") == sorted([c for c in chunks if c.patient_id == "P002"], key=lambda c: c.chunk_id)


def test_query_similarity_equals_dot_product(store, chunks, embedder):
    q = embedder.embed_query("metformin A1c T2DM")
    vecs = {c.chunk_id: v for c, v in zip(chunks, embedder.embed_documents([c.text for c in chunks]))}
    hits = store.query(q, k=10, patient_id="P001")
    assert hits
    for chunk, sim in hits:
        assert sim == pytest.approx(float(np.dot(vecs[chunk.chunk_id], q)), abs=1e-4)
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)
    # the returned Chunk is the one written
    written = {c.chunk_id: c for c in chunks}
    assert all(chunk == written[chunk.chunk_id] for chunk, _ in hits)


def test_query_filters_patient_and_doc_types(store, embedder):
    q = embedder.embed_query("losartan lisinopril cough")
    for pid, n in (("P001", 4), ("P002", 3), ("P003", 3)):
        hits = store.query(q, k=100, patient_id=pid)  # k far above the patient's chunk count
        assert len(hits) == n
        assert {c.patient_id for c, _ in hits} == {pid}
    hits = store.query(q, k=10, patient_id="P002", doc_types=["clinical_note", "vitals"])
    assert {c.doc_type for c, _ in hits} == {"clinical_note", "vitals"}
    assert store.query(q, k=10, patient_id="P999") == []
    assert store.query(q, k=0, patient_id="P001") == []


@pytest.mark.parametrize("pid", ["", None])
def test_query_requires_patient_id(store, embedder, pid):
    with pytest.raises(ValueError):
        store.query(embedder.embed_query("metformin"), k=3, patient_id=pid)


def test_build_where():
    assert build_where("P001") == {"patient_id": "P001"}
    assert build_where("P001", ["vitals"]) == {"$and": [{"patient_id": "P001"}, {"doc_type": {"$in": ["vitals"]}}]}
    assert build_where("P001", []) == {"patient_id": "P001"}


def test_get_embeddings_aligned(store, chunks, embedder):
    got_chunks, arr = store.get_embeddings("P003")
    assert [c.chunk_id for c in got_chunks] == sorted(c.chunk_id for c in chunks if c.patient_id == "P003")
    assert arr.shape == (3, embedder.dim) and arr.dtype == np.float32
    expected = embedder.embed_documents([c.text for c in got_chunks])
    np.testing.assert_allclose(arr, expected, atol=1e-6)
    all_chunks, all_arr = store.get_embeddings()
    assert len(all_chunks) == len(chunks) and all_arr.shape[0] == len(chunks)


def test_reset_and_persistence(tmp_path, store, chunks, embedder):
    reopened = ChromaStore(tmp_path / "chroma", "ehr_chunks_test")
    assert reopened.count() == len(chunks)
    reopened.reset()
    assert reopened.count() == 0 and reopened.space == "cosine"
    assert reopened.query(embedder.embed_query("metformin"), k=5, patient_id="P001") == []
    # re-add after reset, re-adding an id replaces instead of duplicating
    vecs = embedder.embed_documents([c.text for c in chunks])
    reopened.add(chunks, vecs)
    reopened.add(chunks[:2], vecs[:2])
    assert reopened.count() == len(chunks)


def test_add_validates_inputs(store, chunks, embedder):
    vecs = embedder.embed_documents([c.text for c in chunks])
    with pytest.raises(ValueError):
        store.add(chunks, vecs[:-1])
    with pytest.raises(ValueError):
        store.add([chunks[0], chunks[0]], vecs[:2])
    store.add([], np.zeros((0, embedder.dim), dtype=np.float32))  # no-op
