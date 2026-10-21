"""End-to-end tests for ehr_rag.pipeline on a small generated dataset (tfidf + extractive, no network)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from ehr_rag.config import Settings
from ehr_rag.generation import GenerationError, GenerationResult
from ehr_rag.ingest.phi_guard import NonSyntheticDataError
from ehr_rag.pipeline import (
    CHROMA_DIRNAME,
    CHUNKS_FILENAME,
    MANIFEST_FILENAME,
    EHRRagPipeline,
    IndexNotBuiltError,
    PHIDetectedError,
)
from ehr_rag.synth.generator import generate_dataset, load_documents
from ehr_rag.types import RETRIEVAL_MODES, Answer, Document


def _settings(root: Path) -> Settings:
    return Settings().with_overrides(
        data_dir=root / "data", index_dir=root / "index", reports_dir=root / "reports",
        embedder="tfidf", llm="extractive", tfidf_dim=64,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    settings = _settings(root)
    dataset = generate_dataset(n_patients=4, seed=7)
    dataset.write(settings.data_dir)
    pipe = EHRRagPipeline.build_index(settings)
    return settings, dataset, pipe


@pytest.fixture(scope="module")
def loaded(built):
    settings, _, _ = built
    return EHRRagPipeline.load(settings)


class StubFailingGenerator:
    name = "claude"

    def __init__(self):
        self.calls = 0

    def generate(self, question, patient_id, contexts):
        self.calls += 1
        raise GenerationError("Claude API call failed (HTTP 529): overloaded", warnings=["pre-failure warning"])


class StubOkGenerator:
    name = "stub"

    def generate(self, question, patient_id, contexts):
        return GenerationResult(text="ok [S1]", citations=[], abstained=False, mode="llm", model="stub-model",
                                warnings=["stub warning"])


# ------------------------------------------------------------------ build / load


def test_build_writes_index_files(built):
    settings, dataset, pipe = built
    out = settings.embedder_index_dir
    for name in (CHUNKS_FILENAME, MANIFEST_FILENAME, "tfidf_svd.joblib", CHROMA_DIRNAME):
        assert (out / name).exists(), name
    manifest = json.loads((out / MANIFEST_FILENAME).read_text())
    n_chunks = sum(1 for line in (out / CHUNKS_FILENAME).read_text().splitlines() if line.strip())
    assert manifest["embedder"] == "tfidf"
    assert manifest["n_chunks"] == n_chunks == pipe.store.count() == len(pipe.chunks)
    assert manifest["n_docs"] == len(load_documents(settings.data_dir))
    assert manifest["n_patients"] == 4
    assert manifest["data_seed"] == 7
    assert manifest["dim"] == pipe.embedder.dim
    assert manifest["created_at"] and manifest["model"]
    assert pipe.patient_ids == sorted(p["patient_id"] for p in dataset.patients)


def test_load_matches_build(built, loaded):
    _, _, pipe = built
    assert loaded.store.count() == pipe.store.count()
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in pipe.chunks]
    assert loaded.embedder.dim == pipe.embedder.dim
    q = "most recent creatinine"
    a = [r.chunk.chunk_id for r in pipe.retrieve(q, "P001")]
    b = [r.chunk.chunk_id for r in loaded.retrieve(q, "P001")]
    assert a == b and a


def test_build_accepts_str_paths(tmp_path, built):
    settings, _, _ = built
    s = dataclasses.replace(_settings(tmp_path), data_dir=str(settings.data_dir), index_dir=str(tmp_path / "ix"))
    docs = [d for d in load_documents(settings.data_dir) if d.patient_id == "P002"]
    pipe = EHRRagPipeline.build_index(s, docs)
    assert pipe.patient_ids == ["P002"]
    assert (tmp_path / "ix" / "tfidf" / MANIFEST_FILENAME).exists()


# ------------------------------------------------------------------ ask / isolation


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_every_retrieved_chunk_belongs_to_patient(loaded, mode):
    for pid in loaded.patient_ids:
        for q in ("current medications", "most recent hemoglobin A1c", "allergies", "why was it stopped"):
            hits = loaded.retrieve(q, pid, k=8, mode=mode)
            assert hits, (pid, q, mode)
            assert all(h.chunk.patient_id == pid for h in hits)
            assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert loaded.retriever.isolation_violations == 0


def test_ask_end_to_end(loaded, built):
    _, dataset, _ = built
    allergy = next(a for a in dataset.allergies if a["patient_id"] in loaded.patient_ids)
    pid = allergy["patient_id"]
    ans = loaded.ask("What is the patient allergic to?", pid)
    assert isinstance(ans, Answer)
    assert ans.patient_id == pid and ans.mode == "extractive" and ans.model is None
    assert ans.retrieved and all(r.chunk.patient_id == pid for r in ans.retrieved)
    assert len(ans.retrieved) <= loaded.settings.top_k
    assert ans.latency_ms > 0
    assert not ans.abstained
    assert allergy["substance"].lower() in ans.text.lower()
    retrieved_ids = {r.chunk.chunk_id for r in ans.retrieved}
    assert ans.citations and all(c.chunk_id in retrieved_ids for c in ans.citations)
    # Answer round-trips through JSON
    assert Answer.from_dict(json.loads(json.dumps(ans.to_dict()))).text == ans.text


def test_ask_respects_k_mode_and_doc_types(loaded):
    ans = loaded.ask("blood pressure", "P001", k=3, mode="bm25", doc_types=["vitals"])
    assert 0 < len(ans.retrieved) <= 3
    assert all(r.source == "bm25" and r.chunk.doc_type == "vitals" for r in ans.retrieved)


def test_unknown_patient_raises_listing_valid_ids(loaded):
    with pytest.raises(ValueError) as exc:
        loaded.ask("anything", "P999")
    assert "P999" in str(exc.value) and "P001" in str(exc.value)
    with pytest.raises(ValueError):
        loaded.retrieve("anything", "")


def test_generation_error_falls_back_to_extractive(loaded):
    stub = StubFailingGenerator()
    original = loaded.generator
    loaded.generator = stub
    try:
        ans = loaded.ask("What are the current medications?", "P001")
    finally:
        loaded.generator = original
    assert stub.calls == 1
    assert ans.mode == "extractive"
    assert "pre-failure warning" in ans.warnings
    assert any(w.startswith("Claude failed:") and "529" in w for w in ans.warnings)
    assert ans.text and ans.retrieved


def test_successful_generator_result_maps_onto_answer(built):
    settings, _, _ = built
    pipe = EHRRagPipeline.load(settings, generator=StubOkGenerator())
    ans = pipe.ask("current medications", "P002")
    assert (ans.text, ans.mode, ans.model, ans.warnings) == ("ok [S1]", "llm", "stub-model", ["stub warning"])


def test_non_generation_errors_are_not_swallowed(loaded):
    class Boom:
        name = "claude"

        def generate(self, *a):
            raise NonSyntheticDataError(["x"])

    original = loaded.generator
    loaded.generator = Boom()
    try:
        with pytest.raises(NonSyntheticDataError):
            loaded.ask("current medications", "P001")
    finally:
        loaded.generator = original


# ------------------------------------------------------------------ raw tables


def test_list_patients_and_summary(loaded, built):
    _, dataset, _ = built
    patients = loaded.list_patients()
    assert [p["patient_id"] for p in patients] == sorted(p["patient_id"] for p in dataset.patients)
    assert all(p["indexed"] and p["synthetic"] is True for p in patients)
    pid = patients[0]["patient_id"]
    s = loaded.patient_summary(pid)
    assert s["demographics"]["patient_id"] == pid
    assert s["active_problems"] and all(c["status"] == "active" for c in s["active_problems"])
    assert all(m["status"] == "active" and m["end_date"] is None for m in s["active_medications"])
    exp_allergies = [a for a in dataset.allergies if a["patient_id"] == pid]
    assert s["allergies"] == exp_allergies and s["nkda"] == (not exp_allergies)
    with pytest.raises(ValueError):
        loaded.patient_summary("P999")


# ------------------------------------------------------------------ failure modes


def test_load_without_index_raises(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(IndexNotBuiltError) as exc:
        EHRRagPipeline.load(settings)
    msg = str(exc.value)
    assert "python -m ehr_rag ingest --embedder tfidf" in msg
    assert f"EHR_RAG_INDEX_DIR={settings.index_dir}" in msg


def test_load_detects_chunk_count_mismatch(tmp_path, built):
    settings, _, _ = built
    s = _settings(tmp_path).with_overrides(data_dir=settings.data_dir)
    docs = [d for d in load_documents(settings.data_dir) if d.patient_id == "P003"]
    EHRRagPipeline.build_index(s, docs)
    chunks_path = s.embedder_index_dir / CHUNKS_FILENAME
    lines = chunks_path.read_text().splitlines()
    chunks_path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(IndexNotBuiltError, match="chunks"):
        EHRRagPipeline.load(s)


def test_load_detects_missing_embedder(tmp_path, built):
    settings, _, _ = built
    s = _settings(tmp_path).with_overrides(data_dir=settings.data_dir)
    docs = [d for d in load_documents(settings.data_dir) if d.patient_id == "P004"]
    EHRRagPipeline.build_index(s, docs)
    (s.embedder_index_dir / "tfidf_svd.joblib").unlink()
    with pytest.raises(IndexNotBuiltError, match="embedder"):
        EHRRagPipeline.load(s)


def test_build_without_documents_tells_user_to_generate(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m ehr_rag generate"):
        EHRRagPipeline.build_index(_settings(tmp_path))


def _docs_subset(settings: Settings) -> list[Document]:
    return [d for d in load_documents(settings.data_dir) if d.patient_id == "P001"]


def test_phi_refusal_lists_doc_ids(tmp_path, built):
    settings, _, _ = built
    s = _settings(tmp_path)
    docs = _docs_subset(settings)
    bad = dataclasses.replace(docs[0], text=docs[0].text + "\nCall back at 555-867-5309.")
    with pytest.raises(PHIDetectedError) as exc:
        EHRRagPipeline.build_index(s, [bad] + docs[1:])
    assert bad.doc_id in str(exc.value) and "phone" in str(exc.value)
    assert list(exc.value.findings) == [bad.doc_id]
    assert not s.embedder_index_dir.exists()  # nothing was written


def test_non_synthetic_refusal(tmp_path, built):
    settings, _, _ = built
    s = _settings(tmp_path)
    docs = _docs_subset(settings)
    bad = dataclasses.replace(docs[1], metadata={**docs[1].metadata, "synthetic": False})
    with pytest.raises(NonSyntheticDataError) as exc:
        EHRRagPipeline.build_index(s, docs[:1] + [bad] + docs[2:])
    assert exc.value.offending_ids == [bad.doc_id]
    assert not s.embedder_index_dir.exists()


def test_rebuild_replaces_previous_index(tmp_path, built):
    settings, _, _ = built
    s = _settings(tmp_path).with_overrides(data_dir=settings.data_dir)
    all_docs = load_documents(settings.data_dir)
    EHRRagPipeline.build_index(s, all_docs)
    small = EHRRagPipeline.build_index(s, [d for d in all_docs if d.patient_id == "P001"])
    reloaded = EHRRagPipeline.load(s)
    assert reloaded.store.count() == small.store.count() == len(small.chunks)
    assert reloaded.patient_ids == ["P001"]
