"""End-to-end EHR-RAG pipeline: build/load the index, retrieve for one patient, generate a cited answer.

Index layout (all under ``settings.embedder_index_dir``, i.e. ``<index_dir>/<embedder>/``)::

    chunks.jsonl          one Chunk.to_dict() per line (source of the BM25 index and the random baseline)
    tfidf_svd.joblib      fitted tfidf embedder   (or minilm.json for MiniLM)
    chroma/               Chroma persistent store, collection ``settings.collection_name``
    index_manifest.json   written last; its presence marks a complete build

Safety gates during :meth:`EHRRagPipeline.build_index`: every document must carry ``synthetic=True``
(:func:`~ehr_rag.ingest.phi_guard.assert_synthetic`) and the PHI tripwire
(:func:`~ehr_rag.ingest.phi_guard.check_documents`) must find nothing, otherwise nothing is indexed.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ehr_rag.config import Settings
from ehr_rag.embeddings import Embedder, get_embedder, load_embedder
from ehr_rag.generation import ClaudeGenerator, ExtractiveGenerator, GenerationError, Generator, get_generator
from ehr_rag.ingest.chunking import chunk_documents
from ehr_rag.ingest.phi_guard import PHIFinding, assert_synthetic, check_documents
from ehr_rag.retrieval import BM25Index, Retriever
from ehr_rag.types import Answer, Chunk, Document, RetrievedChunk
from ehr_rag.vectorstore import ChromaStore

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "index_manifest.json"
CHROMA_DIRNAME = "chroma"
INDEX_FORMAT_VERSION = 1
_EMBED_BATCH = 256
_MAX_IDS_IN_MESSAGE = 20


class IndexNotBuiltError(RuntimeError):
    """The index for the configured embedder is missing, incomplete or inconsistent."""


class PHIDetectedError(ValueError):
    """Raised by :meth:`EHRRagPipeline.build_index` when the PHI tripwire flags any document.

    ``findings`` maps doc_id -> list of :class:`~ehr_rag.ingest.phi_guard.PHIFinding`.
    """

    def __init__(self, findings: dict[str, list[PHIFinding]]):
        self.findings = dict(findings)
        ids = sorted(self.findings)
        parts = [f"{d} ({', '.join(sorted({f.kind for f in self.findings[d]}))})" for d in ids[:_MAX_IDS_IN_MESSAGE]]
        more = len(ids) - _MAX_IDS_IN_MESSAGE
        suffix = f" (+{more} more)" if more > 0 else ""
        super().__init__(
            f"Refusing to index: PHI-like patterns found in {len(ids)} document(s): {'; '.join(parts)}{suffix}. "
            "EHR-RAG only indexes synthetic data from ehr_rag.synth."
        )


# --------------------------------------------------------------------------- helpers


def _coerce_paths(settings: Settings) -> Settings:
    """Turn str path overrides into Paths (``load_settings(index_dir="x")`` leaves them as str)."""
    return settings.with_overrides(
        data_dir=Path(settings.data_dir).expanduser(),
        index_dir=Path(settings.index_dir).expanduser(),
        reports_dir=Path(settings.reports_dir).expanduser(),
    )


def _rebuild_command(settings: Settings) -> str:
    cmd = f"python -m ehr_rag ingest --embedder {settings.embedder}"
    default = Settings()
    env = []
    if Path(settings.data_dir).resolve() != default.data_dir.resolve():
        env.append(f"EHR_RAG_DATA_DIR={settings.data_dir}")
    if Path(settings.index_dir).resolve() != default.index_dir.resolve():
        env.append(f"EHR_RAG_INDEX_DIR={settings.index_dir}")
    return " ".join(env + [cmd])


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _data_seed(data_dir: Path) -> Optional[int]:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        seed = json.loads(path.read_text(encoding="utf-8")).get("seed")
        return int(seed) if seed is not None else None
    except (ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Could not read seed from %s", path)
        return None


def _embed_in_batches(embedder: Embedder, texts: list[str], batch: int = _EMBED_BATCH) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        parts.append(np.asarray(embedder.embed_documents(texts[start : start + batch]), dtype=np.float32))
        logger.info("Embedded %d/%d chunks", min(start + batch, len(texts)), len(texts))
    if not parts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.vstack(parts)


def _generator_label(generator: Any) -> str:
    if isinstance(generator, ClaudeGenerator) or getattr(generator, "name", "") == "claude":
        return "Claude"
    return f"Generator {getattr(generator, 'name', type(generator).__name__)!r}"


# --------------------------------------------------------------------------- pipeline


class EHRRagPipeline:
    """Patient-scoped retrieval + grounded generation over an index built from synthetic documents."""

    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        embedder: Embedder,
        bm25: BM25Index,
        chunks: list[Chunk],
        generator: Optional[Generator] = None,
        manifest: Optional[dict[str, Any]] = None,
    ):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.chunks = chunks
        self.manifest = dict(manifest or {})
        self.retriever = Retriever(
            store, embedder, bm25, rrf_k=settings.rrf_k, candidate_pool=settings.candidate_pool
        )
        self.generator: Generator = generator if generator is not None else get_generator(settings)
        self._fallback: Optional[ExtractiveGenerator] = None
        self._tables: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ build / load

    @classmethod
    def build_index(
        cls,
        settings: Settings,
        documents: list[Document] | None = None,
        *,
        generator: Optional[Generator] = None,
    ) -> "EHRRagPipeline":
        """Validate, chunk, embed and persist ``documents`` (default: ``settings.data_dir/documents.jsonl``)."""
        settings = _coerce_paths(settings)
        if documents is None:
            docs_path = settings.data_dir / "documents.jsonl"
            if not docs_path.exists():
                raise FileNotFoundError(
                    f"No documents at {docs_path}. Generate the synthetic dataset first: "
                    f"`python -m ehr_rag generate` (or set EHR_RAG_DATA_DIR to a generated dataset)."
                )
            from ehr_rag.synth.generator import load_documents

            documents = load_documents(settings.data_dir)
        documents = list(documents)
        if not documents:
            raise ValueError("No documents to index")

        # Safety gates: nothing is written unless every document is synthetic and PHI-clean.
        assert_synthetic(documents)
        findings = check_documents(documents)
        if findings:
            raise PHIDetectedError(findings)

        t0 = time.perf_counter()
        chunks = chunk_documents(
            documents, max_chars=settings.chunk_max_chars, overlap_sentences=settings.chunk_overlap_sentences
        )
        assert_synthetic(chunks)
        if not chunks:
            raise ValueError("Chunking produced no chunks")
        logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))

        texts = [c.text for c in chunks]
        embedder = get_embedder(settings)
        logger.info("Fitting %s embedder on %d chunks", embedder.name, len(texts))
        embedder.fit(texts)
        vectors = _embed_in_batches(embedder, texts)

        out_dir = settings.embedder_index_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / MANIFEST_FILENAME
        if manifest_path.exists():  # an interrupted rebuild must not look complete
            manifest_path.unlink()

        store = ChromaStore(out_dir / CHROMA_DIRNAME, settings.collection_name)
        store.reset()
        store.add(chunks, vectors)
        _write_jsonl(out_dir / CHUNKS_FILENAME, [c.to_dict() for c in chunks])
        embedder.save(out_dir)

        patient_ids = sorted({c.patient_id for c in chunks})
        manifest = {
            "synthetic": True,
            "format_version": INDEX_FORMAT_VERSION,
            "embedder": embedder.name,
            "dim": int(vectors.shape[1]),
            "model": getattr(embedder, "model_name", None) or f"tfidf-svd(dim={embedder.dim})",
            "collection": settings.collection_name,
            "n_docs": len(documents),
            "n_chunks": len(chunks),
            "n_patients": len(patient_ids),
            "patient_ids": patient_ids,
            "data_dir": str(settings.data_dir),
            "data_seed": _data_seed(settings.data_dir),
            "chunk_max_chars": settings.chunk_max_chars,
            "chunk_overlap_sentences": settings.chunk_overlap_sentences,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Built %s index: %d docs, %d chunks, %d patients, dim=%d in %.1fs -> %s",
            embedder.name, len(documents), len(chunks), len(patient_ids), manifest["dim"],
            time.perf_counter() - t0, out_dir,
        )
        return cls(settings, store, embedder, BM25Index(chunks), chunks, generator=generator, manifest=manifest)

    @classmethod
    def load(cls, settings: Settings, *, generator: Optional[Generator] = None) -> "EHRRagPipeline":
        """Open a previously built index; raise :class:`IndexNotBuiltError` with the fix if it is unusable."""
        settings = _coerce_paths(settings)
        out_dir = settings.embedder_index_dir
        fix = f"Build it with: {_rebuild_command(settings)}"

        def fail(problem: str) -> IndexNotBuiltError:
            return IndexNotBuiltError(f"{settings.embedder} index at {out_dir} is not usable: {problem}. {fix}")

        manifest_path = out_dir / MANIFEST_FILENAME
        chunks_path = out_dir / CHUNKS_FILENAME
        chroma_dir = out_dir / CHROMA_DIRNAME
        for path, what in ((manifest_path, "index manifest"), (chunks_path, "chunk file"),
                           (chroma_dir, "Chroma store")):
            if not path.exists():
                raise fail(f"missing {what} {path.name}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise fail(f"unreadable {MANIFEST_FILENAME} ({exc})") from exc
        if manifest.get("embedder") != settings.embedder:
            raise fail(f"manifest says embedder={manifest.get('embedder')!r}")

        chunks = [Chunk.from_dict(row) for row in _read_jsonl(chunks_path)]
        if len(chunks) != manifest.get("n_chunks"):
            raise fail(f"{CHUNKS_FILENAME} has {len(chunks)} chunks but the manifest says {manifest.get('n_chunks')}")
        try:
            embedder = load_embedder(settings)
        except FileNotFoundError as exc:
            raise fail(f"missing embedder files ({exc})") from exc
        if int(manifest.get("dim", -1)) != int(embedder.dim):
            raise fail(f"embedder dim {embedder.dim} != manifest dim {manifest.get('dim')}")

        store = ChromaStore(chroma_dir, settings.collection_name)
        n_store = store.count()
        if n_store != len(chunks):
            raise fail(f"Chroma collection {settings.collection_name} has {n_store} vectors but "
                       f"{CHUNKS_FILENAME} has {len(chunks)} chunks")
        logger.info("Loaded %s index: %d chunks, %d patients from %s", settings.embedder, len(chunks),
                    manifest.get("n_patients", 0), out_dir)
        return cls(settings, store, embedder, BM25Index(chunks), chunks, generator=generator, manifest=manifest)

    # ------------------------------------------------------------------ query

    @property
    def patient_ids(self) -> list[str]:
        """Patients present in the index."""
        return self.bm25.patient_ids

    def _require_patient(self, patient_id: str) -> None:
        valid = self.patient_ids
        if not patient_id or patient_id not in valid:
            shown = ", ".join(valid[:50]) + (" ..." if len(valid) > 50 else "")
            raise ValueError(f"Unknown patient_id {patient_id!r}. Valid ids: {shown}")

    def retrieve(
        self,
        question: str,
        patient_id: str,
        k: int | None = None,
        mode: str | None = None,
        doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k chunks of ``patient_id`` (defaults from settings); never returns another patient's chunk."""
        self._require_patient(patient_id)
        hits = self.retriever.retrieve(
            question,
            patient_id,
            k=self.settings.top_k if k is None else int(k),
            mode=mode or self.settings.retrieval_mode,
            doc_types=doc_types,
        )
        leaked = [h.chunk.chunk_id for h in hits if h.chunk.patient_id != patient_id]
        if leaked:  # Retriever already guards; this is a belt-and-braces hard stop
            raise RuntimeError(f"Patient isolation violated for {patient_id}: {leaked}")
        return hits

    def _fallback_generator(self) -> ExtractiveGenerator:
        if self._fallback is None:
            self._fallback = ExtractiveGenerator()
        return self._fallback

    def ask(
        self,
        question: str,
        patient_id: str,
        k: int | None = None,
        mode: str | None = None,
        doc_types: list[str] | None = None,
    ) -> Answer:
        """Retrieve for one patient and answer with citations; falls back to extractive if the LLM fails."""
        t0 = time.perf_counter()
        contexts = self.retrieve(question, patient_id, k=k, mode=mode, doc_types=doc_types)
        warnings: list[str] = []
        try:
            result = self.generator.generate(question, patient_id, contexts)
        except GenerationError as err:
            label = _generator_label(self.generator)
            logger.warning("%s failed (%s); falling back to extractive generator", label, err)
            warnings.extend(err.warnings)
            warnings.append(f"{label} failed: {err}. Answer produced by the offline extractive generator.")
            result = self._fallback_generator().generate(question, patient_id, contexts)
        warnings.extend(result.warnings)
        return Answer(
            question=question,
            patient_id=patient_id,
            text=result.text,
            citations=list(result.citations),
            retrieved=contexts,
            abstained=bool(result.abstained),
            mode=result.mode,
            model=result.model,
            warnings=warnings,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------ raw tables

    def _table(self, name: str) -> list[dict]:
        if name not in self._tables:
            path = Path(self.settings.data_dir) / f"{name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run `python -m ehr_rag generate` first")
            self._tables[name] = _read_jsonl(path)
        return self._tables[name]

    def list_patients(self) -> list[dict]:
        """Patient records from ``patients.jsonl`` (plus active problem abbreviations), sorted by id.

        ``indexed`` tells whether the patient has chunks in the loaded index.
        """
        indexed = set(self.patient_ids)
        try:
            patients = self._table("patients")
            conditions = self._table("conditions")
        except FileNotFoundError as exc:
            logger.warning("%s; listing indexed patient ids only", exc)
            return [{"patient_id": pid, "indexed": True, "problems": []} for pid in sorted(indexed)]
        problems: dict[str, list[str]] = {}
        for c in conditions:
            if c.get("status") == "active":
                problems.setdefault(c["patient_id"], []).append(c.get("abbreviation") or c.get("display", ""))
        out = []
        for p in sorted(patients, key=lambda r: r["patient_id"]):
            row = dict(p)
            row["problems"] = problems.get(p["patient_id"], [])
            row["indexed"] = p["patient_id"] in indexed
            out.append(row)
        return out

    def patient_summary(self, patient_id: str) -> dict:
        """Demographics, active problems, active medications and allergies from the raw tables."""
        patients = {p["patient_id"]: p for p in self._table("patients")}
        if patient_id not in patients:
            raise ValueError(f"Unknown patient_id {patient_id!r}. Valid ids: {', '.join(sorted(patients))}")

        def rows(table: str) -> list[dict]:
            return [r for r in self._table(table) if r.get("patient_id") == patient_id]

        problems = sorted((c for c in rows("conditions") if c.get("status") == "active"),
                          key=lambda c: (c.get("onset_date") or "", c.get("condition_key", "")))
        meds = sorted((m for m in rows("medications") if m.get("status") == "active"),
                      key=lambda m: (m.get("name", ""), m.get("med_id", "")))
        allergies = rows("allergies")
        encounters = sorted(rows("encounters"), key=lambda e: e.get("date", ""))
        last = encounters[-1] if encounters else None
        return {
            "patient_id": patient_id,
            "demographics": dict(patients[patient_id]),
            "active_problems": problems,
            "active_medications": meds,
            "allergies": allergies,
            "nkda": not allergies,
            "n_encounters": len(encounters),
            "last_encounter_date": (last.get("discharge_date") or last.get("date")) if last else None,
            "synthetic": True,
        }


__all__ = ["EHRRagPipeline", "IndexNotBuiltError", "PHIDetectedError"]
