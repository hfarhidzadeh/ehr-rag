"""Persistent ChromaDB vector store for :class:`~ehr_rag.types.Chunk` objects.

* One collection per embedder (``Settings.collection_name``), HNSW ``cosine`` space, no Chroma-side
  embedding function (vectors always come from :mod:`ehr_rag.embeddings`), telemetry off.
* Chroma metadata must be scalar and non-None, so chunk fields are stored flat
  (``chunk_id``, ``doc_id``, ``patient_id``, ``doc_type``, ``date``, ``title``, ``section``) and
  ``Chunk.metadata`` is stored as a JSON string under ``metadata_json``. ``section`` is omitted from the
  Chroma record when it is ``None``; the chunk text is stored as the Chroma document.
* :meth:`ChromaStore.query` always filters on ``patient_id`` (mandatory) and re-checks it on the results.

Chroma 1.5.x reports cosine *distance* (``1 - cos``); :meth:`ChromaStore.query` returns
``similarity = 1 - distance``, which equals the dot product for L2-normalised vectors.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ehr_rag.types import Chunk

logger = logging.getLogger(__name__)

_GET_PAGE_SIZE = 2000
_HNSW_CONFIG: dict[str, Any] = {"space": "cosine", "ef_search": 200, "ef_construction": 200}


def _chunk_to_record(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "patient_id": chunk.patient_id,
        "doc_type": chunk.doc_type,
        "date": chunk.date,
        "title": chunk.title,
        "metadata_json": json.dumps(chunk.metadata, sort_keys=True, ensure_ascii=False),
    }
    if chunk.section is not None:
        meta["section"] = chunk.section
    return meta


def _record_to_chunk(chunk_id: str, text: str | None, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=meta.get("chunk_id", chunk_id),
        doc_id=meta["doc_id"],
        patient_id=meta["patient_id"],
        doc_type=meta["doc_type"],
        date=meta["date"],
        title=meta["title"],
        section=meta.get("section"),
        text=text or "",
        metadata=json.loads(meta.get("metadata_json") or "{}"),
    )


def build_where(patient_id: str, doc_types: list[str] | None = None) -> dict[str, Any]:
    """Chroma ``where`` filter: ``patient_id`` equality, AND-ed with ``doc_type $in`` when given."""
    if not patient_id:
        raise ValueError("patient_id is required: retrieval is always restricted to one patient")
    clause: dict[str, Any] = {"patient_id": patient_id}
    if doc_types:
        return {"$and": [clause, {"doc_type": {"$in": list(doc_types)}}]}
    return clause


class ChromaStore:
    """Thin wrapper over a persistent Chroma collection holding chunk embeddings."""

    def __init__(self, persist_dir: Path, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            self.collection_name,
            configuration={"hnsw": dict(_HNSW_CONFIG)},
            metadata={"synthetic": True},
            embedding_function=None,
        )

    @property
    def space(self) -> str | None:
        """The HNSW distance space actually configured on the collection (expected ``"cosine"``)."""
        with warnings.catch_warnings():  # Chroma warns about the "legacy" (absent) embedding function
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = self._collection.configuration or {}
        hnsw = cfg.get("hnsw") or {}
        return hnsw.get("space")

    def reset(self) -> None:
        """Drop and recreate the collection (all vectors are removed)."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # collection did not exist
            logger.debug("delete_collection(%s) ignored: %s", self.collection_name, exc)
        self._collection = self._open_collection()
        logger.info("Reset Chroma collection %s in %s", self.collection_name, self.persist_dir)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 512) -> None:
        """Insert chunks with their vectors in batches. Re-adding an existing ``chunk_id`` replaces it."""
        chunks = list(chunks)
        emb = np.asarray(embeddings, dtype=np.float32)
        if not chunks:
            return
        if emb.ndim != 2 or emb.shape[0] != len(chunks):
            raise ValueError(f"embeddings shape {emb.shape} does not match {len(chunks)} chunks")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id values in add()")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=emb[start : start + batch_size],
                metadatas=[_chunk_to_record(c) for c in batch],
                documents=[c.text for c in batch],
            )
        logger.info("Added %d chunks (dim=%d) to %s", len(chunks), emb.shape[1], self.collection_name)

    def query(
        self,
        embedding: np.ndarray,
        k: int,
        patient_id: str,
        doc_types: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks of one patient by cosine similarity (``1 - distance``), descending.

        ``k`` larger than the patient's chunk count returns all of that patient's chunks.
        """
        where = build_where(patient_id, doc_types)
        if k <= 0:
            return []
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=int(k),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[tuple[Chunk, float]] = []
        for cid, meta, text, dist in zip(ids, metas, docs, dists):
            chunk = _record_to_chunk(cid, text, meta)
            if chunk.patient_id != patient_id:  # defence in depth; Chroma's filter should make this impossible
                logger.error("Vector store returned chunk %s of patient %s for patient %s; dropped",
                             chunk.chunk_id, chunk.patient_id, patient_id)
                continue
            out.append((chunk, 1.0 - float(dist)))
        out.sort(key=lambda t: (-t[1], t[0].chunk_id))
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def _get_all(self, patient_id: str | None, with_embeddings: bool) -> tuple[list[Chunk], list[Any]]:
        include = ["metadatas", "documents"] + (["embeddings"] if with_embeddings else [])
        where = {"patient_id": patient_id} if patient_id else None
        chunks: list[Chunk] = []
        vectors: list[Any] = []
        offset = 0
        while True:
            res = self._collection.get(where=where, include=include, limit=_GET_PAGE_SIZE, offset=offset)
            ids = res.get("ids") or []
            if not ids:
                break
            metas, docs = res["metadatas"], res["documents"]
            embs = res.get("embeddings") if with_embeddings else None
            for i, cid in enumerate(ids):
                chunks.append(_record_to_chunk(cid, docs[i], metas[i]))
                if with_embeddings:
                    vectors.append(embs[i])
            if len(ids) < _GET_PAGE_SIZE:
                break
            offset += len(ids)
        return chunks, vectors

    def get_chunks(self, patient_id: str | None = None) -> list[Chunk]:
        """All stored chunks (optionally of one patient), sorted by ``chunk_id``."""
        chunks, _ = self._get_all(patient_id, with_embeddings=False)
        return sorted(chunks, key=lambda c: c.chunk_id)

    def get_embeddings(self, patient_id: str | None = None) -> tuple[list[Chunk], np.ndarray]:
        """Chunks sorted by ``chunk_id`` and their vectors as an ``(n, dim)`` float32 array (same order)."""
        chunks, vectors = self._get_all(patient_id, with_embeddings=True)
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].chunk_id)
        chunks = [chunks[i] for i in order]
        if not order:
            return [], np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray([vectors[i] for i in order], dtype=np.float32)
        return chunks, arr
