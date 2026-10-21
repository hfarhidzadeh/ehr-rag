"""Patient-scoped retrieval: dense (Chroma), lexical (BM25), hybrid (RRF) and a random baseline control.

Patient isolation is a hard filter at every layer: Chroma ``where`` on ``patient_id``, one BM25 index per
patient, and a final re-check of ``chunk.patient_id`` on every result in :meth:`Retriever.retrieve`
(mismatches are dropped and logged, or raise :class:`PatientIsolationError` with ``strict=True``).
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import defaultdict

import numpy as np

from ehr_rag.embeddings import Embedder
from ehr_rag.types import RETRIEVAL_MODES, Chunk, RetrievedChunk
from ehr_rag.vectorstore import ChromaStore

logger = logging.getLogger(__name__)

# ISO dates and decimals stay whole ("2025-07-10", "7.2", "0.84"); otherwise runs of letters/digits
# ("hba1c", "a1c", "1000", "mg", "egfr", "losartan", "e11"). "s/p" -> "s", "p"; "132/84" -> "132", "84".
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[a-z0-9]+(?:\.\d+)*")

# Small English stopword list. Negations ("no", "not") and clinical words are deliberately kept.
STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if of to in on at by for with from as into about than then so
    is are was were be been being am has have had do does did doing
    this that these those it its there here
    what which who whom whose when where why how
    i me my we our you your he him his she her they them their
    can could would should will shall may might must
    any some all each every much many more most other such own same very just also please tell show""".split()
)


class PatientIsolationError(RuntimeError):
    """A retriever produced a chunk belonging to a different patient than the one requested."""


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25; keeps decimals, ISO dates and drug names, drops stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(ranked_ids: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: ``score(id) = sum 1 / (k + rank)`` (1-based rank). Ties sorted by id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_ids:
        seen: set[str] = set()
        for rank, cid in enumerate(ranking, start=1):
            if cid in seen:  # a list should not vote twice for one id
                continue
            seen.add(cid)
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


def _seed_for(query: str, patient_id: str) -> int:
    digest = hashlib.sha256(f"{patient_id}\x1f{query}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class BM25Index:
    """One ``BM25Okapi`` per patient over tokenised chunk text."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        by_patient: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            by_patient[c.patient_id].append(c)
        self._chunks: dict[str, list[Chunk]] = {}
        self._tokens: dict[str, list[list[str]]] = {}
        self._bm25: dict[str, BM25Okapi] = {}
        for pid, items in by_patient.items():
            items = sorted(items, key=lambda c: c.chunk_id)
            toks = [tokenize(c.text) for c in items]
            self._chunks[pid] = items
            self._tokens[pid] = toks
            if any(toks):  # BM25Okapi divides by the average document length
                self._bm25[pid] = BM25Okapi(toks)
        logger.info("Built BM25 index: %d chunks over %d patients", len(chunks), len(self._chunks))

    @property
    def patient_ids(self) -> list[str]:
        return sorted(self._chunks)

    def chunks_for(self, patient_id: str) -> list[Chunk]:
        """The patient's chunks sorted by ``chunk_id`` (empty for an unknown patient)."""
        return list(self._chunks.get(patient_id, []))

    def search(
        self, query: str, patient_id: str, k: int, doc_types: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top-``k`` of the patient's chunks sharing at least one query token, by BM25 score (desc)."""
        if not patient_id:
            raise ValueError("patient_id is required: retrieval is always restricted to one patient")
        bm25 = self._bm25.get(patient_id)
        if bm25 is None or k <= 0:
            return []
        q = set(tokenize(query))
        if not q:
            return []
        scores = bm25.get_scores(sorted(q))  # sorted: deterministic float summation order
        allowed = set(doc_types) if doc_types else None
        hits: list[tuple[Chunk, float]] = []
        for chunk, toks, score in zip(self._chunks[patient_id], self._tokens[patient_id], scores):
            if allowed is not None and chunk.doc_type not in allowed:
                continue
            if q.isdisjoint(toks):
                continue
            hits.append((chunk, float(score)))
        hits.sort(key=lambda t: (-t[1], t[0].chunk_id))
        return hits[:k]


class Retriever:
    """Runs one of :data:`~ehr_rag.types.RETRIEVAL_MODES` for a single patient."""

    def __init__(
        self,
        store: ChromaStore,
        embedder: Embedder,
        bm25: BM25Index,
        rrf_k: int = 60,
        candidate_pool: int = 30,
        strict: bool = False,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.rrf_k = int(rrf_k)
        self.candidate_pool = int(candidate_pool)
        self.strict = strict
        self.isolation_violations = 0  # count of wrong-patient chunks dropped (should stay 0)

    # ------------------------------------------------------------------ isolation

    def _guard(self, hits: list[tuple[Chunk, float]], patient_id: str, stage: str) -> list[tuple[Chunk, float]]:
        """Drop (or raise on, with ``strict``) any hit whose chunk belongs to another patient.

        Applied to every candidate list *before* fusion / truncation, so a leaked chunk can neither be
        returned nor push a legitimate chunk out of the top-k.
        """
        kept: list[tuple[Chunk, float]] = []
        for chunk, score in hits:
            if chunk.patient_id == patient_id:
                kept.append((chunk, score))
                continue
            self.isolation_violations += 1
            msg = (f"{stage} retrieval returned chunk {chunk.chunk_id} of patient {chunk.patient_id} "
                   f"for patient {patient_id}")
            if self.strict:
                raise PatientIsolationError(msg)
            logger.error("%s; dropped", msg)
        return kept

    # ------------------------------------------------------------------ modes

    def _bm25(self, query: str, patient_id: str, k: int, doc_types: list[str] | None) -> list[tuple[Chunk, float]]:
        return self._guard(self.bm25.search(query, patient_id, k, doc_types=doc_types), patient_id, "bm25")

    def _dense(self, query: str, patient_id: str, k: int, doc_types: list[str] | None) -> list[tuple[Chunk, float]]:
        vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        if not np.any(vec):  # e.g. tfidf query sharing no vocabulary: no semantic signal
            logger.debug("Zero query vector for %r; dense retrieval returns nothing", query)
            return []
        return self._guard(self.store.query(vec, k, patient_id, doc_types=doc_types), patient_id, "dense")

    def _random(self, query: str, patient_id: str, k: int, doc_types: list[str] | None) -> list[tuple[Chunk, float]]:
        pool = self.bm25.chunks_for(patient_id) or self.store.get_chunks(patient_id)
        pool = [c for c, _ in self._guard([(c, 0.0) for c in pool], patient_id, "random")]
        if doc_types:
            pool = [c for c in pool if c.doc_type in set(doc_types)]
        pool = sorted(pool, key=lambda c: c.chunk_id)
        rng = random.Random(_seed_for(query, patient_id))
        return [(c, 0.0) for c in rng.sample(pool, min(k, len(pool)))]

    def _hybrid(self, query: str, patient_id: str, k: int, doc_types: list[str] | None) -> list[tuple[Chunk, float]]:
        pool = max(self.candidate_pool, k)
        dense = self._dense(query, patient_id, pool, doc_types)
        lexical = self._bm25(query, patient_id, pool, doc_types)
        by_id = {c.chunk_id: c for c, _ in dense}
        by_id.update({c.chunk_id: c for c, _ in lexical})
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in lexical]], k=self.rrf_k
        )
        return [(by_id[cid], score) for cid, score in fused[:k]]

    # ------------------------------------------------------------------ public

    def retrieve(
        self,
        query: str,
        patient_id: str,
        k: int = 6,
        mode: str = "hybrid",
        doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks of ``patient_id`` for ``query``. Empty ``doc_types`` means no type filter."""
        if not patient_id:
            raise ValueError("patient_id is required: retrieval is always restricted to one patient")
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
        if k <= 0:
            return []
        doc_types = list(doc_types) if doc_types else None
        if mode == "dense":
            hits = self._dense(query, patient_id, k, doc_types)
        elif mode == "bm25":
            hits = self._bm25(query, patient_id, k, doc_types)
        elif mode == "hybrid":
            hits = self._hybrid(query, patient_id, k, doc_types)
        else:
            hits = self._random(query, patient_id, k, doc_types)

        hits = self._guard(hits, patient_id, mode)  # final re-check (no-op unless a stage misbehaved)
        out: list[RetrievedChunk] = []
        for chunk, score in hits:
            if doc_types and chunk.doc_type not in doc_types:
                continue
            out.append(RetrievedChunk(chunk=chunk, score=float(score), rank=len(out) + 1, source=mode))
            if len(out) >= k:
                break
        return out
