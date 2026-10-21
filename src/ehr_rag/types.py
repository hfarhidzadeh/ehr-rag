"""Shared data contracts for every EHR-RAG module.

All modules exchange these dataclasses. Keep them JSON-serialisable (``to_dict`` / ``from_dict``)
so they can be written to JSONL, stored as Chroma metadata, and shown in the UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional

# --------------------------------------------------------------------------- vocab

DocType = Literal[
    "demographics",
    "problem_list",
    "medication_list",
    "allergy_list",
    "lab_panel",
    "vitals",
    "clinical_note",
]
DOC_TYPES: tuple[str, ...] = (
    "demographics",
    "problem_list",
    "medication_list",
    "allergy_list",
    "lab_panel",
    "vitals",
    "clinical_note",
)

NoteType = Literal["progress_note", "discharge_summary", "ed_note", "radiology_report", "consult_note"]
NOTE_TYPES: tuple[str, ...] = ("progress_note", "discharge_summary", "ed_note", "radiology_report", "consult_note")

RetrievalMode = Literal["dense", "bm25", "hybrid", "random"]
RETRIEVAL_MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "random")

QuestionCategory = Literal[
    "fact_lookup",       # positive: value present in chart
    "temporal",          # positive: "most recent X"
    "semantic_synonym",  # positive: lay wording vs. abbreviated note text
    "verbatim",          # positive control: question quotes chunk text
    "absent_fact",       # negative control: fact not in this patient's chart
    "cross_patient",     # negative control: another patient's fact asked under this patient
]
POSITIVE_CATEGORIES: tuple[str, ...] = ("fact_lookup", "temporal", "semantic_synonym", "verbatim")
NEGATIVE_CATEGORIES: tuple[str, ...] = ("absent_fact", "cross_patient")


# --------------------------------------------------------------------------- doc ids
# Deterministic ids let the evaluation module know which document holds a fact
# without re-reading the generator's internals.

def demographics_doc_id(patient_id: str) -> str:
    return f"{patient_id}-demographics"


def problems_doc_id(patient_id: str) -> str:
    return f"{patient_id}-problems"


def medications_doc_id(patient_id: str) -> str:
    return f"{patient_id}-medications"


def allergies_doc_id(patient_id: str) -> str:
    return f"{patient_id}-allergies"


def lab_panel_doc_id(patient_id: str, date: str) -> str:
    return f"{patient_id}-labs-{date}"


def vitals_doc_id(patient_id: str, date: str) -> str:
    return f"{patient_id}-vitals-{date}"


def chunk_id_for(doc_id: str, index: int) -> str:
    return f"{doc_id}::{index:03d}"


# --------------------------------------------------------------------------- helpers

class _Serde:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in names})


# --------------------------------------------------------------------------- records

@dataclass
class Document(_Serde):
    """One retrievable source: a clinical note or a rendered structured table.

    ``text`` for clinical notes uses upper-case section headers on their own line, e.g.
    ``"CHIEF COMPLAINT:\\nShortness of breath...\\n\\nASSESSMENT AND PLAN:\\n..."``.
    """

    doc_id: str
    patient_id: str
    doc_type: str            # one of DOC_TYPES
    date: str                # ISO yyyy-mm-dd (for lists: date last updated)
    title: str               # e.g. "Discharge Summary", "Lab Panel 2025-03-14"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)  # must include "synthetic": True


@dataclass
class Chunk(_Serde):
    """A retrievable unit. ``text`` already includes the header context line."""

    chunk_id: str            # chunk_id_for(doc_id, i)
    doc_id: str
    patient_id: str
    doc_type: str
    date: str
    title: str
    section: Optional[str]   # note section name, or None for structured docs
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)  # carries "synthetic": True


@dataclass
class RetrievedChunk(_Serde):
    chunk: Chunk
    score: float             # higher is better (cosine sim, BM25 score, or RRF score)
    rank: int                # 1-based
    source: str              # "dense" | "bm25" | "hybrid" | "random"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedChunk":
        return cls(
            chunk=Chunk.from_dict(data["chunk"]),
            score=float(data["score"]),
            rank=int(data["rank"]),
            source=data["source"],
        )


@dataclass
class Citation(_Serde):
    label: str               # "S1".."Sk" — position in the context given to the generator
    chunk_id: str
    doc_id: str
    doc_type: str
    date: str
    title: str
    snippet: str             # short supporting excerpt


@dataclass
class Answer(_Serde):
    question: str
    patient_id: str
    text: str                # answer prose with inline [S#] markers
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    abstained: bool          # True when the record does not support an answer
    mode: str                # "llm" | "extractive"
    model: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Answer":
        d = dict(data)
        d["citations"] = [Citation.from_dict(c) for c in d.get("citations", [])]
        d["retrieved"] = [RetrievedChunk.from_dict(r) for r in d.get("retrieved", [])]
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class EvalQuestion(_Serde):
    qid: str
    patient_id: str          # patient the question is asked under (retrieval filter)
    question: str
    category: str            # one of QuestionCategory
    expected_answer: Optional[str]          # None => the system should abstain
    answer_keywords: list[str] = field(default_factory=list)  # any-match, case-insensitive
    gold_doc_ids: list[str] = field(default_factory=list)     # docs that contain the answer
    metadata: dict[str, Any] = field(default_factory=dict)    # e.g. source_patient_id for cross_patient
