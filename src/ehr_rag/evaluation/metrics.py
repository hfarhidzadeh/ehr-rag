"""Evaluation metrics: retrieval (Recall@k, MRR), answer keyword match, abstention, citations, leakage, latency.

Row schema
----------
``aggregate`` consumes one ``dict`` per evaluated question (``make_row`` builds them):

=====================  ===================  ==========================================================
key                    type                 meaning
=====================  ===================  ==========================================================
``qid``                str                  ``EvalQuestion.qid``
``category``           str                  one of ``QuestionCategory``
``patient_id``         str                  patient the question was asked under (retrieval filter)
``expected_abstain``   bool                 True for negative controls (``expected_answer is None``)
``retrieved_doc_ids``  list[str]            ``doc_id`` of every retrieved chunk, in rank order. Positions
                                            are *chunk* ranks, so a document can appear more than once.
``retrieved_patient_ids`` list[str]         ``patient_id`` of every retrieved chunk, same order
``gold_doc_ids``       list[str]            ``EvalQuestion.gold_doc_ids`` (empty for negatives)
``answer_keywords``    list[str]            ``EvalQuestion.answer_keywords`` (needed for answer accuracy)
``answer_text``        str | None           generated answer; None when generation was skipped
``abstained``          bool | None          ``Answer.abstained``; None when generation was skipped
``citations_valid``    bool | None          every citation points at a retrieved chunk with a valid label
``latency_ms``         float | None         end-to-end latency of the question
``k``                  int (optional)       retrieval depth; defaults to ``len(retrieved_doc_ids)``
``answer_correct``     bool (optional)      precomputed match; otherwise derived from text + keywords
``retrieved_evidence`` list[bool] (opt.)    per retrieved chunk: from a gold doc *and* its body has an answer keyword
=====================  ===================  ==========================================================

Metric definitions (``None`` whenever a metric has no eligible rows in a group, never 0):

* ``recall@1`` / ``recall@3`` / ``recall@k`` - hit rate: 1.0 if *any* gold document is among the first
  1 / 3 / k retrieved chunks (every gold document holds the answer, so one suffices). Positives with gold only.
* ``primary_recall@1`` / ``primary_recall@k`` - the same hit rate on ``gold_doc_ids[0]`` only. The question
  builder always puts the canonical document first (the structured list / lab panel holding the fact, or the
  quoted note), so this is stricter than ``recall@k`` when a fact is repeated in many notes (allergies and
  medications are listed in every note, which inflates the random-retriever baseline on ``recall@k``).
* ``evidence_recall@1`` / ``evidence_recall@k`` - chunk-level hit rate: a retrieved chunk counts only if it
  belongs to a gold document *and* its own text contains an answer keyword (``retrieved_evidence``). Gold is
  document-level, so ``recall@k`` also credits e.g. the PHYSICAL EXAM chunk of a note whose A&P holds the
  answer; this stricter variant does not. Positives whose rows carry ``retrieved_evidence``.
* ``mrr`` - mean reciprocal rank of the first gold document. Positives with gold only.
* ``answer_accuracy`` - positives with a generated answer: share that did not abstain *and* match a keyword.
* ``answered_accuracy`` - the same numerator over the non-abstained positives only.
* ``false_abstention_rate`` - positives with a generated answer that abstained.
* ``correct_abstention_rate`` - negatives with a generated answer that abstained.
* ``citation_validity`` - share of rows with ``citations_valid`` not None that are True.
* ``leakage_rate`` - share of questions with any retrieved chunk from another patient (must be 0);
  ``leaked_chunks`` is the total count of such chunks.
* ``latency_mean_ms`` / ``latency_p95_ms`` - over rows with a latency; ``n`` - number of rows.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ehr_rag.types import NEGATIVE_CATEGORIES, POSITIVE_CATEGORIES, Answer, EvalQuestion, RetrievedChunk

logger = logging.getLogger(__name__)

CATEGORY_ORDER: tuple[str, ...] = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES

# ISO dates stay whole (so "7" never matches the month of "2025-07-10"); otherwise alphanumeric runs with
# optional decimal parts ("7.4", "e11.9", "hba1c"). Punctuation and whitespace are separators.
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[a-z0-9]+(?:\.\d+)*")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_NUMBER_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$")   # "20mg" -> "20", "mg"


# --------------------------------------------------------------------------- answer matching


def _canonical_number(token: str) -> str:
    """``"7.40"`` -> ``"7.4"``, ``"07"`` -> ``"7"``, ``"7.0"`` -> ``"7"``, ``"0.84"`` -> ``"0.84"``."""
    whole, _, frac = token.partition(".")
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


def normalize_tokens(text: str) -> list[str]:
    """Lower-cased, punctuation-free tokens with numbers in canonical form (used for keyword matching)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        m = _NUMBER_UNIT_RE.match(tok)
        if m:
            out.extend([_canonical_number(m.group(1)), m.group(2)])
        elif _NUMBER_RE.match(tok):
            out.append(_canonical_number(tok))
        else:
            out.append(tok)
    return out


def _contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and list(haystack[i:i + n]) == list(needle):
            return True
    return False


def matched_keywords(answer_text: str, keywords: Iterable[str]) -> list[str]:
    """Keywords whose token sequence occurs contiguously in ``answer_text`` (see ``answer_correct``)."""
    tokens = normalize_tokens(answer_text)
    return [kw for kw in keywords if _contains(tokens, normalize_tokens(kw))]


def answer_correct(answer_text: str, keywords: list[str]) -> bool:
    """True if any keyword occurs in the answer.

    Case-insensitive and whitespace/punctuation-insensitive: both sides are tokenised by ``normalize_tokens``
    and a keyword matches when its tokens appear contiguously in the answer's tokens. Numeric tokens compare
    numerically (``"7.40"`` == ``"7.4"``) and whole-token (``"7.4"`` does not match ``"17.4"``); ISO dates
    are single tokens. No keywords, or an empty answer, is never correct.
    """
    if not answer_text or not keywords:
        return False
    return bool(matched_keywords(answer_text, keywords))


# --------------------------------------------------------------------------- retrieval


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int, fraction: bool = False) -> float:
    """Hit rate at ``k``: 1.0 if any gold doc is among the first ``k`` retrieved ids, else 0.0.

    With ``fraction=True`` returns the classic recall (share of distinct gold docs found in the top ``k``).
    Empty gold or ``k <= 0`` gives 0.0.
    """
    gold = set(gold_doc_ids)
    if not gold or k <= 0:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    if fraction:
        return len(top & gold) / len(gold)
    return 1.0 if top & gold else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    """``1 / rank`` (1-based) of the first retrieved id that is a gold doc; 0.0 if none (or no gold)."""
    gold = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- rows


def citations_valid(answer: Answer, retrieved: Optional[list[RetrievedChunk]] = None) -> Optional[bool]:
    """Every citation label is ``S1..Sk`` and points at the chunk retrieved at that position.

    ``retrieved`` defaults to ``answer.retrieved``. Returns None for an answer without citations that
    abstained (nothing to validate); a non-abstained answer without citations is invalid (ungrounded).
    """
    contexts = answer.retrieved if retrieved is None else retrieved
    if not answer.citations:
        return None if answer.abstained else False
    by_label = {f"S{i}": rc.chunk for i, rc in enumerate(contexts, start=1)}
    for c in answer.citations:
        chunk = by_label.get(c.label)
        if chunk is None or chunk.chunk_id != c.chunk_id or chunk.patient_id != answer.patient_id:
            return False
    return True


def _chunk_body(text: str) -> str:
    """Chunk text without its ``[Patient ... ]`` header line (a header date must not count as evidence)."""
    head, sep, body = text.partition("\n")
    return body if sep and head.startswith("[") and head.rstrip().endswith("]") else text


def make_row(
    question: EvalQuestion,
    retrieved: list[RetrievedChunk],
    answer: Optional[Answer] = None,
    latency_ms: Optional[float] = None,
    k: Optional[int] = None,
) -> dict[str, Any]:
    """Build one row of the schema in the module docstring from a question and its results."""
    row: dict[str, Any] = {
        "qid": question.qid,
        "category": question.category,
        "patient_id": question.patient_id,
        "expected_abstain": question.expected_answer is None,
        "retrieved_doc_ids": [rc.chunk.doc_id for rc in retrieved],
        "retrieved_patient_ids": [rc.chunk.patient_id for rc in retrieved],
        "gold_doc_ids": list(question.gold_doc_ids),
        "answer_keywords": list(question.answer_keywords),
        "answer_text": None,
        "abstained": None,
        "citations_valid": None,
        "latency_ms": latency_ms,
        "k": k if k is not None else len(retrieved),
    }
    if question.expected_answer is not None and question.gold_doc_ids:
        gold = set(question.gold_doc_ids)
        row["retrieved_evidence"] = [rc.chunk.doc_id in gold and answer_correct(_chunk_body(rc.chunk.text),
                                                                                question.answer_keywords)
                                     for rc in retrieved]
    if answer is not None:
        row["answer_text"] = answer.text
        row["abstained"] = bool(answer.abstained)
        row["citations_valid"] = citations_valid(answer, retrieved)
        row["answer_correct"] = answer_correct(answer.text, question.answer_keywords)
        if latency_ms is None:
            row["latency_ms"] = answer.latency_ms
    return row


# --------------------------------------------------------------------------- aggregation


def _mean(values: list[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _is_negative(row: dict[str, Any]) -> bool:
    if row.get("expected_abstain") is not None:
        return bool(row["expected_abstain"])
    return row.get("category") in NEGATIVE_CATEGORIES


def _row_correct(row: dict[str, Any]) -> bool:
    if row.get("answer_correct") is not None:
        return bool(row["answer_correct"])
    return answer_correct(row.get("answer_text") or "", list(row.get("answer_keywords") or []))


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if not _is_negative(r)]
    negatives = [r for r in rows if _is_negative(r)]
    with_gold = [r for r in positives if r.get("gold_doc_ids") and r.get("retrieved_doc_ids") is not None]

    def recall(k_of) -> Optional[float]:
        return _mean([recall_at_k(list(r["retrieved_doc_ids"]), list(r["gold_doc_ids"]), k_of(r))
                      for r in with_gold])

    with_evidence = [r for r in with_gold if r.get("retrieved_evidence") is not None]

    def evidence(k_of) -> Optional[float]:
        return _mean([1.0 if any(r["retrieved_evidence"][:k_of(r)]) else 0.0 for r in with_evidence])

    pos_gen = [r for r in positives if r.get("abstained") is not None]
    neg_gen = [r for r in negatives if r.get("abstained") is not None]
    pos_answered = [r for r in pos_gen if not r["abstained"]]
    pos_correct = [r for r in pos_answered if _row_correct(r)]

    cit = [bool(r["citations_valid"]) for r in rows if r.get("citations_valid") is not None]

    leak_rows = [r for r in rows if r.get("retrieved_patient_ids") is not None]
    leaked_per_row = [sum(1 for p in r["retrieved_patient_ids"] if p != r.get("patient_id")) for r in leak_rows]

    lat = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]

    return {
        "n": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "recall@1": recall(lambda r: 1),
        "recall@3": recall(lambda r: 3),
        "recall@k": recall(lambda r: int(r.get("k") or len(r["retrieved_doc_ids"]))),
        "primary_recall@1": _mean([recall_at_k(list(r["retrieved_doc_ids"]), [r["gold_doc_ids"][0]], 1)
                                   for r in with_gold]),
        "primary_recall@k": _mean([recall_at_k(list(r["retrieved_doc_ids"]), [r["gold_doc_ids"][0]],
                                               int(r.get("k") or len(r["retrieved_doc_ids"]))) for r in with_gold]),
        "evidence_recall@1": evidence(lambda r: 1),
        "evidence_recall@k": evidence(lambda r: int(r.get("k") or len(r["retrieved_doc_ids"]))),
        "mrr": _mean([reciprocal_rank(list(r["retrieved_doc_ids"]), list(r["gold_doc_ids"])) for r in with_gold]),
        "answer_accuracy": (len(pos_correct) / len(pos_gen)) if pos_gen else None,
        "answered_accuracy": (len(pos_correct) / len(pos_answered)) if pos_answered else None,
        "false_abstention_rate": (sum(1 for r in pos_gen if r["abstained"]) / len(pos_gen)) if pos_gen else None,
        "correct_abstention_rate": (sum(1 for r in neg_gen if r["abstained"]) / len(neg_gen)) if neg_gen else None,
        "citation_validity": _mean([1.0 if c else 0.0 for c in cit]),
        "leakage_rate": (sum(1 for n in leaked_per_row if n) / len(leak_rows)) if leak_rows else None,
        "leaked_chunks": sum(leaked_per_row) if leak_rows else None,
        "latency_mean_ms": _mean(lat),
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None,
    }


def aggregate(rows: list[dict]) -> dict:
    """Overall, per-control-type and per-category metrics (see module docstring for definitions).

    Returns ``{"overall": {...}, "by_control": {"positive": {...}, "negative": {...}},
    "by_category": {category: {...}}}``; categories appear in ``QuestionCategory`` order, unknown ones last.
    """
    rows = list(rows)
    cats = sorted({r.get("category") for r in rows},
                  key=lambda c: (CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER), str(c)))
    result = {
        "overall": _group_metrics(rows),
        "by_control": {
            "positive": _group_metrics([r for r in rows if not _is_negative(r)]),
            "negative": _group_metrics([r for r in rows if _is_negative(r)]),
        },
        "by_category": {c: _group_metrics([r for r in rows if r.get("category") == c]) for c in cats},
    }
    overall = result["overall"]
    if overall["leaked_chunks"]:
        logger.error("Patient isolation violated: %d leaked chunks across %d rows", overall["leaked_chunks"],
                     len(rows))
    return result
