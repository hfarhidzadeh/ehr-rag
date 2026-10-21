"""Tests for ehr_rag.evaluation.metrics: keyword matching, retrieval metrics, rows and aggregation."""
from __future__ import annotations

import pytest

from ehr_rag.evaluation.metrics import (
    aggregate,
    answer_correct,
    citations_valid,
    make_row,
    normalize_tokens,
    reciprocal_rank,
    recall_at_k,
)
from ehr_rag.types import Answer, Chunk, Citation, EvalQuestion, RetrievedChunk

# ------------------------------------------------------------------ answer_correct


@pytest.mark.parametrize(
    "answer, keywords, expected",
    [
        ("The most recent A1c was 7.40 % [S1].", ["7.4"], True),
        ("A1c 7.4", ["7.40"], True),
        ("HbA1c   7.6\n(ref 4.0-5.6)", ["7.6"], True),
        ("value 17.4", ["7.4"], False),          # whole-token numbers
        ("value 7.45", ["7.4"], False),
        ("Drawn 2025-07-10.", ["7"], False),     # ISO dates stay whole
        ("Drawn 2025-07-10.", ["2025-07-10"], True),
        ("HbA1c 7.0 on 2025-07-10", ["7"], True),
        ("Takes LISINOPRIL 20MG daily", ["20 mg"], True),
        ("Takes lisinopril 120 mg daily", ["20 mg"], False),
        ("Reaction: Bronchospasm (severe).", ["bronchospasm"], True),
        ("stopped for hyperkalemia (K 5.7)", ["Na 130", "K 5.70"], True),   # any keyword
        ("Dry   cough on lisinopril", ["dry cough"], True),
        ("dry, cough", ["dry cough"], True),     # punctuation is a separator
        ("cough that was dry", ["dry cough"], False),
        ("anything", [], False),
        ("", ["7.4"], False),
        ("eGFR 52 mL/min/1.73m2", ["52"], True),
        ("sacubitril-valsartan 49-51 mg", ["49-51 mg"], True),
    ],
)
def test_answer_correct(answer, keywords, expected):
    assert answer_correct(answer, keywords) is expected


def test_normalize_tokens():
    assert normalize_tokens("A1c 7.40%, Cr 0.840 on 2025-07-10; 20mg") == [
        "a1c", "7.4", "cr", "0.84", "on", "2025-07-10", "20", "mg"]
    assert normalize_tokens("007 and 7.0") == ["7", "and", "7"]


# ------------------------------------------------------------------ retrieval metrics


def test_recall_and_rr():
    ret = ["d3", "d1", "d1", "d2"]
    assert recall_at_k(ret, ["d1"], 1) == 0.0
    assert recall_at_k(ret, ["d1"], 2) == 1.0
    assert recall_at_k(ret, ["d1", "d9"], 4) == 1.0
    assert recall_at_k(ret, ["d1", "d9"], 4, fraction=True) == 0.5
    assert recall_at_k(ret, [], 3) == 0.0
    assert recall_at_k(ret, ["d1"], 0) == 0.0
    assert reciprocal_rank(ret, ["d1", "d2"]) == 0.5
    assert reciprocal_rank(ret, ["d3"]) == 1.0
    assert reciprocal_rank(ret, ["d9"]) == 0.0
    assert reciprocal_rank([], ["d1"]) == 0.0


# ------------------------------------------------------------------ rows


def _chunk(cid: str, doc_id: str, pid: str = "P001") -> Chunk:
    return Chunk(cid, doc_id, pid, "clinical_note", "2025-01-01", "Note", None, f"[Patient {pid}]\ntext",
                 {"synthetic": True})


def _rc(chunks: list[Chunk]) -> list[RetrievedChunk]:
    return [RetrievedChunk(c, 1.0 / i, i, "hybrid") for i, c in enumerate(chunks, start=1)]


def _cit(label: str, c: Chunk) -> Citation:
    return Citation(label, c.chunk_id, c.doc_id, c.doc_type, c.date, c.title, "text")


def test_citations_valid():
    a, b = _chunk("P001-N01::000", "P001-N01"), _chunk("P001-N02::000", "P001-N02")
    ret = _rc([a, b])
    ok = Answer("q", "P001", "x [S2]", [_cit("S2", b)], ret, False, "extractive")
    assert citations_valid(ok) is True
    wrong_pos = Answer("q", "P001", "x [S1]", [_cit("S1", b)], ret, False, "extractive")
    assert citations_valid(wrong_pos) is False
    out_of_range = Answer("q", "P001", "x [S3]", [_cit("S3", a)], ret, False, "extractive")
    assert citations_valid(out_of_range) is False
    assert citations_valid(Answer("q", "P001", "no", [], ret, True, "extractive")) is None
    assert citations_valid(Answer("q", "P001", "ungrounded", [], ret, False, "extractive")) is False


def test_make_row():
    q = EvalQuestion("P001-fact_lookup-1", "P001", "What dose?", "fact_lookup", "20 mg", ["20 mg"],
                     ["P001-medications"])
    chunks = [_chunk("P001-medications::000", "P001-medications"), _chunk("P002-N01::000", "P002-N01", "P002")]
    ret = _rc(chunks)
    ans = Answer(q.question, "P001", "lisinopril 20 mg daily [S1]", [_cit("S1", chunks[0])], ret, False,
                 "extractive", latency_ms=12.5)
    row = make_row(q, ret, ans)
    assert row["retrieved_doc_ids"] == ["P001-medications", "P002-N01"]
    assert row["retrieved_patient_ids"] == ["P001", "P002"]
    assert row["expected_abstain"] is False and row["answer_correct"] is True
    assert row["citations_valid"] is True and row["latency_ms"] == 12.5 and row["k"] == 2
    retrieval_only = make_row(q, ret, None, latency_ms=3.0)
    assert retrieval_only["answer_text"] is None and retrieval_only["abstained"] is None


# ------------------------------------------------------------------ aggregate


def _row(qid, category, *, retrieved, gold, answer=None, abstained=None, keywords=(), expected_abstain=None,
         pids=None, cit=None, latency=10.0):
    return {
        "qid": qid, "category": category, "patient_id": "P001",
        "expected_abstain": expected_abstain if expected_abstain is not None
        else category in ("absent_fact", "cross_patient"),
        "retrieved_doc_ids": retrieved, "retrieved_patient_ids": pids or ["P001"] * len(retrieved),
        "gold_doc_ids": gold, "answer_keywords": list(keywords), "answer_text": answer, "abstained": abstained,
        "citations_valid": cit, "latency_ms": latency,
    }


def test_aggregate_metrics_and_none_for_undefined():
    rows = [
        _row("a", "fact_lookup", retrieved=["g1", "x", "y"], gold=["g1"], answer="dose 20 mg", abstained=False,
             keywords=["20 mg"], cit=True, latency=10),
        _row("b", "fact_lookup", retrieved=["x", "y", "g2"], gold=["g2"], answer="no record", abstained=True,
             keywords=["5 mg"], cit=None, latency=20),
        _row("c", "temporal", retrieved=["x", "g3", "y"], gold=["g3"], answer="A1c 6.9", abstained=False,
             keywords=["7.1"], cit=False, latency=30),
        _row("d", "absent_fact", retrieved=["x", "y", "z"], gold=[], answer="not documented", abstained=True,
             latency=40),
        _row("e", "cross_patient", retrieved=["x", "y", "z"], gold=[], answer="warfarin 5 mg", abstained=False,
             pids=["P001", "P002", "P002"], latency=50),
    ]
    agg = aggregate(rows)
    o = agg["overall"]
    assert o["n"] == 5 and o["n_positive"] == 3 and o["n_negative"] == 2
    assert o["recall@1"] == pytest.approx(1 / 3)
    assert o["recall@3"] == pytest.approx(1.0) and o["recall@k"] == pytest.approx(1.0)
    assert o["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert o["primary_recall@1"] == pytest.approx(1 / 3) and o["primary_recall@k"] == pytest.approx(1.0)
    assert o["answer_accuracy"] == pytest.approx(1 / 3)
    assert o["answered_accuracy"] == pytest.approx(1 / 2)
    assert o["false_abstention_rate"] == pytest.approx(1 / 3)
    assert o["correct_abstention_rate"] == pytest.approx(1 / 2)
    assert o["citation_validity"] == pytest.approx(1 / 2)
    assert o["leakage_rate"] == pytest.approx(1 / 5) and o["leaked_chunks"] == 2
    assert o["latency_mean_ms"] == pytest.approx(30.0) and o["latency_p95_ms"] == pytest.approx(48.0)

    cats = agg["by_category"]
    assert list(cats) == ["fact_lookup", "temporal", "absent_fact", "cross_patient"]
    neg = cats["absent_fact"]
    assert neg["recall@1"] is None and neg["mrr"] is None and neg["answer_accuracy"] is None
    assert neg["false_abstention_rate"] is None and neg["correct_abstention_rate"] == 1.0
    assert neg["citation_validity"] is None and neg["leakage_rate"] == 0.0
    assert cats["fact_lookup"]["correct_abstention_rate"] is None
    assert cats["cross_patient"]["leakage_rate"] == 1.0 and cats["cross_patient"]["correct_abstention_rate"] == 0.0
    assert agg["by_control"]["positive"]["n"] == 3 and agg["by_control"]["negative"]["n"] == 2


def test_aggregate_retrieval_only_and_empty():
    rows = [_row("a", "verbatim", retrieved=["g", "x"], gold=["g"], latency=None),
            _row("b", "absent_fact", retrieved=["x"], gold=[], latency=None)]
    o = aggregate(rows)["overall"]
    assert o["recall@1"] == 1.0 and o["recall@k"] == 1.0
    for key in ("answer_accuracy", "false_abstention_rate", "correct_abstention_rate", "citation_validity",
                "latency_mean_ms", "latency_p95_ms"):
        assert o[key] is None, key
    empty = aggregate([])
    assert empty["overall"]["n"] == 0 and empty["overall"]["recall@1"] is None and empty["by_category"] == {}


def test_aggregate_uses_precomputed_answer_correct_and_k():
    row = _row("a", "fact_lookup", retrieved=["x", "g"], gold=["g"], answer="whatever", abstained=False,
               keywords=["nope"])
    row["answer_correct"] = True
    row["k"] = 1
    o = aggregate([row])["overall"]
    assert o["answer_accuracy"] == 1.0
    assert o["recall@k"] == 0.0 and o["recall@3"] == 1.0

    multi = _row("m", "fact_lookup", retrieved=["note", "x"], gold=["list", "note"])
    m = aggregate([multi])["overall"]
    assert m["recall@1"] == 1.0 and m["primary_recall@1"] == 0.0 and m["primary_recall@k"] == 0.0


def test_evidence_recall_is_chunk_level():
    q = EvalQuestion("P001-fact_lookup-1", "P001", "What dose of lisinopril?", "fact_lookup", "20 mg", ["20 mg"],
                     ["P001-medications", "P001-N01"])
    exam = _chunk("P001-N01::003", "P001-N01")          # gold doc, but this chunk has no answer keyword
    exam.text = "[Patient P001 | Progress Note | 2025-01-01 | Section: Physical Exam]\nLungs: CTAB."
    plan = _chunk("P001-N01::005", "P001-N01")
    plan.text = "[Patient P001 | Progress Note | 2025-01-01 | Section: Assessment and Plan]\nlisinopril 20 mg daily."
    other = _chunk("P001-N02::001", "P001-N02")          # keyword, but not a gold doc
    other.text = "[Patient P001 | Progress Note | 2025-02-01]\natorvastatin 20 mg."
    row = make_row(q, _rc([exam, other, plan]))
    assert row["retrieved_evidence"] == [False, False, True]
    o = aggregate([row])["overall"]
    assert o["recall@1"] == 1.0                                  # document-level: the exam chunk counts
    assert o["evidence_recall@1"] == 0.0 and o["evidence_recall@k"] == 1.0
    # the header line is not evidence (a header date can equal an onset date)
    onset = EvalQuestion("P001-semantic_synonym-1", "P001", "When?", "semantic_synonym", "CAP onset 2026-01-22",
                         ["2026-01-22"], ["P001-N05"])
    hdr = _chunk("P001-N05::000", "P001-N05")
    hdr.text = "[Patient P001 | ED Note | 2026-01-22]\nCough and fever."
    assert make_row(onset, _rc([hdr]))["retrieved_evidence"] == [False]
    negative = EvalQuestion("P001-absent_fact-1", "P001", "q", "absent_fact", None, [], [])
    assert "retrieved_evidence" not in make_row(negative, _rc([hdr]))
