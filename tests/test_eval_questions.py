"""Tests for ehr_rag.evaluation.questions: categories, gold docs, controls, validation and IO."""
from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from ehr_rag.config import PROJECT_ROOT
from ehr_rag.evaluation.metrics import answer_correct
from ehr_rag.evaluation.questions import (
    CATEGORIES,
    build_questions,
    condition_surface_forms,
    read_questions,
    reason_keywords,
    term_in_text,
    validate_questions,
    write_questions,
)
from ehr_rag.ingest.chunking import chunk_documents
from ehr_rag.synth import vocab as V
from ehr_rag.synth.generator import format_lab_value, generate_dataset, load_documents
from ehr_rag.types import NEGATIVE_CATEGORIES, POSITIVE_CATEGORIES, EvalQuestion, medications_doc_id

DATA_DIR: Path = PROJECT_ROOT / "data" / "synthetic"
EXPECTED_PER_PATIENT = {"fact_lookup": 2, "temporal": 1, "semantic_synonym": 1, "verbatim": 1, "absent_fact": 1,
                        "cross_patient": 1}


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_patients=8, seed=7)


@pytest.fixture(scope="module")
def docs(dataset):
    return dataset.to_documents()


@pytest.fixture(scope="module")
def chunks(docs):
    return chunk_documents(docs)


@pytest.fixture(scope="module")
def questions(dataset, chunks):
    return build_questions(dataset, chunks, seed=11)


@pytest.fixture(scope="module")
def doc_by_id(docs):
    return {d.doc_id: d for d in docs}


def _patient_text(docs, pid: str) -> str:
    return "\n".join(d.text for d in docs if d.patient_id == pid)


# ------------------------------------------------------------------ structure / determinism


def test_counts_per_patient_and_unique_qids(questions, dataset):
    assert len({q.qid for q in questions}) == len(questions)
    by_pid: dict[str, Counter] = {}
    for q in questions:
        by_pid.setdefault(q.patient_id, Counter())[q.category] += 1
    assert set(by_pid) == {p["patient_id"] for p in dataset.patients}
    for pid, counts in by_pid.items():
        assert dict(counts) == EXPECTED_PER_PATIENT, pid
    assert {q.category for q in questions} == set(CATEGORIES)
    assert all(q.metadata["synthetic"] is True for q in questions)


def test_deterministic_and_seed_dependent(dataset, chunks, questions):
    again = build_questions(dataset, chunks, seed=11)
    assert [q.to_dict() for q in again] == [q.to_dict() for q in questions]
    other = build_questions(dataset, chunks, seed=12)
    assert [q.to_dict() for q in other] != [q.to_dict() for q in questions]


def test_fact_kinds_are_varied(questions):
    kinds = Counter(q.metadata["fact_kind"] for q in questions if q.category == "fact_lookup")
    assert len(kinds) >= 3
    for q in questions:
        if q.category == "fact_lookup":
            pair = [x.metadata["fact_kind"] for x in questions
                    if x.patient_id == q.patient_id and x.category == "fact_lookup"]
            assert len(set(pair)) == 2


def test_without_chunks_skips_verbatim(dataset):
    qs = build_questions(dataset, None, seed=11)
    assert "verbatim" not in {q.category for q in qs}
    assert {"fact_lookup", "temporal", "semantic_synonym", "absent_fact", "cross_patient"} <= {q.category for q in qs}


# ------------------------------------------------------------------ positives


def test_positive_gold_exists_and_contains_keyword(questions, doc_by_id):
    for q in questions:
        if q.category not in POSITIVE_CATEGORIES:
            continue
        assert q.expected_answer and q.answer_keywords and q.gold_doc_ids, q.qid
        assert all(g in doc_by_id and doc_by_id[g].patient_id == q.patient_id for g in q.gold_doc_ids), q.qid
        assert any(answer_correct(doc_by_id[g].text, q.answer_keywords) for g in q.gold_doc_ids), q.qid
        assert len(set(q.gold_doc_ids)) == len(q.gold_doc_ids)


def test_lab_value_and_temporal_keywords_are_rendered_values(questions, dataset, doc_by_id):
    labs = {(r["patient_id"], r["test_key"], r["date"]): r for r in dataset.labs}
    seen = 0
    for q in questions:
        if q.category == "temporal" or q.metadata.get("fact_kind") == "lab_value":
            key, day = q.metadata["test_key"], q.metadata["date"]
            row = labs[(q.patient_id, key, day)]
            assert q.answer_keywords == [format_lab_value(key, row["value"])]
            assert q.gold_doc_ids[0] == f"{q.patient_id}-labs-{day}"
            for g in q.gold_doc_ids[1:]:
                assert [key, day] in doc_by_id[g].metadata["mentions"]["labs"]
            seen += 1
        if q.category == "temporal":
            dates = sorted(r["date"] for (p, k, _), r in labs.items() if p == q.patient_id and k == key)
            assert len(dates) >= 2 and day == dates[-1]
            assert "most recent" in q.question
    assert seen >= 8


def test_med_and_allergy_facts(questions, dataset, doc_by_id):
    meds = {m["med_id"]: m for m in dataset.medications}
    for q in questions:
        kind = q.metadata.get("fact_kind")
        if kind == "med_dose":
            med = meds[q.metadata["med_id"]]
            assert med["status"] == "active" and q.answer_keywords == [med["dose"]]
            assert med["name"] in q.question and q.gold_doc_ids[0] == medications_doc_id(q.patient_id)
        elif kind == "stop_reason":
            med = meds[q.metadata["med_id"]]
            assert med["status"] == "discontinued"
            assert any(answer_correct(med["discontinue_reason"], [k]) for k in q.answer_keywords)
            for g in q.gold_doc_ids[1:]:
                assert doc_by_id[g].date == med["end_date"]
                assert med["med_id"] in doc_by_id[g].metadata["mentions"]["medications"]
        elif kind == "allergy_reaction":
            assert q.metadata["substance"] in q.question
            assert q.gold_doc_ids[0] == f"{q.patient_id}-allergies"


def test_semantic_questions_use_only_lay_terms(questions):
    sem = [q for q in questions if q.category == "semantic_synonym"]
    assert sem
    for q in sem:
        key = q.metadata["condition_key"]
        assert not any(term_in_text(t, q.question) for t in condition_surface_forms(key)), q.question
        assert any(term_in_text(t, q.question) for t in V.CONDITIONS[key].lay_terms)
        assert V.CONDITIONS[key].abbreviation.lower() not in q.question.lower()


def test_verbatim_quotes_chunk_sentence(questions, chunks, doc_by_id):
    by_cid = {c.chunk_id: c for c in chunks}
    for q in [q for q in questions if q.category == "verbatim"]:
        ch = by_cid[q.metadata["chunk_id"]]
        assert q.gold_doc_ids == [ch.doc_id] and ch.patient_id == q.patient_id
        sentence = q.expected_answer
        assert len(sentence.split()) >= 8
        assert sentence in ch.text and sentence in q.question
        assert len(q.answer_keywords) == 2
        assert all(term_in_text(k, sentence) for k in q.answer_keywords)
        others = [d for d in doc_by_id.values() if d.patient_id == q.patient_id and d.doc_id != ch.doc_id]
        assert not any(" ".join(sentence.split()) in " ".join(d.text.split()) for d in others)


# ------------------------------------------------------------------ negatives


def test_negative_controls_absent_from_patient(questions, docs):
    negs = [q for q in questions if q.category in NEGATIVE_CATEGORIES]
    assert negs
    for q in negs:
        assert q.expected_answer is None and q.gold_doc_ids == [] and q.answer_keywords == []
        text = _patient_text(docs, q.patient_id)
        assert q.metadata["key_term"]
        for term in q.metadata["key_terms"]:
            assert not term_in_text(term, text), (q.qid, term)


def test_cross_patient_source_holds_fact(questions, docs, dataset):
    holders = Counter()
    for pid in {p["patient_id"] for p in dataset.patients}:
        holders.update({a["substance"] for a in dataset.allergies if a["patient_id"] == pid}
                       | {m["name"] for m in dataset.medications if m["patient_id"] == pid})
    for q in [q for q in questions if q.category == "cross_patient"]:
        src = q.metadata["source_patient_id"]
        assert src != q.patient_id
        assert term_in_text(q.metadata["key_term"], _patient_text(docs, src))
        assert q.metadata["n_holders"] == holders[q.metadata["key_term"]]
        assert q.metadata["source_doc_ids"][0].startswith(src + "-")


def test_negative_wording_matches_positive_templates(questions):
    starts = {tuple(q.question.split()[:3]) for q in questions if q.category in POSITIVE_CATEGORIES}
    for q in questions:
        if q.category in NEGATIVE_CATEGORIES:
            assert tuple(q.question.split()[:3]) in starts, q.question


# ------------------------------------------------------------------ validation catches violations


def _replace(questions, qid, **changes):
    return [dataclasses.replace(q, **changes) if q.qid == qid else q for q in questions]


def test_validation_rejects_bad_questions(questions, docs, chunks):
    validate_questions(questions, docs, chunks)   # sanity: the real set passes
    pos = next(q for q in questions if q.category == "fact_lookup")
    neg = next(q for q in questions if q.category == "absent_fact")
    sem = next(q for q in questions if q.category == "semantic_synonym")
    med_list = next(d for d in docs if d.doc_id == medications_doc_id(neg.patient_id)).text
    own_med = med_list.split("\n")[1][2:].split()[0]   # first drug on the patient's own med list
    abbr = V.CONDITIONS[sem.metadata["condition_key"]].abbreviation

    cases = [
        (_replace(questions, pos.qid, answer_keywords=["zzz-not-in-chart"]), "no keyword"),
        (_replace(questions, pos.qid, gold_doc_ids=["P999-missing"]), "do not exist"),
        (_replace(questions, neg.qid, metadata={**neg.metadata, "key_terms": [own_med]}), "occur in patient"),
        (_replace(questions, neg.qid, gold_doc_ids=[pos.gold_doc_ids[0]]), "negative question"),
        (_replace(questions, sem.qid, question=sem.question.rstrip("?") + f" ({abbr})?"), "clinical term"),
        (questions + [questions[0]], "duplicate qid"),
    ]
    for bad, msg in cases:
        with pytest.raises(ValueError, match=msg):
            validate_questions(bad, docs, chunks)


def test_reason_keywords():
    assert reason_keywords("hyperkalemia (K 5.7); stopped")[:2] == ["K 5.7", "hyperkalemia"]
    kws = reason_keywords("persistent dry cough (likely ACE inhibitor-related); switched to losartan")
    assert "dry cough" in kws and "ACE inhibitor" in kws
    assert "day course" in reason_keywords("completed 7-day course")
    assert reason_keywords("dose increased to 100 mg after AFib with RVR admission") == ["AFib with RVR"]
    for sw in V.SWITCHES:
        reason = sw.reason.replace("{lab}", "X 1.0")
        assert any(answer_correct(reason, [k]) for k in reason_keywords(reason)), reason


# ------------------------------------------------------------------ io


def test_write_read_roundtrip(tmp_path, questions):
    path = tmp_path / "sub" / "eval_questions.jsonl"
    write_questions(questions, path)
    back = read_questions(path)
    assert [q.to_dict() for q in back] == [q.to_dict() for q in questions]
    assert all(isinstance(q, EvalQuestion) for q in back)
    with pytest.raises(FileNotFoundError):
        read_questions(tmp_path / "missing.jsonl")


@pytest.mark.skipif(not (DATA_DIR / "eval_questions.jsonl").exists(), reason="eval_questions.jsonl not built")
def test_written_question_file_is_valid():
    qs = read_questions(DATA_DIR / "eval_questions.jsonl")
    docs = load_documents(DATA_DIR)
    validate_questions(qs, docs)
    counts = Counter(q.category for q in qs)
    n_patients = len({d.patient_id for d in docs})
    assert counts == Counter({c: n * n_patients for c, n in EXPECTED_PER_PATIENT.items()})
