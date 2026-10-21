"""Tests for ehr_rag.ingest.chunking using hand-made synthetic Documents."""
from __future__ import annotations

import re

import pytest

from ehr_rag.ingest.chunking import (
    SECTION_HEADER_RE,
    chunk_document,
    chunk_documents,
    header_for,
    split_sentences,
)
from ehr_rag.types import Chunk, Document, chunk_id_for

NOTE_TEXT = (
    "CHIEF COMPLAINT:\n"
    "SOB and DOE x 3 days.\n\n"
    "HISTORY OF PRESENT ILLNESS:\n"
    "68 yo M with T2DM, HTN, CKD3 and s/p MI presents with worsening SOB. Seen by Dr. Quill on 2025-03-10. "
    "A1c was 7.2 % on 2025-02-01. He takes metformin 500 mg. b.i.d. with meals. Reports orthopnea. "
    "Denies chest pain. Weight up 3 kg since last visit. Furosemide 40 mg PO q.d. was started.\n\n"
    "REVIEW OF SYSTEMS:\n\n"
    "HOSPITAL COURSE:\n"
    "Diuresed with IV furosemide. Net negative 4.5 L. Cr peaked at 1.8 and returned to 1.4.\n\n"
    "ASSESSMENT AND PLAN:\n"
    "1. Acute on chronic HFrEF exacerbation, improved.\n"
    "2. T2DM, continue metformin 500 mg PO BID.\n"
)


def make_note(text: str = NOTE_TEXT, **meta) -> Document:
    metadata = {"synthetic": True, "note_type": "discharge_summary", "author": "Dr. Avery Quill (synthetic)"}
    metadata.update(meta)
    return Document(
        doc_id="P001-N02",
        patient_id="P001",
        doc_type="clinical_note",
        date="2025-03-14",
        title="Discharge Summary",
        text=text,
        metadata=metadata,
    )


def make_med_list(n: int = 30) -> Document:
    lines = ["Medication list for patient P001 as of 2026-06-30:"]
    for i in range(n):
        lines.append(
            f"- drug{i:02d} {100 + i} mg PO BID — started 2024-05-{(i % 28) + 1:02d} — for T2DM (active)"
        )
    return Document(
        doc_id="P001-medications",
        patient_id="P001",
        doc_type="medication_list",
        date="2026-06-30",
        title="Medication List",
        text="\n".join(lines),
        metadata={"synthetic": True},
    )


def make_lab_panel() -> Document:
    text = (
        "- Hemoglobin A1c (LOINC 4548-4): 7.2 % (ref 4.0-5.6) H\n"
        "- Creatinine (LOINC 2160-0): 1.4 mg/dL (ref 0.6-1.3) H\n"
        "- Potassium (LOINC 2823-3): 4.1 mmol/L (ref 3.5-5.1)"
    )
    return Document(
        doc_id="P001-labs-2025-03-14",
        patient_id="P001",
        doc_type="lab_panel",
        date="2025-03-14",
        title="Lab Panel 2025-03-14",
        text=text,
        metadata={"synthetic": True, "panel": "BMP"},
    )


def body(chunk: Chunk) -> str:
    header, _, rest = chunk.text.partition("\n")
    assert header.startswith("[Patient ") and header.endswith("]")
    return rest


# --------------------------------------------------------------------------- header


def test_header_format_exact():
    doc = make_note()
    assert header_for(doc, "Hospital Course") == (
        "[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]"
    )
    assert header_for(doc, "HOSPITAL COURSE") == (
        "[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]"
    )
    assert header_for(doc, None) == "[Patient P001 | Discharge Summary | 2025-03-14]"
    assert header_for(doc, "ASSESSMENT AND PLAN").endswith("Section: Assessment and Plan]")
    assert header_for(doc, "HPI").endswith("Section: HPI]")


def test_section_header_regex():
    for line in ["CHIEF COMPLAINT:", "ASSESSMENT AND PLAN:", "HISTORY OF PRESENT ILLNESS (HPI):", "IMPRESSION:"]:
        assert SECTION_HEADER_RE.match(line)
    for line in ["Chief complaint:", "- A1C:", "PLAN: continue metformin", "A:"]:
        assert not SECTION_HEADER_RE.match(line)


# --------------------------------------------------------------------------- sentences


def test_sentence_split_keeps_decimals_abbreviations_dates():
    text = (
        "Seen by Dr. Quill on 2025-03-10. A1c was 7.2 % on 2025-02-01. "
        "He takes metformin 500 mg. b.i.d. with meals. Aspirin 81 mg q.d. daily. Reports orthopnea."
    )
    sentences = [s for s, _ in split_sentences(text)]
    assert sentences == [
        "Seen by Dr. Quill on 2025-03-10.",
        "A1c was 7.2 % on 2025-02-01.",
        "He takes metformin 500 mg. b.i.d. with meals.",
        "Aspirin 81 mg q.d. daily.",
        "Reports orthopnea.",
    ]


def test_sentence_split_more_negatives():
    for text in [
        "Temp 37.2 C and Cr 1.4 mg/dL.",
        "Take 1 tab p.o. q.d. as directed.",
        "Follow up with Dr. Lee e.g. Monday.",
        "Visit on 03.14.2025 went well.",
    ]:
        assert len(split_sentences(text)) == 1, text


def test_sentence_split_lines_are_boundaries():
    units = split_sentences("1. HFrEF, improved.\n2. T2DM, continue metformin.")
    assert [u[0] for u in units] == ["1. HFrEF, improved.", "2. T2DM, continue metformin."]
    assert all(newline for _, newline in units)


# --------------------------------------------------------------------------- notes


def test_note_sections_and_empty_sections_skipped():
    chunks = chunk_document(make_note(), max_chars=2000)
    sections = [c.section for c in chunks]
    assert sections == ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HOSPITAL COURSE", "ASSESSMENT AND PLAN"]
    assert "REVIEW OF SYSTEMS" not in sections
    assert chunks[2].text.startswith(
        "[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]\n"
    )
    assert body(chunks[0]) == "SOB and DOE x 3 days."
    # list items keep their line breaks
    assert body(chunks[3]) == (
        "1. Acute on chronic HFrEF exacerbation, improved.\n2. T2DM, continue metformin 500 mg PO BID."
    )


def test_long_section_split_with_overlap_within_section_only():
    max_chars = 200
    chunks = chunk_document(make_note(), max_chars=max_chars, overlap_sentences=1)
    hpi = [c for c in chunks if c.section == "HISTORY OF PRESENT ILLNESS"]
    assert len(hpi) >= 3
    for c in chunks:
        assert len(c.text) <= max_chars, c.text
    for prev, nxt in zip(hpi, hpi[1:]):
        prev_sents = [s for s, _ in split_sentences(body(prev))]
        next_sents = [s for s, _ in split_sentences(body(nxt))]
        assert next_sents[0] == prev_sents[-1]  # one sentence carried over
    # No sentence split mid-way: every HPI sentence appears intact in some chunk.
    hpi_text = NOTE_TEXT.split("HISTORY OF PRESENT ILLNESS:\n")[1].split("\n\n")[0]
    for sent, _ in split_sentences(hpi_text):
        assert any(sent in body(c) for c in hpi), sent
    # First chunk of the next section does not start with the last HPI sentence.
    course = [c for c in chunks if c.section == "HOSPITAL COURSE"]
    last_hpi_sentence = split_sentences(body(hpi[-1]))[-1][0]
    assert last_hpi_sentence not in body(course[0])


def test_overlap_zero_has_no_repeats():
    chunks = chunk_document(make_note(), max_chars=200, overlap_sentences=0)
    hpi = [c for c in chunks if c.section == "HISTORY OF PRESENT ILLNESS"]
    all_sents = [s for c in hpi for s, _ in split_sentences(body(c))]
    assert len(all_sents) == len(set(all_sents))


def test_oversized_sentence_becomes_own_chunk():
    long_sentence = "Patient reports " + "intermittent dyspnea " * 30 + "today."
    note = make_note("HPI:\nShort intro. " + long_sentence + " Short outro.\n")
    chunks = chunk_document(note, max_chars=150)
    bodies = [body(c) for c in chunks]
    assert long_sentence in bodies
    assert any(b.startswith("Short intro.") for b in bodies)
    assert any(b.endswith("Short outro.") for b in bodies)


def test_note_without_headers_and_preamble():
    note = make_note("Brief synthetic telehealth note. Patient doing well.\n\nPLAN:\nContinue meds.")
    chunks = chunk_document(note)
    assert [c.section for c in chunks] == [None, "PLAN"]
    assert chunks[0].text.startswith("[Patient P001 | Discharge Summary | 2025-03-14]\n")


# --------------------------------------------------------------------------- structured docs


def test_structured_doc_groups_whole_lines_and_repeats_title():
    doc = make_med_list(30)
    chunks = chunk_document(doc, max_chars=400)
    assert len(chunks) > 1
    lines = doc.text.splitlines()
    title, records = lines[0], lines[1:]
    seen_records = []
    for c in chunks:
        assert c.section is None
        assert c.text.startswith("[Patient P001 | Medication List | 2026-06-30]\n")
        assert len(c.text) <= 400
        body_lines = body(c).split("\n")
        assert body_lines[0] == title  # title repeated in every chunk
        for line in body_lines[1:]:
            assert line in records  # never split
            seen_records.append(line)
    assert seen_records == records  # all records once, in order (no overlap for structured docs)


def test_structured_doc_without_title_line():
    doc = make_lab_panel()
    chunks = chunk_document(doc, max_chars=150)
    lines = doc.text.splitlines()
    assert [body(c) for c in chunks] == lines  # each line alone: none fits with another
    chunks = chunk_document(doc, max_chars=900)
    assert len(chunks) == 1 and body(chunks[0]) == doc.text
    assert chunks[0].text.startswith("[Patient P001 | Lab Panel 2025-03-14 | 2025-03-14]\n")


def test_long_line_becomes_own_chunk():
    long_line = "- " + "x" * 500
    doc = make_lab_panel()
    doc.text = "- short line one\n" + long_line + "\n- short line two"
    chunks = chunk_document(doc, max_chars=200)
    assert [body(c) for c in chunks] == ["- short line one", long_line, "- short line two"]


# --------------------------------------------------------------------------- ids & metadata


def test_chunk_ids_unique_and_stable():
    docs = [make_note(), make_med_list(30), make_lab_panel()]
    a = chunk_documents(docs, max_chars=300)
    b = chunk_documents(docs, max_chars=300)
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]
    note_chunks = [c for c in a if c.doc_id == "P001-N02"]
    assert [c.chunk_id for c in note_chunks] == [chunk_id_for("P001-N02", i) for i in range(len(note_chunks))]


def test_duplicate_doc_ids_rejected():
    with pytest.raises(ValueError, match="duplicate doc_id"):
        chunk_documents([make_lab_panel(), make_lab_panel()])


def test_metadata_copied_and_note_type_added():
    note = make_note()
    chunks = chunk_document(note)
    for c in chunks:
        assert c.metadata["synthetic"] is True
        assert c.metadata["note_type"] == "discharge_summary"
        assert c.metadata["author"] == "Dr. Avery Quill (synthetic)"
        assert (c.doc_id, c.patient_id, c.doc_type, c.date, c.title) == (
            "P001-N02", "P001", "clinical_note", "2025-03-14", "Discharge Summary"
        )
    chunks[0].metadata["synthetic"] = False  # copies, not shared references
    assert note.metadata["synthetic"] is True and chunks[1].metadata["synthetic"] is True

    # note_type inferred from the title when the doc metadata lacks it
    meta = {"synthetic": True}
    inferred = Document("P001-N03", "P001", "clinical_note", "2025-04-01", "Progress Note", "PLAN:\nOK.", meta)
    assert chunk_document(inferred)[0].metadata["note_type"] == "progress_note"

    labs = chunk_document(make_lab_panel())
    assert labs[0].metadata == {"synthetic": True, "panel": "BMP"}  # no note_type for structured docs


def test_non_synthetic_flag_propagates_unchanged():
    doc = make_lab_panel()
    doc.metadata = {}
    assert "synthetic" not in chunk_document(doc)[0].metadata


def test_empty_document_yields_no_chunks():
    doc = make_lab_panel()
    doc.text = "  \n\n"
    assert chunk_document(doc) == []
    assert chunk_document(make_note("CHIEF COMPLAINT:\n\nPLAN:\n")) == []


def test_chunks_have_no_section_header_lines_in_body():
    for c in chunk_document(make_note(), max_chars=250):
        assert not any(re.match(SECTION_HEADER_RE, line) for line in body(c).splitlines())


def test_sentence_split_positive_boundaries():
    # positive control: real boundaries after numbers/values are still split
    assert [s for s, _ in split_sentences("Cr was 1.4. Next visit soon.")] == ["Cr was 1.4.", "Next visit soon."]
    assert [s for s, _ in split_sentences("BP 132/84. HR 72.")] == ["BP 132/84.", "HR 72."]
    assert [s for s, _ in split_sentences("Improving! Discharged home? Yes.")] == [
        "Improving!", "Discharged home?", "Yes."
    ]
