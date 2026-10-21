"""Tests for ehr_rag.ingest.phi_guard: true positives, true negatives, and the synthetic gate."""
from __future__ import annotations

import pytest

from ehr_rag.ingest.phi_guard import (
    NonSyntheticDataError,
    PHIFinding,
    assert_synthetic,
    check_documents,
    scan_phi,
)
from ehr_rag.types import Chunk, Document, RetrievedChunk


def kinds(text: str) -> list[str]:
    return [f.kind for f in scan_phi(text)]


# --------------------------------------------------------------------------- true positives


@pytest.mark.parametrize(
    "text, kind, match",
    [
        ("SSN 123-45-6789 on file", "ssn", "123-45-6789"),
        ("ssn: 123 45 6789", "ssn", "123 45 6789"),
        ("Call (713) 555-0142 tomorrow", "phone", "(713) 555-0142"),
        ("cell 713-555-0142.", "phone", "713-555-0142"),
        ("cell 713.555.0142", "phone", "713.555.0142"),
        ("call +1 713 555 0142", "phone", "+1 713 555 0142"),
        ("call 1-800-555-0199 now", "phone", "1-800-555-0199"),
        ("email jane.doe@example.org please", "email", "jane.doe@example.org"),
        ("see https://portal.example.com/chart?id=7.", "url", "https://portal.example.com/chart?id=7"),
        ("see www.example.com", "url", "www.example.com"),
        ("Lives at 123 Main St with daughter", "street_address", "123 Main St"),
        ("home: 4500 Oak Hollow Drive.", "street_address", "4500 Oak Hollow Drive"),
        ("addr 12 W 5th Ave", "street_address", "12 W 5th Ave"),
        ("MRN: 00482913", "mrn", "MRN: 00482913"),
        ("mrn 1234567", "mrn", "mrn 1234567"),
        ("Medical Record Number: A-1234567", "mrn", "Medical Record Number: A-1234567"),
        ("id HOSP-12345678 transferred", "mrn", "HOSP-12345678"),
    ],
)
def test_true_positives(text, kind, match):
    findings = scan_phi(text)
    assert [(f.kind, f.match) for f in findings] == [(kind, match)]
    f = findings[0]
    assert text[f.start:f.end] == f.match


def test_multiple_findings_ordered_and_nested_dropped():
    text = "Pt email a@b.com, phone 713-555-0142, link http://x.example.com/call/713-555-0142"
    findings = scan_phi(text)
    assert [f.kind for f in findings] == ["email", "phone", "url"]
    assert findings == sorted(findings, key=lambda f: f.start)
    assert isinstance(findings[0], PHIFinding)


# --------------------------------------------------------------------------- true negatives


@pytest.mark.parametrize(
    "text",
    [
        "Seen on 2025-03-14; next visit 2026-06-30.",
        "A1c 7.2 % (ref 4.0-5.6) H; Cr 1.4 mg/dL; K 4.1 mmol/L; Plt 250000; Na 140.",
        "metformin 1000 mg PO BID; lisinopril 20 mg PO daily; insulin glargine 18 units qHS",
        "BP 132/84, HR 72, RR 16, SpO2 97%, Temp 36.8 C, Wt 92.5 kg, BMI 31.2",
        "Hemoglobin A1c (LOINC 4548-4); creatinine (LOINC 2160-0); LDL (LOINC 13457-7)",
        "T2DM (E11.9), HTN (I10), CKD3 (N18.30), CHF (I50.22), s/p MI (I25.2)",
        "MRN SYN-100001; patient P001; encounter P001-E01; med P001-M03; note P001-N02",
        "chunk P001-labs-2025-03-14::000 and doc P001-vitals-2025-03-14",
        "Follow up in 3 months with Dr. Quill. Seen by 2 Dr. residents.",
        "Walked 200 ft without SOB. Weight 210 lbs. 1000 Units heparin.",
        "[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]",
        "Ref range 70-99 mg/dL; eGFR 45 mL/min/1.73m2; dose 0.125 mg; 1 Tab PO.",
    ],
)
def test_true_negatives(text):
    assert scan_phi(text) == []


# --------------------------------------------------------------------------- documents


def _doc(doc_id: str, text: str, synthetic=True) -> Document:
    meta = {"synthetic": synthetic} if synthetic is not None else {}
    return Document(doc_id, "P001", "clinical_note", "2025-03-14", "Progress Note", text, meta)


def test_check_documents_only_reports_docs_with_findings():
    docs = [
        _doc("P001-N01", "HPI:\nSOB. A1c 7.2 on 2025-03-14. MRN SYN-100001."),
        _doc("P001-N02", "SOCIAL HISTORY:\nLives at 123 Main St. Phone 713-555-0142."),
    ]
    result = check_documents(docs)
    assert list(result) == ["P001-N02"]
    assert [f.kind for f in result["P001-N02"]] == ["street_address", "phone"]


# --------------------------------------------------------------------------- synthetic gate


def _chunk(chunk_id: str, metadata: dict) -> Chunk:
    return Chunk(chunk_id, chunk_id.split("::")[0], "P001", "lab_panel", "2025-03-14", "Lab Panel", None, "x",
                 metadata)


def test_assert_synthetic_passes_for_flagged_items():
    assert_synthetic([_doc("P001-N01", "x"), _chunk("P001-labs::000", {"synthetic": True})])
    assert_synthetic([])


def test_assert_synthetic_lists_offending_ids():
    items = [
        _doc("P001-N01", "x"),
        _doc("P001-N02", "x", synthetic=None),         # missing flag
        _doc("P001-N03", "x", synthetic=False),
        _chunk("P002-labs::000", {"synthetic": "true"}),  # string is not accepted
        _chunk("P002-labs::001", {"synthetic": True}),
    ]
    with pytest.raises(NonSyntheticDataError) as exc:
        assert_synthetic(items)
    assert exc.value.offending_ids == ["P001-N02", "P001-N03", "P002-labs::000"]
    message = str(exc.value)
    for oid in ["P001-N02", "P001-N03", "P002-labs::000"]:
        assert oid in message
    assert "P001-N01" not in message
    assert isinstance(exc.value, ValueError)


def test_assert_synthetic_accepts_retrieved_chunks_and_generators():
    good = RetrievedChunk(_chunk("P001-labs::000", {"synthetic": True}), 0.9, 1, "dense")
    bad = RetrievedChunk(_chunk("P001-labs::001", {}), 0.8, 2, "dense")
    assert_synthetic(iter([good]))
    with pytest.raises(NonSyntheticDataError, match="P001-labs::001"):
        assert_synthetic(x for x in [good, bad])


def test_assert_synthetic_truncates_long_lists():
    items = [_chunk(f"P001-d::{i:03d}", {}) for i in range(25)]
    with pytest.raises(NonSyntheticDataError) as exc:
        assert_synthetic(items)
    assert len(exc.value.offending_ids) == 25
    assert "(+5 more)" in str(exc.value)
