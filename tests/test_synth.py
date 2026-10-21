"""Tests for the synthetic EHR generator (ehr_rag.synth)."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import date

import pytest

from ehr_rag import types as T
from ehr_rag.synth import vocab as V
from ehr_rag.synth.generator import (
    REFERENCE_DATE,
    TABLES,
    SyntheticDataset,
    format_lab_value,
    generate_dataset,
    is_active_on,
    lab_flag,
    load_documents,
    render_medication,
    render_note_text,
)

SCHEMA: dict[str, set[str]] = {
    "patients": {"patient_id", "mrn", "name", "sex", "birth_date", "age", "pcp", "synthetic"},
    "encounters": {"patient_id", "encounter_id", "date", "encounter_type", "reason", "provider", "discharge_date",
                   "synthetic"},
    "conditions": {"patient_id", "condition_key", "display", "icd10", "abbreviation", "onset_date", "status",
                   "synthetic"},
    "medications": {"patient_id", "med_id", "name", "dose", "route", "frequency", "start_date", "end_date",
                    "status", "indication", "discontinue_reason", "synthetic"},
    "allergies": {"patient_id", "substance", "reaction", "severity", "synthetic"},
    "labs": {"patient_id", "date", "encounter_id", "panel", "test_key", "name", "loinc", "value", "unit", "ref_low",
             "ref_high", "flag", "synthetic"},
    "vitals": {"patient_id", "date", "encounter_id", "systolic", "diastolic", "heart_rate", "resp_rate", "temp_c",
               "spo2", "weight_kg", "bmi", "synthetic"},
    "notes": {"patient_id", "note_id", "encounter_id", "date", "note_type", "author", "title", "sections",
              "mentions", "synthetic"},
}
HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 /&(),'-]+:$")
MED_SECTIONS = ("MEDICATIONS", "DISCHARGE MEDICATIONS")


def _word_re(term: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


MED_NAME_RE = {name: _word_re(name) for name in V.MEDICATIONS}
LAB_QUOTE_RE = {key: re.compile(rf"(?<![\w-]){re.escape(t.short)} (-?\d+(?:\.\d+)?)(?![\d.]*\d)")
                for key, t in V.LAB_TESTS.items()}


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return generate_dataset(n_patients=30, seed=42)


@pytest.fixture(scope="module")
def docs(ds):
    return ds.to_documents()


def _by_pid(rows):
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["patient_id"], []).append(r)
    return out


def _dump(dataset: SyntheticDataset) -> str:
    return json.dumps({t: getattr(dataset, t) for t in TABLES}, sort_keys=True)


# --------------------------------------------------------------------------- determinism / schema

def test_determinism_same_and_different_seed():
    a, b = generate_dataset(8, seed=7), generate_dataset(8, seed=7)
    assert _dump(a) == _dump(b)
    assert [d.to_dict() for d in a.to_documents()] == [d.to_dict() for d in b.to_documents()]
    assert _dump(generate_dataset(8, seed=8)) != _dump(a)


def test_schema_and_synthetic_flags(ds, docs):
    for table, keys in SCHEMA.items():
        rows = getattr(ds, table)
        assert rows, table
        for rec in rows:
            assert keys <= set(rec), (table, keys - set(rec))
            assert rec["synthetic"] is True
    for d in docs:
        assert d.metadata.get("synthetic") is True
        assert d.doc_type in T.DOC_TYPES
    for n in ds.notes:
        assert n["note_type"] in T.NOTE_TYPES
        assert set(n["mentions"]) == {"conditions", "medications", "labs"}
    for e in ds.encounters:
        assert e["encounter_type"] in {"outpatient", "ed", "inpatient", "telehealth"}
        assert (e["discharge_date"] is not None) == (e["encounter_type"] == "inpatient")
    for m in ds.medications:
        assert m["status"] in {"active", "discontinued"}
        assert (m["end_date"] is None) == (m["status"] == "active") == (m["discontinue_reason"] is None)
        assert m["name"] == m["name"].lower() and m["indication"] in V.CONDITIONS
    for a in ds.allergies:
        assert a["severity"] in {"mild", "moderate", "severe"}
    for c in ds.conditions:
        assert c["status"] in {"active", "resolved"}


def test_id_formats(ds):
    pids = [p["patient_id"] for p in ds.patients]
    assert pids == [f"P{i:03d}" for i in range(1, 31)]
    assert all(re.fullmatch(r"P\d{3}-E\d{2}", e["encounter_id"]) for e in ds.encounters)
    assert all(re.fullmatch(r"P\d{3}-M\d{2}", m["med_id"]) for m in ds.medications)
    assert all(re.fullmatch(r"P\d{3}-N\d{2}", n["note_id"]) for n in ds.notes)
    for table in ("encounters", "medications", "notes"):
        key = {"encounters": "encounter_id", "medications": "med_id", "notes": "note_id"}[table]
        ids = [r[key] for r in getattr(ds, table)]
        assert len(ids) == len(set(ids))
        assert all(r[key].startswith(r["patient_id"] + "-") for r in getattr(ds, table))


# --------------------------------------------------------------------------- documents

def test_doc_ids_match_helpers_and_are_unique(ds, docs):
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))
    got = set(ids)
    expected: set[str] = set()
    for p in ds.patients:
        pid = p["patient_id"]
        expected |= {T.demographics_doc_id(pid), T.problems_doc_id(pid), T.medications_doc_id(pid),
                     T.allergies_doc_id(pid)}
    expected |= {T.lab_panel_doc_id(r["patient_id"], r["date"]) for r in ds.labs}
    expected |= {T.vitals_doc_id(r["patient_id"], r["date"]) for r in ds.vitals}
    expected |= {n["note_id"] for n in ds.notes}
    assert got == expected
    type_of = {d.doc_id: d.doc_type for d in docs}
    for n in ds.notes:
        assert type_of[n["note_id"]] == "clinical_note"
    for d in docs:
        assert d.patient_id == d.metadata["patient_id"]
        assert d.doc_id.startswith(d.patient_id + "-")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.date)


def test_structured_doc_line_formats(ds, docs):
    med_line = re.compile(r"^- [a-z][a-z -]+ \S.* — started \d{4}-\d{2}-\d{2}( — stopped \d{4}-\d{2}-\d{2})? — "
                          r"for \S.* \((active|discontinued: .+)\)$")
    lab_line = re.compile(r"^- .+: -?\d+(\.\d+)?( \S+)? \(ref [^)]+\)( \[[HL]\])?$")
    for d in docs:
        lines = d.text.splitlines()
        if d.doc_type == "medication_list":
            assert all(med_line.match(x) for x in lines[1:]), d.text
            assert len(lines) - 1 == sum(1 for m in ds.medications if m["patient_id"] == d.patient_id)
        elif d.doc_type == "lab_panel":
            assert all(lab_line.match(x) for x in lines[1:]), d.text
        elif d.doc_type == "allergy_list" and not any(a["patient_id"] == d.patient_id for a in ds.allergies):
            assert "No known drug allergies (NKDA)." in d.text


def test_note_documents_render_sections(ds, docs):
    texts = {d.doc_id: d.text for d in docs}
    for n in ds.notes:
        assert texts[n["note_id"]] == render_note_text(n)
        for name, body in n["sections"]:
            assert HEADER_RE.match(f"{name}:"), name
            assert body.strip()
        header_lines = [x for x in texts[n["note_id"]].splitlines() if HEADER_RE.match(x)]
        assert header_lines == [f"{name}:" for name, _ in n["sections"]]


# --------------------------------------------------------------------------- per-patient minimums

def test_per_patient_minimums(ds):
    enc, labs, meds, notes, conds = (_by_pid(ds.encounters), _by_pid(ds.labs), _by_pid(ds.medications),
                                     _by_pid(ds.notes), _by_pid(ds.conditions))
    for p in ds.patients:
        pid = p["patient_id"]
        dates = sorted(e["date"] for e in enc[pid])
        assert len(dates) >= 3
        span = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days
        assert span >= 540, (pid, span)
        assert dates[-1] < REFERENCE_DATE.isoformat()
        assert all(e["discharge_date"] is None or e["discharge_date"] < REFERENCE_DATE.isoformat()
                   for e in enc[pid])
        lab_dates = {r["date"] for r in labs[pid]}
        assert len(lab_dates) >= 2
        per_test = Counter((r["test_key"]) for r in labs[pid])
        assert max(per_test.values()) >= 2          # at least one trend
        disc = [m for m in meds[pid] if m["status"] == "discontinued"]
        assert disc and all(m["discontinue_reason"] for m in disc)
        assert any("course" not in m["discontinue_reason"] for m in disc)   # a real switch/stop, not only courses
        assert len(notes[pid]) >= 2
        chronic = [c for c in conds[pid] if V.CONDITIONS[c["condition_key"]].chronic]
        assert V.MIN_CHRONIC_CONDITIONS <= len(chronic) <= V.MAX_CHRONIC_CONDITIONS
        assert all(c["onset_date"] <= dates[-1] for c in conds[pid])
        assert p["mrn"].startswith("SYN-")
        assert p["age"] == (REFERENCE_DATE.year - int(p["birth_date"][:4])
                            - ((REFERENCE_DATE.month, REFERENCE_DATE.day) <
                               (int(p["birth_date"][5:7]), int(p["birth_date"][8:]))))


def test_discharge_summaries_only_for_inpatient(ds):
    enc = {e["encounter_id"]: e for e in ds.encounters}
    for n in ds.notes:
        e = enc[n["encounter_id"]]
        assert n["patient_id"] == e["patient_id"]
        if n["note_type"] == "discharge_summary":
            assert e["encounter_type"] == "inpatient" and n["date"] == e["discharge_date"]
        elif e["encounter_type"] != "inpatient":
            assert n["date"] == e["date"]
        else:
            assert e["date"] <= n["date"] <= e["discharge_date"]
    inpatient = {e["encounter_id"] for e in ds.encounters if e["encounter_type"] == "inpatient"}
    assert inpatient == {n["encounter_id"] for n in ds.notes if n["note_type"] == "discharge_summary"}


def test_note_lengths_and_variety(ds):
    for n in ds.notes:
        words = len(render_note_text(n).split())
        assert 150 <= words <= 450, (n["note_id"], n["note_type"], words)
    assert {n["note_type"] for n in ds.notes} == set(T.NOTE_TYPES)
    hpis = [dict(n["sections"])["HISTORY OF PRESENT ILLNESS"] for n in ds.notes if n["note_type"] == "progress_note"]
    assert len(set(hpis)) == len(hpis)
    openers = Counter(h.split(" with ")[0].split()[-1] for h in hpis)
    assert len(openers) >= 3


# --------------------------------------------------------------------------- language / PHI

def test_lay_terms_never_appear(ds, docs):
    lay = V.all_lay_terms()
    assert len(lay) >= 28
    patterns = [_word_re(t) for t in lay]
    texts = [d.text for d in docs] + [d.title for d in docs]
    for table in TABLES:
        for rec in getattr(ds, table):
            texts.append(json.dumps(rec))
    for spec in V.CONDITIONS.values():
        texts += [spec.display, spec.abbreviation, *spec.note_terms]
    for text in texts:
        for pat in patterns:
            assert not pat.search(text), (pat.pattern, text[:200])


def test_vocab_coverage():
    chronic = [c for c in V.CONDITIONS.values() if c.chronic]
    assert len(chronic) >= 14
    for c in V.CONDITIONS.values():
        assert len(c.lay_terms) >= 2 and c.icd10 and c.abbreviation in c.note_terms
        assert all(m in V.MEDICATIONS for m in c.meds) and all(t in V.LAB_TESTS for t in c.labs)
    assert len(V.MEDICATIONS) >= 25
    assert len(V.ALLERGENS) >= 12
    required = {"hba1c", "glucose", "creatinine", "egfr", "bun", "potassium", "sodium", "ldl", "hdl",
                "triglycerides", "total_cholesterol", "tsh", "bnp", "hemoglobin", "wbc", "platelets", "alt", "ast",
                "inr"}
    assert required <= set(V.LAB_TESTS)
    for s in V.SWITCHES:
        assert s.from_med[0] in V.MEDICATIONS and s.from_med[1] in V.MEDICATIONS[s.from_med[0]].doses
        if s.to_med:
            assert s.to_med[1] in V.MEDICATIONS[s.to_med[0]].doses
    for cond, slots in V.REGIMENS.items():
        for slot in slots:
            for opt in slot:
                if opt:
                    assert opt[1] in V.MEDICATIONS[opt[0]].doses, (cond, opt)


PHI_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\(\d{3}\)\s*|\d{3}[-. ])\d{3}[-. ]\d{4}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"https?://|www\.", re.IGNORECASE),
    "address": re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|"
                          r"Boulevard|Blvd|Court|Ct|Way)\b\.?"),
    "mrn": re.compile(r"\bMRN[:#\s]*(?!SYN-)[A-Z0-9-]{4,}", re.IGNORECASE),
}


def test_no_phi_like_patterns(ds, docs):
    texts = [d.text for d in docs] + [json.dumps(r) for t in TABLES for r in getattr(ds, t)]
    for text in texts:
        for kind, pat in PHI_PATTERNS.items():
            assert not pat.search(text), (kind, pat.search(text).group(0), text[:120])
    firsts = set(V.FEMALE_FIRST_NAMES) | set(V.MALE_FIRST_NAMES)
    for p in ds.patients:
        first, last = p["name"].split(" ")
        assert first in firsts and last in V.LAST_NAMES
        assert re.fullmatch(r"SYN-\d{6}", p["mrn"])
    providers = {x for group in V.PROVIDERS.values() for x in group}
    assert all(x.endswith("(synthetic)") for x in providers)
    assert {p["pcp"] for p in ds.patients} <= providers
    assert {e["provider"] for e in ds.encounters} <= providers
    assert {n["author"] for n in ds.notes} <= providers
    assert len({p["name"] for p in ds.patients}) == len(ds.patients)


# --------------------------------------------------------------------------- note / structured consistency

def test_note_medication_sections_are_active_on_note_date(ds):
    meds = _by_pid(ds.medications)
    for n in ds.notes:
        active = [m for m in meds[n["patient_id"]] if is_active_on(m, n["date"])]
        for name, body in n["sections"]:
            if name not in MED_SECTIONS:
                continue
            for med_name, pat in MED_NAME_RE.items():
                if pat.search(body):
                    assert any(m["name"] == med_name for m in active), (n["note_id"], med_name, n["date"])
            for line in body.splitlines():
                if line.startswith("- "):
                    rendered = re.sub(r" \((NEW|CHANGED: [^)]*)\)$", "", line[2:])
                    assert rendered in {render_medication(m) for m in active}, (n["note_id"], line)


def test_note_lab_quotes_match_lab_table(ds):
    labs = _by_pid(ds.labs)
    for n in ds.notes:
        text = render_note_text(n)
        rows = {(r["test_key"], r["date"]): r for r in labs.get(n["patient_id"], [])}
        mentioned = [tuple(x) for x in n["mentions"]["labs"]]
        assert len(mentioned) == len(set(mentioned))
        for key in mentioned:
            assert key in rows, (n["note_id"], key)
            row = rows[key]
            assert row["date"] <= n["date"]
            quote = f"{V.LAB_TESTS[row['test_key']].short} {format_lab_value(row['test_key'], row['value'])}"
            assert re.search(rf"(?<![\w-]){re.escape(quote)}(?![\d.]*\d)", text), (n["note_id"], quote)
        for test_key, pat in LAB_QUOTE_RE.items():
            for match in pat.finditer(text):
                value = match.group(1)
                ok = [k for k in mentioned if k[0] == test_key
                      and format_lab_value(test_key, rows[k]["value"]) == value]
                assert ok, (n["note_id"], match.group(0))


def test_note_vitals_match_vitals_table(ds):
    vitals = {(v["patient_id"], v["date"]): v for v in ds.vitals}
    enc = {e["encounter_id"]: e for e in ds.encounters}
    for n in ds.notes:
        sections = dict(n["sections"])
        if "VITAL SIGNS" not in sections:
            continue
        v = vitals[(n["patient_id"], enc[n["encounter_id"]]["date"])]
        body = sections["VITAL SIGNS"]
        assert f"BP {v['systolic']}/{v['diastolic']}, HR {v['heart_rate']}, RR {v['resp_rate']}" in body
        assert f"SpO2 {v['spo2']}%" in body and f"BMI {v['bmi']:.1f}" in body


def test_note_mentions_exactly_reflect_text(ds):
    conds = {pid: [c["condition_key"] for c in rows] for pid, rows in _by_pid(ds.conditions).items()}
    meds = {m["med_id"]: m for m in ds.medications}
    for n in ds.notes:
        text = render_note_text(n)
        found = {k for k in V.CONDITIONS if any(_word_re(t).search(text) for t in V.CONDITIONS[k].note_terms)}
        assert found == set(n["mentions"]["conditions"]), (n["note_id"], found ^ set(n["mentions"]["conditions"]))
        assert found <= set(conds[n["patient_id"]])
        ids = n["mentions"]["medications"]
        assert len(ids) == len(set(ids))
        assert all(meds[i]["patient_id"] == n["patient_id"] for i in ids)
        assert all(meds[i]["start_date"] <= n["date"] for i in ids)
        # allergens (e.g. doxycycline) are not medication records: ignore the ALLERGIES section
        med_text = "\n".join(body for name, body in n["sections"] if name != "ALLERGIES")
        named = {name for name, pat in MED_NAME_RE.items() if pat.search(med_text)}
        assert named == {meds[i]["name"] for i in ids}, (n["note_id"], named ^ {meds[i]["name"] for i in ids})


def test_discontinue_reasons_supported_by_labs(ds):
    labs = {(r["patient_id"], r["test_key"], r["date"]): r for r in ds.labs}
    short_to_key = {t.short: k for k, t in V.LAB_TESTS.items()}
    n_checked = 0
    for m in ds.medications:
        reason = m["discontinue_reason"] or ""
        match = re.search(r"\((A1c|K|Na|LDL|TSH|INR) (\d+(?:\.\d+)?)\)", reason)
        if not match:
            continue
        key = short_to_key[match.group(1)]
        row = labs[(m["patient_id"], key, m["end_date"])]
        assert format_lab_value(key, row["value"]) == match.group(2)
        assert row["flag"] in {"H", "L"}
        n_checked += 1
    assert n_checked >= 3


# --------------------------------------------------------------------------- labs

def test_lab_values_formats_and_flags(ds):
    sex = {p["patient_id"]: p["sex"] for p in ds.patients}
    for r in ds.labs:
        spec = V.LAB_TESTS[r["test_key"]]
        assert r["loinc"] == spec.loinc and r["unit"] == spec.unit and r["panel"] == spec.panel
        assert isinstance(r["value"], float) and round(r["value"], spec.decimals) == r["value"]
        assert (r["ref_low"], r["ref_high"]) == V.ref_range(r["test_key"], sex[r["patient_id"]])
        assert r["flag"] == lab_flag(r["value"], r["ref_low"], r["ref_high"])
    keys = Counter((r["patient_id"], r["date"], r["test_key"]) for r in ds.labs)
    assert max(keys.values()) == 1
    assert format_lab_value("hba1c", 7.0) == "7.0" and format_lab_value("creatinine", 1.3) == "1.30"


def test_labs_reflect_conditions(ds):
    conds = {pid: {c["condition_key"] for c in rows} for pid, rows in _by_pid(ds.conditions).items()}
    meds = _by_pid(ds.medications)
    for r in ds.labs:
        c = conds[r["patient_id"]]
        if r["test_key"] == "egfr" and "ckd3" in c:
            assert r["value"] < 60
        if r["test_key"] == "egfr" and "ckd3" not in c and r["encounter_id"] not in _acute(ds):
            assert r["value"] >= 60
        if r["test_key"] == "hba1c" and "t2dm" in c:
            assert r["value"] >= 6.3
        if r["test_key"] == "hba1c" and "t2dm" not in c:
            assert r["value"] < 6.5
        if r["test_key"] == "bnp" and "hfref" in c:
            assert r["flag"] == "H"
        if r["test_key"] == "inr":   # INR only while on warfarin
            assert any(m["name"] == "warfarin" and m["start_date"] < r["date"] and
                       (m["end_date"] is None or m["end_date"] >= r["date"]) for m in meds[r["patient_id"]])
    # A1c trends down for at least some T2DM patients
    a1c = _by_pid([r for r in ds.labs if r["test_key"] == "hba1c"])
    falling = [pid for pid, rows in a1c.items() if "t2dm" in conds[pid] and len(rows) >= 3
               and rows[-1]["value"] < rows[0]["value"]]
    assert falling


def _acute(ds) -> set[str]:
    return {e["encounter_id"] for e in ds.encounters if e["encounter_type"] in ("inpatient", "ed")}


# --------------------------------------------------------------------------- allergies / distinctiveness

def test_allergies_and_nkda(ds):
    with_allergy = {a["patient_id"] for a in ds.allergies}
    nkda = len(ds.patients) - len(with_allergy)
    assert 0.2 <= nkda / len(ds.patients) <= 0.4
    excluded = {a.substance: set(a.excludes) for a in V.ALLERGENS}
    meds = _by_pid(ds.medications)
    for a in ds.allergies:
        assert not excluded[a["substance"]] & {m["name"] for m in meds[a["patient_id"]]}


def test_distinctive_facts(ds):
    items: dict[str, set[str]] = {}
    for m in ds.medications:
        items.setdefault(m["patient_id"], set()).add("med:" + m["name"])
    for a in ds.allergies:
        items.setdefault(a["patient_id"], set()).add("allergy:" + a["substance"])
    counts = Counter(x for v in items.values() for x in v)
    distinctive = [pid for pid, v in items.items() if any(counts[x] <= 3 for x in v)]
    assert len(distinctive) / len(ds.patients) >= 0.6


# --------------------------------------------------------------------------- io / speed

def test_write_load_roundtrip(ds, docs, tmp_path):
    ds.write(tmp_path)
    for table in TABLES:
        assert (tmp_path / f"{table}.jsonl").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["synthetic"] is True and manifest["seed"] == 42
    assert manifest["counts"] == {t: len(getattr(ds, t)) for t in TABLES}
    loaded = SyntheticDataset.load(tmp_path)
    assert loaded.seed == 42
    assert _dump(loaded) == _dump(ds)
    assert [d.to_dict() for d in load_documents(tmp_path)] == [d.to_dict() for d in docs]
    assert [d.to_dict() for d in loaded.to_documents()] == [d.to_dict() for d in docs]


def test_generation_is_fast():
    start = time.perf_counter()
    generate_dataset(30, seed=123).to_documents()
    assert time.perf_counter() - start < 10


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 99])
def test_other_seeds_generate_cleanly(seed, caplog):
    dataset = generate_dataset(30, seed=seed)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    for n in dataset.notes:
        assert 150 <= len(render_note_text(n).split()) <= 450
