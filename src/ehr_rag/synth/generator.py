"""Deterministic synthetic EHR generator.

``generate_dataset(n_patients, seed)`` builds raw tables (patients, encounters, conditions, medications,
allergies, labs, vitals, notes) plus retrievable ``Document`` objects. Every record carries
``"synthetic": True``; every random choice flows from ``random.Random`` seeded by ``seed`` (no global state, no
wall-clock time). Dates are relative to ``REFERENCE_DATE`` (2026-06-30).

Number / text formats (evaluation matches answer text against these)
---------------------------------------------------------------------
* Dates: ISO ``YYYY-MM-DD`` everywhere.
* Lab values are stored rounded to a fixed number of decimals per test and always rendered with exactly that
  many decimals (``format_lab_value``): HbA1c 1 (``7.6``), glucose 0 (``154``), creatinine 2 (``1.32``),
  eGFR 0, BUN 0, potassium 1 (``4.4``), sodium 0, LDL/HDL/TG/total cholesterol 0, TSH 2 (``3.45``), BNP 0,
  hemoglobin 1, WBC 1, platelets 0, ALT/AST 0, INR 1 (``2.4``).
* In notes a lab is always quoted as ``"<short> <value>"`` (e.g. ``A1c 7.6``, ``Cr 1.32``, ``K 5.7``,
  ``eGFR 52``, ``total cholesterol 182``); ``LabTestSpec.short`` holds the label.
* Lab panel document line: ``- Hemoglobin A1c (HbA1c): 7.6 % (ref 4.0-5.6) [H]`` (flag suffix only if abnormal;
  one-sided ranges render as ``ref >=60``).
* Medication line: ``- metformin 1000 mg PO BID — started 2024-05-02 — for T2DM (active)`` or
  ``- lisinopril 10 mg PO daily — started 2023-02-11 — stopped 2025-01-10 — for HTN (discontinued: <reason>)``.
* Vitals in notes: ``BP 138/84, HR 76, RR 16, T 36.8 C, SpO2 97% on RA, Wt 92.4 kg, BMI 31.2``.

A medication is *active on* date ``d`` when ``start_date <= d`` and (``end_date`` is None or ``d < end_date``).
A note's MEDICATIONS / DISCHARGE MEDICATIONS section lists only medications active on the note date (for a visit
with a medication change: the reconciled post-visit list). Labs and vitals reflect the regimen taken *before*
the visit. Note ``mentions``: ``conditions`` = patient conditions whose ``note_terms`` occur in the note text,
``medications`` = med_ids named in the note, ``labs`` = ``[test_key, date]`` for every lab value quoted.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from ehr_rag.synth import vocab as V
from ehr_rag.types import (
    Document,
    allergies_doc_id,
    demographics_doc_id,
    lab_panel_doc_id,
    medications_doc_id,
    problems_doc_id,
    vitals_doc_id,
)

logger = logging.getLogger(__name__)

REFERENCE_DATE: date = date(2026, 6, 30)
GENERATOR_VERSION = "1.0"
TABLES: tuple[str, ...] = ("patients", "encounters", "conditions", "medications", "allergies", "labs", "vitals",
                           "notes")
NOTE_MIN_WORDS = 150
NOTE_MAX_WORDS = 450

_STATIN_REDUCTION: dict[tuple[str, str], float] = {
    ("simvastatin", "20 mg"): 0.30, ("atorvastatin", "20 mg"): 0.43, ("atorvastatin", "40 mg"): 0.49,
    ("atorvastatin", "80 mg"): 0.55, ("rosuvastatin", "10 mg"): 0.46, ("rosuvastatin", "20 mg"): 0.52,
}
_NOTE_ORDER = {"radiology_report": 0, "ed_note": 1, "progress_note": 1, "consult_note": 2, "discharge_summary": 3}


# --------------------------------------------------------------------------- public helpers

def format_lab_value(test_key: str, value: float) -> str:
    """Render a lab value with the test's fixed number of decimals."""
    return f"{value:.{V.LAB_TESTS[test_key].decimals}f}"


def format_ref_range(test_key: str, ref_low: Optional[float], ref_high: Optional[float]) -> str:
    if ref_low is not None and ref_high is not None:
        return f"{format_lab_value(test_key, ref_low)}-{format_lab_value(test_key, ref_high)}"
    if ref_low is not None:
        return f">={format_lab_value(test_key, ref_low)}"
    if ref_high is not None:
        return f"<={format_lab_value(test_key, ref_high)}"
    return "n/a"


def lab_flag(value: float, ref_low: Optional[float], ref_high: Optional[float]) -> str:
    if ref_high is not None and value > ref_high:
        return "H"
    if ref_low is not None and value < ref_low:
        return "L"
    return ""


def is_active_on(med: dict, day: str) -> bool:
    """True if the medication record is active on ISO date ``day``."""
    return med["start_date"] <= day and (med["end_date"] is None or day < med["end_date"])


def render_medication(med: dict) -> str:
    """``"metformin 1000 mg PO BID"``."""
    return f"{med['name']} {med['dose']} {med['route']} {med['frequency']}"


def ckd_epi_2021(creatinine: float, age: int, sex: str) -> float:
    """Race-free CKD-EPI 2021 creatinine equation (mL/min/1.73m2)."""
    female = sex == "female"
    kappa, alpha = (0.7, -0.241) if female else (0.9, -0.302)
    ratio = creatinine / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age
    return egfr * (1.012 if female else 1.0)


def _creatinine_for_egfr(egfr: float, age: int, sex: str) -> float:
    lo, hi = 0.2, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ckd_epi_2021(mid, age, sex) > egfr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _iso(d: date) -> str:
    return d.isoformat()


def _date(s: str) -> date:
    return date.fromisoformat(s)


def _age_on(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- dataset

@dataclass
class SyntheticDataset:
    patients: list[dict] = field(default_factory=list)
    encounters: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    allergies: list[dict] = field(default_factory=list)
    labs: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    seed: int = 42

    # ------------------------------------------------------------------ documents
    def to_documents(self) -> list[Document]:
        """Render every patient's tables and notes as retrievable documents (see PLAN.md §3)."""
        by_pid: dict[str, dict[str, list[dict]]] = {}
        for table in TABLES:
            for rec in getattr(self, table):
                by_pid.setdefault(rec["patient_id"], {t: [] for t in TABLES})[table].append(rec)
        docs: list[Document] = []
        for patient in self.patients:
            docs.extend(_patient_documents(patient, by_pid[patient["patient_id"]]))
        return docs

    # ------------------------------------------------------------------ io
    def write(self, out_dir: Path) -> None:
        """Write ``<table>.jsonl`` for each table, ``documents.jsonl`` and ``manifest.json``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for table in TABLES:
            rows = getattr(self, table)
            counts[table] = len(rows)
            _write_jsonl(out_dir / f"{table}.jsonl", rows)
        docs = self.to_documents()
        _write_jsonl(out_dir / "documents.jsonl", [d.to_dict() for d in docs])
        manifest = {
            "synthetic": True,
            "generator": "ehr_rag.synth.generator",
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "n_patients": len(self.patients),
            "reference_date": _iso(REFERENCE_DATE),
            "counts": counts,
            "documents": len(docs),
            "doc_type_counts": dict(sorted(Counter(d.doc_type for d in docs).items())),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote synthetic dataset (%d patients, %d documents) to %s", len(self.patients), len(docs),
                    out_dir)

    @classmethod
    def load(cls, data_dir: Path) -> "SyntheticDataset":
        data_dir = Path(data_dir)
        manifest_path = data_dir / "manifest.json"
        seed = 0
        if manifest_path.exists():
            seed = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("seed", 0))
        tables = {t: _read_jsonl(data_dir / f"{t}.jsonl") for t in TABLES}
        return cls(seed=seed, **tables)


def load_documents(data_dir: Path) -> list[Document]:
    """Read ``documents.jsonl`` written by ``SyntheticDataset.write``."""
    return [Document.from_dict(r) for r in _read_jsonl(Path(data_dir) / "documents.jsonl")]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m ehr_rag generate` first")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- document rendering

def _meta(pid: str, doc_type: str, **extra: Any) -> dict[str, Any]:
    return {"synthetic": True, "patient_id": pid, "doc_type": doc_type, "source": "ehr_rag.synth", **extra}


def _patient_documents(patient: dict, recs: dict[str, list[dict]]) -> list[Document]:
    pid = patient["patient_id"]
    dates = [e["date"] for e in recs["encounters"]] + [e["discharge_date"] for e in recs["encounters"]
                                                        if e["discharge_date"]]
    as_of = max(dates) if dates else _iso(REFERENCE_DATE)
    docs: list[Document] = []

    lines = [
        f"Demographics for patient {pid} (synthetic record, not a real person).",
        f"- Name: {patient['name']}",
        f"- MRN: {patient['mrn']}",
        f"- Sex: {patient['sex']}",
        f"- Date of birth: {patient['birth_date']}",
        f"- Age: {patient['age']}",
    ]
    if patient.get("height_cm") is not None:
        lines.append(f"- Height: {patient['height_cm']} cm")
    lines.append(f"- PCP: {patient['pcp']}")
    docs.append(Document(demographics_doc_id(pid), pid, "demographics", as_of, "Demographics", "\n".join(lines),
                         _meta(pid, "demographics")))

    lines = [f"Problem list as of {as_of}."]
    for c in sorted(recs["conditions"], key=lambda c: (c["status"] != "active", c["onset_date"])):
        lines.append(f"- {c['display']} ({c['abbreviation']}) — ICD-10 {c['icd10']} — onset {c['onset_date']} — "
                     f"{c['status']}")
    docs.append(Document(problems_doc_id(pid), pid, "problem_list", as_of, "Problem List", "\n".join(lines),
                         _meta(pid, "problem_list")))

    lines = [f"Medication list as of {as_of}."]
    meds = sorted(recs["medications"], key=lambda m: (m["status"] != "active", m["start_date"], m["name"]))
    for m in meds:
        indication = V.CONDITIONS[m["indication"]].abbreviation
        if m["status"] == "active":
            lines.append(f"- {render_medication(m)} — started {m['start_date']} — for {indication} (active)")
        else:
            lines.append(f"- {render_medication(m)} — started {m['start_date']} — stopped {m['end_date']} — "
                         f"for {indication} (discontinued: {m['discontinue_reason']})")
    docs.append(Document(medications_doc_id(pid), pid, "medication_list", as_of, "Medication List",
                         "\n".join(lines), _meta(pid, "medication_list")))

    lines = [f"Allergy list as of {as_of}."]
    if recs["allergies"]:
        lines += [f"- {a['substance']} — {a['reaction']} ({a['severity']})" for a in recs["allergies"]]
    else:
        lines.append("No known drug allergies (NKDA).")
    docs.append(Document(allergies_doc_id(pid), pid, "allergy_list", as_of, "Allergy List", "\n".join(lines),
                         _meta(pid, "allergy_list")))

    enc_by_id = {e["encounter_id"]: e for e in recs["encounters"]}
    labs_by_date: dict[str, list[dict]] = {}
    for lab in recs["labs"]:
        labs_by_date.setdefault(lab["date"], []).append(lab)
    for day in sorted(labs_by_date):
        rows = labs_by_date[day]
        enc = enc_by_id[rows[0]["encounter_id"]]
        panels = [p for p in V.PANEL_ORDER if any(r["panel"] == p for r in rows)]
        lines = [f"Lab results drawn {day} ({enc['encounter_type']} encounter {enc['encounter_id']}); "
                 f"panels: {', '.join(panels)}."]
        for r in rows:
            unit = f" {r['unit']}" if r["unit"] else ""
            flag = f" [{r['flag']}]" if r["flag"] else ""
            lines.append(f"- {r['name']}: {format_lab_value(r['test_key'], r['value'])}{unit} "
                         f"(ref {format_ref_range(r['test_key'], r['ref_low'], r['ref_high'])}){flag}")
        docs.append(Document(lab_panel_doc_id(pid, day), pid, "lab_panel", day, f"Lab Panel {day}", "\n".join(lines),
                             _meta(pid, "lab_panel", encounter_id=enc["encounter_id"], panels=panels,
                                   tests=[r["test_key"] for r in rows])))

    for v in sorted(recs["vitals"], key=lambda v: v["date"]):
        enc = enc_by_id[v["encounter_id"]]
        lines = [
            f"Vital signs recorded {v['date']} ({enc['encounter_type']} encounter {enc['encounter_id']}).",
            f"- Blood pressure: {v['systolic']}/{v['diastolic']} mmHg",
            f"- Heart rate: {v['heart_rate']} bpm",
            f"- Respiratory rate: {v['resp_rate']} breaths/min",
            f"- Temperature: {v['temp_c']:.1f} C",
            f"- SpO2: {v['spo2']}%",
            f"- Weight: {v['weight_kg']:.1f} kg",
            f"- BMI: {v['bmi']:.1f} kg/m2",
        ]
        docs.append(Document(vitals_doc_id(pid, v["date"]), pid, "vitals", v["date"], f"Vital Signs {v['date']}",
                             "\n".join(lines), _meta(pid, "vitals", encounter_id=v["encounter_id"])))

    for n in sorted(recs["notes"], key=lambda n: n["note_id"]):
        docs.append(Document(n["note_id"], pid, "clinical_note", n["date"], n["title"], render_note_text(n),
                             _meta(pid, "clinical_note", note_id=n["note_id"], note_type=n["note_type"],
                                   encounter_id=n["encounter_id"], author=n["author"], mentions=n["mentions"])))
    return docs


def render_note_text(note: dict) -> str:
    """Sections as ``SECTION NAME:\\n<text>`` separated by blank lines."""
    return "\n\n".join(f"{name}:\n{text}" for name, text in note["sections"])


# --------------------------------------------------------------------------- generation

@dataclass
class _Shared:
    med_usage: Counter = field(default_factory=Counter)
    allergen_usage: Counter = field(default_factory=Counter)
    used_names: set = field(default_factory=set)


def generate_dataset(n_patients: int = 30, seed: int = 42) -> SyntheticDataset:
    """Generate ``n_patients`` synthetic patients deterministically from ``seed``."""
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1")
    shared = _Shared()
    ds = SyntheticDataset(seed=seed)
    for index in range(1, n_patients + 1):
        tables = _PatientBuilder(index, seed, shared).build()
        for table in TABLES:
            getattr(ds, table).extend(tables[table])
    logger.info("Generated %d synthetic patients (seed=%d)", n_patients, seed)
    return ds


_CONDITION_TERM_RE: dict[str, re.Pattern] = {
    key: re.compile("|".join(rf"(?<![\w]){re.escape(t)}(?![\w])" for t in spec.note_terms), re.IGNORECASE)
    for key, spec in V.CONDITIONS.items()
}


def mentioned_conditions(text: str, condition_keys: list[str]) -> list[str]:
    """Condition keys (from ``condition_keys``) whose ``note_terms`` occur in ``text``."""
    return [k for k in condition_keys if _CONDITION_TERM_RE[k].search(text)]


@dataclass
class _Plan:
    """An encounter plus planning details used by the note writers."""
    enc: dict
    kind: str                     # first | followup | tele | consult | ed | inpatient
    reason_key: str = ""          # ED / inpatient reason, consult specialty
    scenarios: list[V.SwitchScenario] = field(default_factory=list)
    imaging: list[str] = field(default_factory=list)


_ED_REASONS: dict[str, dict[str, str]] = {
    "chest_pain": {"label": "Chest pain", "cc": "Chest pain.",
                   "hpi": "Reports 2 hours of substernal chest pressure at rest, non-radiating, resolved before "
                          "arrival; no diaphoresis or syncope.",
                   "course": "ECG without acute ischemic changes; serial hs-troponin below the 99th percentile. "
                             "CXR without acute process. Chest pain-free throughout the ED stay.",
                   "dx": "Chest pain, low risk; ACS ruled out",
                   "plan": "Outpatient stress testing via cardiology; continue home regimen.", "imaging": "cxr"},
    "asthma_exac": {"label": "Asthma exacerbation", "cc": "Wheezing and SOB.",
                    "hpi": "2 days of worsening wheezing, chest tightness and cough after a viral URI; using rescue "
                           "inhaler every 2-3 hours.",
                    "course": "Received nebulized bronchodilators x3 and oral steroids with marked improvement; "
                              "ambulated without desaturation before discharge.",
                    "dx": "Moderate asthma exacerbation", "plan": "Steroid burst as below; asthma action plan "
                    "reviewed.", "imaging": "cxr"},
    "palpitations": {"label": "Palpitations", "cc": "Palpitations.",
                     "hpi": "Several hours of palpitations and mild DOE after missing one dose of home "
                            "rate-control medication; no chest pain or syncope.",
                     "course": "ECG AFib with ventricular rate in the 110s, no ischemic changes. Home rate-control "
                               "dose given with rate improving to the 80s. CXR without acute process.",
                     "dx": "AFib with mildly elevated ventricular rate", "plan": "Resume home regimen; adherence "
                     "reviewed.", "imaging": "cxr"},
    "knee_fall": {"label": "Fall with right knee pain", "cc": "Right knee pain after a fall.",
                  "hpi": "Tripped on a rug at home and landed on the right knee; no head strike or LOC; able to bear "
                         "weight with pain.",
                  "course": "XR right knee without fracture; degenerative changes noted. Pain improved with ice and "
                            "an oral analgesic.",
                  "dx": "Right knee contusion after mechanical fall", "plan": "Knee sleeve, ice, PT referral; fall "
                  "precautions reviewed.", "imaging": "knee_xr"},
    "epigastric": {"label": "Epigastric pain", "cc": "Epigastric pain.",
                   "hpi": "Burning epigastric discomfort after a large late meal, no radiation, no exertional "
                          "component, no melena.",
                   "course": "ECG without acute ischemic changes. Symptoms improved with an oral antacid.",
                   "dx": "Dyspepsia", "plan": "Avoid late meals; continue home acid suppression.", "imaging": ""},
    "copd_flare": {"label": "COPD exacerbation (mild)", "cc": "SOB.",
                   "hpi": "3 days of increased DOE and sputum production, no fever or chest pain.",
                   "course": "Nebulized bronchodilators x2 with improvement; CXR without consolidation. Ambulated "
                             "without desaturation.",
                   "dx": "Mild COPD exacerbation", "plan": "Steroid burst as below; continue maintenance inhaler.",
                   "imaging": "cxr"},
    "lightheadedness": {"label": "Lightheadedness", "cc": "Lightheadedness.",
                        "hpi": "Lightheadedness on standing for 2 days with reduced oral intake during a viral "
                               "illness; no syncope, chest pain or focal weakness.",
                        "course": "Orthostatic vital signs mildly positive; improved after oral hydration. ECG "
                                  "without acute changes.",
                        "dx": "Orthostatic lightheadedness, likely volume depletion",
                        "plan": "Oral hydration; rise slowly.", "imaging": ""},
}

_INPATIENT_REASONS: dict[str, dict[str, str]] = {
    "hf_exac": {"label": "Acute on chronic HFrEF exacerbation", "cc": "SOB and leg swelling.",
                "hpi": "1 week of progressive DOE, 3-pillow orthopnea and LE edema after dietary sodium "
                       "indiscretion.",
                "dx": "Acute on chronic HFrEF exacerbation"},
    "copd_exac": {"label": "Acute COPD exacerbation", "cc": "SOB.",
                  "hpi": "4 days of increased dyspnea, cough and purulent sputum after a viral URI.",
                  "dx": "Acute COPD exacerbation"},
    "cap": {"label": "Community-acquired pneumonia", "cc": "Fever and productive cough.",
            "hpi": "3 days of fever, productive cough and right-sided pleuritic chest pain.",
            "dx": "Community-acquired pneumonia, RLL"},
    "afib_rvr": {"label": "AFib with rapid ventricular response", "cc": "Palpitations and DOE.",
                 "hpi": "1 day of palpitations and DOE; ED ECG showed AFib with RVR.",
                 "dx": "AFib with RVR"},
}


class _PatientBuilder:
    """Builds one patient's records. All randomness comes from ``self.rng``."""

    def __init__(self, index: int, seed: int, shared: _Shared) -> None:
        self.index = index
        self.seed = seed
        self.pid = f"P{index:03d}"
        self.rng = random.Random(f"ehr-rag:{seed}:{index}")
        self.shared = shared
        self.meds: list[dict] = []
        self.labs: list[dict] = []
        self.vitals: dict[str, dict] = {}
        self.notes: list[dict] = []
        self.plans: list[_Plan] = []
        self.scenario_at: dict[str, list[V.SwitchScenario]] = {}
        self.scenario_meds: dict[str, tuple[dict, Optional[dict]]] = {}

    # ------------------------------------------------------------------ top level
    def build(self) -> dict[str, list[dict]]:
        self._demographics()
        self._choose_conditions()
        self._allergies()
        self._schedule()
        self._choose_scenarios()
        self._onsets()
        self._medications()
        self._labs_and_vitals()
        self._finalize_reasons()
        self._notes()
        for name in sorted({m["name"] for m in self.meds}):
            self.shared.med_usage[name] += 1
        return {
            "patients": [self.patient],
            "encounters": [p.enc for p in self.plans],
            "conditions": self.condition_rows,
            "medications": self.meds,
            "allergies": self.allergy_rows,
            "labs": self.labs,
            "vitals": [self.vitals[k] for k in sorted(self.vitals)],
            "notes": self.notes,
        }

    # ------------------------------------------------------------------ demographics
    def _demographics(self) -> None:
        rng = self.rng
        self.sex = rng.choice(("female", "male"))
        firsts = V.FEMALE_FIRST_NAMES if self.sex == "female" else V.MALE_FIRST_NAMES
        for _ in range(1000):
            name = f"{rng.choice(firsts)} {rng.choice(V.LAST_NAMES)}"
            if name not in self.shared.used_names:
                break
        self.shared.used_names.add(name)
        age = rng.randint(42, 86)
        self.birth = date(REFERENCE_DATE.year - age - 1, 7, 1) + timedelta(days=rng.randint(0, 364))
        self.age = _age_on(self.birth, REFERENCE_DATE)
        self.height_cm = rng.randint(165, 190) if self.sex == "male" else rng.randint(152, 173)
        self.pcp = rng.choice(V.PROVIDERS["primary_care"])
        self.patient = {
            "patient_id": self.pid, "mrn": f"SYN-{100000 + self.index}", "name": name, "sex": self.sex,
            "birth_date": _iso(self.birth), "age": self.age, "pcp": self.pcp, "height_cm": self.height_cm,
            "synthetic": True,
        }

    def _age_at(self, d: date) -> int:
        return _age_on(self.birth, d)

    # ------------------------------------------------------------------ conditions
    def _choose_conditions(self) -> None:
        rng = self.rng
        keys = list(V.PROFILES)
        profile = V.PROFILES[rng.choices(keys, weights=[V.PROFILES[k].weight for k in keys])[0]]
        self.profile = profile.key
        conds = list(profile.core)
        for key, p in profile.optional:
            if rng.random() < p and key not in conds:
                conds.append(key)
        if rng.random() < 0.45 or len(conds) < V.MIN_CHRONIC_CONDITIONS:
            options = [k for k in V.ADD_ON_CONDITIONS if k not in conds and self._compatible(k, conds)]
            if options:
                conds.append(rng.choice(options))
        while len(conds) < V.MIN_CHRONIC_CONDITIONS:
            options = [k for k in V.ADD_ON_CONDITIONS if k not in conds and self._compatible(k, conds)]
            conds.append(rng.choice(options))
        conds = conds[: V.MAX_CHRONIC_CONDITIONS]
        self.conds: list[str] = [k for k in V.CONDITION_PRIORITY if k in conds]
        self.ef = rng.choice((25, 30, 35)) if "hfref" in self.conds else None
        self.mi_vessel = rng.choice(("LAD", "RCA", "LCx"))
        self.pack_years = rng.randint(25, 60)
        smoke = (f"former smoker, {self.pack_years} pack-years, quit {rng.randint(2, 15)} years ago"
                 if "copd" in self.conds else rng.choice(V.SOCIAL_TOBACCO))
        self.social = (f"Tobacco: {smoke}. Alcohol: {rng.choice(V.SOCIAL_ALCOHOL)}. "
                       f"Lives {rng.choice(V.SOCIAL_LIVING)}; independent in ADLs. "
                       f"Works as {rng.choice(V.SOCIAL_OCCUPATION)}.")

    @staticmethod
    def _compatible(key: str, conds: list[str]) -> bool:
        return not any(key in pair and (set(pair) - {key}) & set(conds) for pair in V.INCOMPATIBLE_CONDITIONS)

    def _onsets(self) -> None:
        rng = self.rng
        first = self.first_date
        onset: dict[str, date] = {}
        for key in self.conds:
            onset[key] = first - timedelta(days=rng.randint(400, 15 * 365))
        if "prior_mi" in onset:
            mi = first - timedelta(days=rng.randint(500, 8 * 365))
            onset["prior_mi"] = mi
            if "cad" in onset:
                onset["cad"] = mi
            if "hfref" in onset:
                onset["hfref"] = min(mi + timedelta(days=rng.randint(30, 300)), first - timedelta(days=200))
        if "afib" in onset and "hfref" in onset:
            onset["afib"] = max(onset["afib"], onset["hfref"])
        self.onset = onset
        rows = []
        for key in self.conds:
            spec = V.CONDITIONS[key]
            rows.append({"patient_id": self.pid, "condition_key": key, "display": spec.display, "icd10": spec.icd10,
                         "abbreviation": spec.abbreviation, "onset_date": _iso(onset[key]), "status": "active",
                         "synthetic": True})
        for plan in self.plans:
            if plan.kind == "inpatient" and plan.reason_key == "cap":
                spec = V.CONDITIONS["cap"]
                self.onset["cap"] = _date(plan.enc["date"])
                rows.append({"patient_id": self.pid, "condition_key": "cap", "display": spec.display,
                             "icd10": spec.icd10, "abbreviation": spec.abbreviation, "onset_date": plan.enc["date"],
                             "status": "resolved", "synthetic": True})
        self.condition_rows = rows
        self.all_conds = [r["condition_key"] for r in rows]

    # ------------------------------------------------------------------ allergies
    def _allergies(self) -> None:
        rng = self.rng
        self.allergy_rows: list[dict] = []
        self.excluded_meds: set[str] = set()
        # NKDA is stratified: round(NKDA_PROBABILITY * 10) patients in every block of 10 (exact ~30%)
        block, pos = divmod(self.index - 1, 10)
        nkda = random.Random(f"nkda:{self.seed}:{block}").sample(range(10), round(V.NKDA_PROBABILITY * 10))
        if pos in nkda:
            return
        n = 1 if rng.random() < 0.65 else 2
        chosen: list[V.AllergenSpec] = []
        for _ in range(n):
            pool = [a for a in V.ALLERGENS if a not in chosen]
            open_pool = [a for a in pool if self.shared.allergen_usage[a.substance] < V.MAX_PATIENTS_PER_ALLERGEN]
            pool = open_pool or pool
            low = min(self.shared.allergen_usage[a.substance] for a in pool)
            least = [a for a in pool if self.shared.allergen_usage[a.substance] <= low + 1]
            chosen.append(rng.choice(least))
        for a in chosen:
            reaction, severity = rng.choice(a.reactions)
            self.allergy_rows.append({"patient_id": self.pid, "substance": a.substance, "reaction": reaction,
                                      "severity": severity, "synthetic": True})
            self.shared.allergen_usage[a.substance] += 1
            self.excluded_meds.update(a.excludes)

    def _allergy_text(self) -> str:
        if not self.allergy_rows:
            return "NKDA."
        return "; ".join(f"{a['substance']} ({a['reaction']})" for a in self.allergy_rows) + "."

    # ------------------------------------------------------------------ encounters
    def _schedule(self) -> None:
        rng = self.rng
        conds = self.conds
        n_enc = rng.randint(4, 7)
        first = REFERENCE_DATE - timedelta(days=rng.randint(690, 730))
        last = REFERENCE_DATE - timedelta(days=rng.randint(5, 40))
        self.first_date = first
        n_mid = n_enc - 2
        span = (last - first).days
        dates = [first] + [first + timedelta(days=int(span * i / (n_mid + 1)) + rng.randint(-15, 15))
                           for i in range(1, n_mid + 1)] + [last]

        specials: list[tuple[str, str]] = []
        inpatient_options = [r for r, c in (("hf_exac", "hfref"), ("copd_exac", "copd"), ("afib_rvr", "afib"))
                             if c in conds]
        if inpatient_options or self.age >= 65 or "asthma" in conds:
            inpatient_options.append("cap")
        p_inpatient = 0.75 if any(c in conds for c in ("hfref", "copd", "afib")) else 0.25
        if inpatient_options and rng.random() < p_inpatient:
            specials.append(("inpatient", rng.choice(inpatient_options)))
        if rng.random() < 0.4:
            ed_options = [r for r, c in (("chest_pain", "cad"), ("asthma_exac", "asthma"), ("palpitations", "afib"),
                                         ("knee_fall", "oa"), ("epigastric", "gerd"), ("copd_flare", "copd"))
                          if c in conds] or ["lightheadedness"]
            specials.append(("ed", rng.choice(ed_options)))
        specialties = sorted({V.SPECIALTY_FOR_CONDITION[c] for c in conds if c in V.SPECIALTY_FOR_CONDITION})
        if specialties and rng.random() < 0.6:
            specials.append(("consult", rng.choice(specialties)))
        if rng.random() < 0.5:
            specials.append(("tele", ""))
        specials = specials[:n_mid]
        mid_kinds = specials + [("followup", "")] * (n_mid - len(specials))
        rng.shuffle(mid_kinds)
        kinds = [("first", "")] + mid_kinds + [("followup", "")]

        for i, (d, (kind, reason_key)) in enumerate(zip(dates, kinds), start=1):
            enc_type = {"first": "outpatient", "followup": "outpatient", "consult": "outpatient", "tele": "telehealth",
                        "ed": "ed", "inpatient": "inpatient"}[kind]
            provider = self.pcp
            discharge = None
            if kind == "consult":
                provider = rng.choice(V.PROVIDERS[reason_key])
            elif kind == "ed":
                provider = rng.choice(V.PROVIDERS["emergency"])
            elif kind == "inpatient":
                provider = rng.choice(V.PROVIDERS["hospital_medicine"])
                discharge = _iso(d + timedelta(days=rng.randint(2, 5)))
            enc = {"patient_id": self.pid, "encounter_id": f"{self.pid}-E{i:02d}", "date": _iso(d),
                   "encounter_type": enc_type, "reason": "", "provider": provider, "discharge_date": discharge,
                   "synthetic": True}
            self.plans.append(_Plan(enc=enc, kind=kind, reason_key=reason_key))

    # ------------------------------------------------------------------ scenarios
    def _class_ok(self, med: str, claimed: set[str]) -> bool:
        spec = V.MEDICATIONS[med]
        if med in self.excluded_meds:
            return False
        for item in spec.avoid_with:
            if item in self.conds or item in claimed:
                return False
            if item == "anticoagulant" and "afib" in self.conds:
                return False
        return True

    def _choose_scenarios(self) -> None:
        rng = self.rng
        chosen: list[tuple[V.SwitchScenario, _Plan]] = []
        claimed: set[str] = set()

        def try_add(s: V.SwitchScenario, plan: _Plan) -> bool:
            meds = [s.from_med[0]] + ([s.to_med[0]] if s.to_med else [])
            classes = {V.MEDICATIONS[m].drug_class for m in meds}
            if classes & claimed or any(c.condition == s.condition for c, _ in chosen):
                return False
            if not all(self._class_ok(m, claimed) for m in meds):
                return False
            chosen.append((s, plan))
            claimed.update(classes)
            return True

        by_key = {s.key: s for s in V.SWITCHES}
        for plan in self.plans:
            if plan.kind == "inpatient" and plan.reason_key in ("hf_exac", "afib_rvr"):
                key = "diuretic_increase" if plan.reason_key == "hf_exac" else "rate_control_increase"
                if not try_add(by_key[key], plan) and plan.reason_key == "afib_rvr":
                    plan.reason_key = "cap"     # rate-control scenario impossible; fall back to a CAP admission

        eligible = [p for p in self.plans[1:] if p.kind == "followup"]
        candidates = [s for s in V.SWITCHES if s.setting == "outpatient" and s.condition in self.conds]
        rng.shuffle(candidates)
        n_out = 1 if rng.random() < 0.55 else 2
        slots = sorted(rng.sample(eligible, min(n_out, len(eligible))), key=lambda p: p.enc["date"])
        for plan in slots:
            for s in candidates:
                if try_add(s, plan):
                    candidates.remove(s)
                    break
        if not chosen:
            raise RuntimeError(f"{self.pid}: no feasible medication change scenario")
        for s, plan in chosen:
            plan.scenarios.append(s)
        self.scenarios = chosen
        self.claimed_by_scenarios = claimed

        # imaging attached to outpatient follow-ups / ED / inpatient
        for plan in self.plans:
            if plan.kind == "ed" and _ED_REASONS[plan.reason_key]["imaging"]:
                plan.imaging.append(_ED_REASONS[plan.reason_key]["imaging"])
            elif plan.kind == "inpatient":
                plan.imaging.append("cxr")
        outpatient = [p for p in self.plans if p.kind == "followup"]
        for kind, cond, prob in (("tte", "hfref", 0.7), ("ldct", "copd", 0.6), ("renal_us", "ckd3", 0.5),
                                 ("knee_xr", "oa", 0.35)):
            if cond in self.conds and rng.random() < prob:
                rng.choice(outpatient).imaging.append(kind)
                break

    # ------------------------------------------------------------------ medications
    def _new_med(self, name: str, dose: str, indication: str, start: date, end: Optional[date],
                 reason: Optional[str]) -> dict:
        spec = V.MEDICATIONS[name]
        rec = {"patient_id": self.pid, "med_id": "", "name": name, "dose": dose, "route": spec.route,
               "frequency": spec.frequency, "start_date": _iso(start), "end_date": _iso(end) if end else None,
               "status": "discontinued" if end else "active", "indication": indication,
               "discontinue_reason": reason, "synthetic": True}
        self.meds.append(rec)
        return rec

    def _chronic_start(self, cond: str) -> date:
        rng = self.rng
        start = self.onset[cond] + timedelta(days=rng.randint(7, 120))
        return min(start, self.first_date - timedelta(days=rng.randint(14, 90)))

    def _medications(self) -> None:
        rng = self.rng
        claimed: set[str] = set()
        for s, plan in self.scenarios:
            change_day = _date(plan.enc["discharge_date"] if s.setting != "outpatient" else plan.enc["date"])
            start = max(self.onset[s.condition] + timedelta(days=7),
                        self.first_date - timedelta(days=rng.randint(30, 240)))
            old = self._new_med(s.from_med[0], s.from_med[1], s.condition, start, change_day, s.reason)
            new = self._new_med(s.to_med[0], s.to_med[1], s.condition, change_day, None, None) if s.to_med else None
            self.scenario_meds[s.key] = (old, new)
            claimed.add(V.MEDICATIONS[s.from_med[0]].drug_class)
            if s.to_med:
                claimed.add(V.MEDICATIONS[s.to_med[0]].drug_class)

        for cond in self.conds:
            scen_classes = {V.MEDICATIONS[m].drug_class for sc, _ in self.scenarios if sc.condition == cond
                            for m in [sc.from_med[0]] + ([sc.to_med[0]] if sc.to_med else [])}
            for slot in V.REGIMENS.get(cond, ()):
                if any(o is not None and V.MEDICATIONS[o[0]].drug_class in scen_classes for o in slot):
                    continue            # the scenario's drug fills this slot
                options: list[Optional[tuple[str, str]]] = []
                for opt in slot:
                    if opt is None:
                        options.append(None)
                        continue
                    spec = V.MEDICATIONS[opt[0]]
                    if spec.drug_class in claimed or not self._class_ok(opt[0], claimed):
                        continue
                    options.append(opt)
                if not options:
                    continue
                weights = [1.0 if o is None else 1.0 / (1 + self.shared.med_usage[o[0]]) for o in options]
                pick = rng.choices(options, weights=weights)[0]
                if pick is None:
                    continue
                self._new_med(pick[0], pick[1], cond, self._chronic_start(cond), None, None)
                claimed.add(V.MEDICATIONS[pick[0]].drug_class)

        # short courses for acute encounters
        self.course_meds: dict[str, list[dict]] = {}
        for plan in self.plans:
            d = _date(plan.enc["date"])
            courses: list[dict] = []
            if plan.kind == "inpatient":
                dc = _date(plan.enc["discharge_date"])
                if plan.reason_key == "copd_exac":
                    end = dc + timedelta(days=rng.randint(1, 3))
                    courses.append(self._new_med("prednisone", "40 mg", "copd", d, end,
                                                 f"completed {(end - d).days}-day course"))
                    end = max(d + timedelta(days=5), dc + timedelta(days=1))
                    courses.append(self._new_med("azithromycin", "250 mg", "copd", d, end,
                                                 f"completed {(end - d).days}-day course"))
                elif plan.reason_key == "cap":
                    abx = next(a for a in ("amoxicillin-clavulanate", "levofloxacin", "doxycycline")
                               if a not in self.excluded_meds)
                    end = d + timedelta(days=7)
                    courses.append(self._new_med(abx, V.MEDICATIONS[abx].doses[0], "cap", d, end,
                                                 "completed 7-day course"))
            elif plan.kind == "ed" and plan.reason_key in ("asthma_exac", "copd_flare"):
                cond = "asthma" if plan.reason_key == "asthma_exac" else "copd"
                courses.append(self._new_med("prednisone", "40 mg", cond, d, d + timedelta(days=5),
                                             "completed 5-day course"))
            self.course_meds[plan.enc["encounter_id"]] = courses

        self.meds.sort(key=lambda m: (m["start_date"], m["name"], m["dose"]))
        for i, m in enumerate(self.meds, start=1):
            m["med_id"] = f"{self.pid}-M{i:02d}"

    def active_meds(self, day: str) -> list[dict]:
        return sorted((m for m in self.meds if is_active_on(m, day)), key=lambda m: (m["name"], m["start_date"]))

    def meds_for(self, cond: str, day: str) -> list[dict]:
        """Active meds charted for ``cond`` or whose drug treats it (e.g. a statin charted under CAD for HLD)."""
        return [m for m in self.active_meds(day)
                if m["indication"] == cond or cond in V.MEDICATIONS[m["name"]].indications]

    def pre_visit_meds(self, day: str) -> list[dict]:
        """Medications taken before a visit on ``day`` (drives labs and vitals)."""
        return [m for m in self.meds if m["start_date"] < day and (m["end_date"] is None or m["end_date"] >= day)]

    # ------------------------------------------------------------------ labs & vitals
    def _labs_and_vitals(self) -> None:
        rng = self.rng
        conds = self.conds
        has = conds.__contains__
        self.egfr_base = rng.uniform(47, 57) if has("ckd3") else max(64.0, rng.uniform(74, 104) -
                                                                      0.4 * max(0, self.age - 50))
        self.ldl_untreated = rng.uniform(145, 185) if (has("hld") or has("cad")) else rng.uniform(80, 116)
        self.hdl_base = (rng.uniform(36, 50) if self.sex == "male" else rng.uniform(45, 66)) - (4 if has("t2dm")
                                                                                               else 0)
        self.tg_base = rng.uniform(90, 150) + (55 if has("t2dm") else 0) + (25 if has("obesity") else 0)
        self.k_base = rng.uniform(3.9, 4.5)
        self.na_base = rng.uniform(137, 142)
        self.bun_base = rng.uniform(21, 30) if has("ckd3") else rng.uniform(10, 18)
        self.hgb_base = (rng.uniform(13.8, 15.8) if self.sex == "male" else rng.uniform(12.4, 14.2)) - (
            1.3 if has("ckd3") else 0)
        self.wbc_base = rng.uniform(5.2, 8.6)
        self.plt_base = rng.uniform(185, 330)
        self.alt_base = rng.uniform(14, 34)
        self.ast_base = rng.uniform(16, 30)
        self.tsh_base = rng.uniform(1.3, 3.2) if has("hypothyroid") else rng.uniform(0.9, 2.6)
        self.bnp_base = (rng.uniform(260, 600) if has("hfref") else rng.uniform(95, 180) if has("afib")
                         else rng.uniform(15, 70))
        titration = any(s.key == "metformin_titration" for s, _ in self.scenarios)
        self.a1c_first = (rng.uniform(8.4, 9.3) if titration else rng.uniform(7.1, 8.6)) if has("t2dm") else (
            rng.uniform(5.4, 6.1) if has("obesity") else rng.uniform(5.0, 5.6))
        self.a1c_current = self.a1c_first
        self.a1c_drawn = False
        bmi_target = rng.uniform(31.5, 38.5) if has("obesity") else rng.uniform(22.5, 28.8)
        self.weight_base = bmi_target * (self.height_cm / 100) ** 2
        self.sbp_offset = rng.uniform(-4, 4)
        self.phq: dict[str, int] = {}

        prev_kind = ""
        for idx, plan in enumerate(self.plans):
            d = plan.enc["date"]
            pre = self.pre_visit_meds(d)
            classes = {V.MEDICATIONS[m["name"]].drug_class for m in pre}
            names = {m["name"] for m in pre}
            overrides = {s.lab_override[0]: s.lab_override for s in plan.scenarios if s.lab_override}
            panels = self._panels_for(plan, idx, names, overrides)
            ctx = {"plan": plan, "pre": pre, "classes": classes, "names": names, "prev_kind": prev_kind,
                   "idx": idx}
            values = self._lab_values(panels, ctx, overrides)
            for panel in V.PANEL_ORDER:
                if panel not in panels:
                    continue
                for test_key in V.PANEL_TESTS[panel]:
                    if test_key not in values:
                        continue
                    spec = V.LAB_TESTS[test_key]
                    low, high = V.ref_range(test_key, self.sex)
                    value = values[test_key]
                    self.labs.append({
                        "patient_id": self.pid, "date": d, "encounter_id": plan.enc["encounter_id"], "panel": panel,
                        "test_key": test_key, "name": spec.name, "loinc": spec.loinc, "value": value,
                        "unit": spec.unit, "ref_low": low, "ref_high": high, "flag": lab_flag(value, low, high),
                        "synthetic": True,
                    })
            if plan.kind != "tele":
                self.vitals[d] = self._vital_values(plan, ctx)
            prev_kind = plan.reason_key if plan.kind == "inpatient" else plan.kind

    def _panels_for(self, plan: _Plan, idx: int, names: set[str], overrides: dict) -> list[str]:
        has = self.conds.__contains__
        kind = plan.kind
        panels: list[str] = []
        warfarin = "warfarin" in names
        if kind == "first":
            panels = ["BMP", "CBC", "Lipid panel", "Hepatic function panel"]
            if has("t2dm") or has("obesity"):
                panels.append("HbA1c")
            if has("hypothyroid") or has("mdd") or has("afib"):
                panels.append("TSH")
            if has("hfref") or has("afib"):
                panels.append("BNP")
        elif kind == "followup":
            panels = ["BMP"]
            if has("t2dm"):
                panels.append("HbA1c")
            if has("hypothyroid"):
                panels.append("TSH")
            if has("hfref"):
                panels.append("BNP")
            is_last = idx == len(self.plans) - 1
            if is_last:
                panels.append("CBC")
                if has("hld") or has("cad"):
                    panels += ["Lipid panel", "Hepatic function panel"]
        elif kind == "consult":
            panels = {"nephrology": ["BMP", "CBC"], "endocrinology": ["BMP"], "cardiology": ["BMP", "BNP"],
                      "pulmonology": ["CBC"], "behavioral_health": []}[plan.reason_key]
            if plan.reason_key == "endocrinology":
                panels += (["HbA1c"] if has("t2dm") else []) + (["TSH"] if has("hypothyroid") else [])
        elif kind == "ed":
            panels = ["BMP", "CBC"]
            if has("hfref") or plan.reason_key in ("chest_pain", "palpitations", "copd_flare"):
                panels.append("BNP")
        elif kind == "inpatient":
            panels = ["BMP", "CBC"]
            if has("hfref") or plan.reason_key in ("hf_exac", "afib_rvr", "copd_exac"):
                panels.append("BNP")
            if plan.reason_key == "afib_rvr":
                panels.append("TSH")
        if warfarin and kind != "tele":
            panels.append("Coagulation")
        for test_key in overrides:
            panel = V.LAB_TESTS[test_key].panel
            if panel not in panels:
                panels.append(panel)
        return [p for p in V.PANEL_ORDER if p in panels]

    def _lab_values(self, panels: list[str], ctx: dict, overrides: dict) -> dict[str, float]:
        rng = self.rng
        plan: _Plan = ctx["plan"]
        has = self.conds.__contains__
        classes, names = ctx["classes"], ctx["names"]
        reason = plan.reason_key if plan.kind in ("inpatient", "ed") else ""
        inpatient = plan.kind == "inpatient"
        d = _date(plan.enc["date"])
        out: dict[str, float] = {}

        def rnd(key: str, v: float) -> float:
            if key in overrides:
                _, lo, hi = overrides[key]
                v = rng.uniform(lo, hi)
            return float(round(v, V.LAB_TESTS[key].decimals))

        # glycemic state evolves at each A1c draw
        if "HbA1c" in panels:
            if has("t2dm"):
                if self.a1c_drawn:
                    prev = self.a1c_current
                    step = -rng.uniform(0.15, 0.6) if prev > 7.2 else rng.uniform(-0.3, 0.25)
                    self.a1c_current = max(6.3, prev + step)
                self.a1c_drawn = True
            else:
                self.a1c_current = self.a1c_first + rng.gauss(0, 0.1)
            out["hba1c"] = rnd("hba1c", self.a1c_current)
            if "hba1c" in overrides:
                self.a1c_current = out["hba1c"]
        if "BMP" in panels:
            if has("t2dm"):
                glucose = 28.7 * self.a1c_current - 46.7 + rng.gauss(0, 14)
                glucose = min(max(glucose, 96), 260)
            else:
                glucose = rng.uniform(82, 99)
            if inpatient:
                glucose += rng.uniform(15, 40)
            out["glucose"] = rnd("glucose", glucose)
            years = (d - self.first_date).days / 365.25
            target = self.egfr_base - 1.0 * years + rng.gauss(0, 2.0)
            lo, hi = (45.5, 58.5) if has("ckd3") else (61.5, 130.0)
            target = min(max(target, lo), hi)
            if inpatient and reason in ("hf_exac", "cap"):
                target *= rng.uniform(0.78, 0.88)
                lo, hi = 20.0, 130.0
            age = self._age_at(d)
            creat = round(_creatinine_for_egfr(target, age, self.sex), 2)
            for _ in range(40):           # keep the rounded pair inside the intended band
                egfr = round(ckd_epi_2021(creat, age, self.sex))
                if egfr < lo:
                    creat = round(creat - 0.01, 2)
                elif egfr > hi:
                    creat = round(creat + 0.01, 2)
                else:
                    break
            out["creatinine"] = float(creat)
            out["egfr"] = float(round(ckd_epi_2021(creat, age, self.sex)))
            bun = self.bun_base + rng.gauss(0, 2) + (rng.uniform(8, 14) if reason == "hf_exac" else 0) + (
                rng.uniform(3, 8) if reason == "cap" else 0)
            out["bun"] = rnd("bun", max(6.0, bun))
            k = (self.k_base + 0.15 * ("raas" in classes) + 0.25 * ("mra" in classes) + 0.2 * has("ckd3")
                 - 0.3 * ("loop" in classes) + rng.gauss(0, 0.15))
            out["potassium"] = rnd("potassium", min(max(k, 3.5), 5.0))
            na = self.na_base + rng.gauss(0, 1.2) - 2 * has("hfref") - 1 * ("thiazide" in classes)
            out["sodium"] = rnd("sodium", min(max(na, 134), 144))
        if "Lipid panel" in panels:
            reduction = max((_STATIN_REDUCTION.get((m["name"], m["dose"]), 0.0) for m in ctx["pre"]), default=0.0)
            ldl = rnd("ldl", max(40.0, self.ldl_untreated * (1 - reduction) + rng.gauss(0, 5)))
            hdl = rnd("hdl", max(28.0, self.hdl_base + rng.gauss(0, 2.5)))
            tg = rnd("triglycerides", max(50.0, self.tg_base * (0.88 if reduction else 1.0) + rng.gauss(0, 12)))
            out.update({"ldl": ldl, "hdl": hdl, "triglycerides": tg,
                        "total_cholesterol": float(round(ldl + hdl + tg / 5))})
        if "TSH" in panels:
            out["tsh"] = rnd("tsh", min(max(self.tsh_base + rng.gauss(0, 0.3), 0.5), 4.3))
        if "BNP" in panels:
            if reason == "hf_exac":
                bnp = rng.uniform(1100, 2200)
            elif reason == "afib_rvr":
                bnp = rng.uniform(220, 420) + (300 if has("hfref") else 0)
            elif has("hfref") and ctx["prev_kind"] == "hf_exac":
                bnp = rng.uniform(420, 780)
            else:
                bnp = self.bnp_base * rng.uniform(0.85, 1.15)
            out["bnp"] = rnd("bnp", bnp)
        if "CBC" in panels:
            out["hemoglobin"] = rnd("hemoglobin", self.hgb_base + rng.gauss(0, 0.3))
            wbc = self.wbc_base + rng.gauss(0, 0.6)
            if reason == "cap":
                wbc = rng.uniform(13.5, 17.8)
            elif reason == "copd_exac":
                wbc = rng.uniform(9.5, 12.5)
            out["wbc"] = rnd("wbc", max(3.8, wbc))
            out["platelets"] = rnd("platelets", self.plt_base + rng.gauss(0, 15))
        if "Hepatic function panel" in panels:
            out["alt"] = rnd("alt", self.alt_base + rng.gauss(0, 3))
            out["ast"] = rnd("ast", self.ast_base + rng.gauss(0, 3))
        if "Coagulation" in panels:
            out["inr"] = rnd("inr", rng.uniform(2.0, 2.9) if "warfarin" in names else rng.uniform(0.9, 1.1))
        return out

    def _vital_values(self, plan: _Plan, ctx: dict) -> dict:
        rng = self.rng
        has = self.conds.__contains__
        classes, names = ctx["classes"], ctx["names"]
        reason = plan.reason_key if plan.kind in ("inpatient", "ed") else ""
        d = _date(plan.enc["date"])
        visit = ctx["idx"]
        if has("hfref"):
            sbp = rng.uniform(102, 122)
        elif has("htn"):
            sbp = max(rng.uniform(124, 134), 152 - 6 * visit) + self.sbp_offset
        else:
            sbp = rng.uniform(110, 126)
        hr = rng.uniform(58, 72) if "beta_blocker" in classes else rng.uniform(66, 86)
        if has("afib"):
            hr = rng.uniform(70, 92)
        rr, temp = rng.randint(14, 18), rng.uniform(36.4, 37.1)
        spo2 = rng.randint(91, 94) if has("copd") else rng.randint(95, 99)
        weight = self.weight_base + rng.gauss(0, 0.6)
        if "semaglutide" in names:
            start = min(_date(m["start_date"]) for m in self.meds if m["name"] == "semaglutide")
            weight -= min(9.0, 0.7 * (d - start).days / 30.4)
        if reason == "hf_exac":
            sbp += 10
            hr, rr, spo2 = rng.uniform(96, 110), rng.randint(22, 26), rng.randint(89, 93)
            weight += rng.uniform(3.0, 4.5)
        elif reason == "copd_exac":
            hr, rr, spo2 = rng.uniform(100, 112), rng.randint(24, 28), rng.randint(86, 89)
        elif reason == "cap":
            hr, rr, spo2, temp = rng.uniform(98, 112), rng.randint(22, 26), rng.randint(90, 93), rng.uniform(38.4,
                                                                                                         39.1)
        elif reason == "afib_rvr":
            hr, rr = rng.uniform(128, 148), rng.randint(20, 24)
        elif reason in ("asthma_exac", "copd_flare"):
            hr, rr, spo2 = rng.uniform(96, 108), rng.randint(22, 24), rng.randint(91, 94)
        elif reason == "palpitations":
            hr = rng.uniform(104, 122)
        elif reason == "chest_pain":
            sbp += 8
        elif reason == "lightheadedness":
            sbp -= 10
        h2 = (self.height_cm / 100) ** 2
        if has("obesity"):
            weight = max(weight, 30.6 * h2)
        else:
            weight = min(weight, 29.8 * h2)
        weight = round(weight, 1)
        sbp = int(round(sbp))
        dbp = int(round(min(max(sbp * 0.6 + rng.gauss(0, 3), 60), 96)))
        return {"patient_id": self.pid, "date": plan.enc["date"], "encounter_id": plan.enc["encounter_id"],
                "systolic": sbp, "diastolic": dbp, "heart_rate": int(round(hr)), "resp_rate": rr,
                "temp_c": round(temp, 1), "spo2": spo2, "weight_kg": weight, "bmi": round(weight / h2, 1),
                "synthetic": True}

    # ------------------------------------------------------------------ reasons
    def lab_at(self, test_key: str, day: str) -> Optional[dict]:
        return next((r for r in self.labs if r["test_key"] == test_key and r["date"] == day), None)

    def latest_lab(self, test_key: str, day: str, before: bool = False) -> Optional[dict]:
        rows = [r for r in self.labs if r["test_key"] == test_key and (r["date"] < day if before else r["date"] <= day)]
        return max(rows, key=lambda r: r["date"]) if rows else None

    @staticmethod
    def lab_text(row: dict) -> str:
        return f"{V.LAB_TESTS[row['test_key']].short} {format_lab_value(row['test_key'], row['value'])}"

    def _finalize_reasons(self) -> None:
        for s, plan in self.scenarios:
            old, _ = self.scenario_meds[s.key]
            if s.lab_override:
                row = self.lab_at(s.lab_override[0], plan.enc["date"])
                old["discontinue_reason"] = s.reason.format(lab=self.lab_text(row))
        for plan in self.plans:
            enc = plan.enc
            abbrs = [V.CONDITIONS[c].abbreviation for c in self.conds[:2]]
            if plan.kind == "first":
                enc["reason"] = "Annual visit and chronic disease management"
            elif plan.kind == "followup":
                base = f"Follow-up of {_join(abbrs)}"
                enc["reason"] = f"{plan.scenarios[0].complaint.rstrip('.')}; {base[0].lower()}{base[1:]}" \
                    if plan.scenarios else base
            elif plan.kind == "tele":
                enc["reason"] = "Telehealth follow-up of chronic conditions"
            elif plan.kind == "consult":
                focus = [V.CONDITIONS[c].abbreviation for c in self.conds
                         if V.SPECIALTY_FOR_CONDITION.get(c) == plan.reason_key]
                enc["reason"] = f"{V.SPECIALTY_TITLES[plan.reason_key]} consultation for {_join(focus)}"
            elif plan.kind == "ed":
                enc["reason"] = _ED_REASONS[plan.reason_key]["label"]
            else:
                enc["reason"] = _INPATIENT_REASONS[plan.reason_key]["label"]

    # ------------------------------------------------------------------ notes
    def _notes(self) -> None:
        drafts: list[dict] = []
        for idx, plan in enumerate(self.plans):
            d = plan.enc["date"]
            if plan.kind in ("first", "followup", "tele"):
                drafts.append(self._compose("progress_note", plan, d, self._progress_note))
            elif plan.kind == "consult":
                drafts.append(self._compose("consult_note", plan, d, self._consult_note))
            elif plan.kind == "ed":
                drafts.append(self._compose("ed_note", plan, d, self._ed_note))
            elif plan.kind == "inpatient":
                admit = _date(d)
                if plan.reason_key in ("hf_exac", "afib_rvr", "copd_exac") and self.rng.random() < 0.5:
                    drafts.append(self._compose("consult_note", plan, _iso(admit + timedelta(days=1)),
                                                self._consult_note))
                drafts.append(self._compose("discharge_summary", plan, plan.enc["discharge_date"],
                                            self._discharge_summary))
            for kind in plan.imaging:
                drafts.append(self._compose("radiology_report", plan, d,
                                            lambda w, p, detail: self._radiology_report(w, p, w.kind, detail),
                                            kind=kind))
        drafts.sort(key=lambda n: (n["date"], _NOTE_ORDER[n["note_type"]]))
        for i, note in enumerate(drafts, start=1):
            note["note_id"] = f"{self.pid}-N{i:02d}"
        self.notes = [{"patient_id": n["patient_id"], "note_id": n["note_id"], "encounter_id": n["encounter_id"],
                       "date": n["date"], "note_type": n["note_type"], "author": n["author"], "title": n["title"],
                       "sections": n["sections"], "mentions": n["mentions"], "synthetic": True} for n in drafts]

    def _compose(self, note_type: str, plan: _Plan, day: str, writer: Callable, kind: str = "") -> dict:
        """Write a note, adjusting detail until it is NOTE_MIN_WORDS..NOTE_MAX_WORDS words long."""
        best: Optional[_NoteWriter] = None
        for detail in (3, 2, 1, 0, 4, 5):
            w = _NoteWriter(self, note_type, plan, day,
                            random.Random(f"note:{self.seed}:{self.pid}:{plan.enc['encounter_id']}:"
                                          f"{note_type}:{day}:{kind}"),
                            kind)
            writer(w, plan, detail)
            n_words = len(w.text().split())
            if NOTE_MIN_WORDS <= n_words <= NOTE_MAX_WORDS:
                best = w
                break
            if best is None or abs(n_words - 300) < abs(len(best.text().split()) - 300):
                best = w
        assert best is not None
        n_words = len(best.text().split())
        if not NOTE_MIN_WORDS <= n_words <= NOTE_MAX_WORDS:
            logger.warning("%s %s on %s has %d words", self.pid, note_type, day, n_words)
        return best.finish()

    # ---- shared section helpers
    def conds_on(self, day: str) -> list[str]:
        return [c for c in self.all_conds if _iso(self.onset[c]) <= day and (c != "cap" or _iso(self.onset[c]) < day)]

    def _pmh(self, w: "_NoteWriter", day: str) -> str:
        parts = []
        for c in self.conds_on(day):
            onset = self.onset[c]
            if c == "prior_mi":
                parts.append(f"{w.cond(c)} (NSTEMI {onset.year}, PCI to {self.mi_vessel})")
            elif c == "hfref":
                parts.append(f"{w.cond(c)} (LVEF {self.ef}%)")
            elif c == "cap":
                parts.append(f"{w.cond(c)} requiring admission {_iso(onset)}, resolved")
            elif c == "copd":
                parts.append(f"{w.cond(c)} (former smoker, {self.pack_years} pack-years)")
            elif c == "osa":
                parts.append(f"{w.cond(c)} on CPAP")
            elif w.rng.random() < 0.5:
                parts.append(f"{w.cond(c)} (dx {onset.year})")
            else:
                parts.append(w.cond(c))
        return _upper_first("; ".join(parts) + ".")

    def _short_pmh(self, w: "_NoteWriter", day: str, n: int = 3) -> str:
        return _join([w.cond(c) for c in self.conds_on(day) if c != "cap"][:n])

    def _med_section(self, w: "_NoteWriter", day: str, strictly_before: bool = False) -> str:
        meds = [m for m in self.active_meds(day) if not strictly_before or m["start_date"] < day]
        if not meds:
            return "None."
        return "\n".join(f"- {w.med(m)}" for m in meds)

    def _vitals_text(self, day: str) -> str:
        v = self.vitals[day]
        room = "on 2 L NC" if v["spo2"] < 90 else "on RA"
        return (f"BP {v['systolic']}/{v['diastolic']}, HR {v['heart_rate']}, RR {v['resp_rate']}, "
                f"T {v['temp_c']:.1f} C, SpO2 {v['spo2']}% {room}, Wt {v['weight_kg']:.1f} kg, BMI {v['bmi']:.1f}.")

    def _labs_section(self, w: "_NoteWriter", day: str, intro: str) -> str:
        rows = [r for r in self.labs if r["date"] == day]
        if not rows:
            return ""
        parts = []
        for panel in V.PANEL_ORDER:
            prow = [r for r in rows if r["panel"] == panel]
            if prow:
                label = {"HbA1c": "Glycemic", "Hepatic function panel": "LFTs", "Coagulation": "Coags",
                         "TSH": "Thyroid", "BNP": "Cardiac"}.get(panel, panel)
                parts.append(f"{label}: " + ", ".join(w.lab(r) for r in prow) + ".")
        return intro + "\n" + "\n".join(parts)

    def _recent_labs_text(self, w: "_NoteWriter", day: str) -> str:
        keys: list[str] = []
        for c, k in (("t2dm", "hba1c"), ("ckd3", "egfr"), ("hypothyroid", "tsh"), ("hfref", "bnp"), ("hld", "ldl")):
            if c in self.conds:
                keys.append(k)
        keys.append("creatinine")
        parts = []
        for k in dict.fromkeys(keys):
            row = self.latest_lab(k, day, before=True)
            if row:
                parts.append(f"{w.lab(row)} ({row['date']})")
        if not parts:
            return "No labs drawn today."
        return "No labs drawn today. Most recent: " + "; ".join(parts) + "."

    def _exam(self, w: "_NoteWriter", day: str, detail: int, extra: str = "") -> str:
        rng = w.rng
        has = self.conds.__contains__
        lines = [rng.choice(V.GENERAL_EXAM)]
        if detail >= 2:
            lines.append("HEENT: MMM, no JVD." if not has("hfref") else "Neck: JVP not elevated.")
        lines.append("CV: irregularly irregular, rate controlled, no murmur." if has("afib")
                     else rng.choice(("CV: RRR, no m/r/g.", "CV: regular rate and rhythm, normal S1/S2.")))
        if has("copd"):
            lines.append("Lungs: diminished breath sounds bilaterally, prolonged expiratory phase, no wheezes.")
        else:
            lines.append(rng.choice(("Lungs: CTAB.", "Lungs: clear to auscultation bilaterally, no wheezes.")))
        if detail >= 1:
            lines.append("Abd: soft, NT/ND.")
        ext = "Ext: no edema."
        if any(s.key == "ccb_edema" for s in w.plan.scenarios):
            ext = "Ext: 1+ bilateral ankle edema."
        elif has("hfref"):
            ext = "Ext: trace bilateral LE edema."
        elif has("oa"):
            ext = "Ext: bilateral knee crepitus, no effusion; no edema."
        lines.append(ext)
        if detail >= 3:
            lines.append(rng.choice(V.NEURO_EXAM))
        if detail >= 2 and (has("mdd") or has("gad")):
            lines.append(rng.choice(V.PSYCH_EXAM))
        if extra:
            lines.append(extra)
        return " ".join(lines)

    def _cond_phrase(self, w: "_NoteWriter", cond: str, day: str, kind: str) -> str:
        meds = self.meds_for(cond, day)
        options = [t for t in V.NOTE_PHRASES[cond][kind] if meds or "{meds}" not in t]
        if not options:
            return f"{w.cond(cond)}: no current pharmacotherapy."
        template = w.rng.choice(options)
        out = template
        if "{cond}" in out:
            out = out.replace("{cond}", w.cond(cond))
        if "{meds}" in out:
            out = out.replace("{meds}", _join([w.med(m) for m in meds]))
        return out[0].upper() + out[1:]

    def _status(self, w: "_NoteWriter", cond: str, day: str, enc_day: str) -> str:
        """Data-driven status sentence for a problem (labs, vitals)."""
        has_vitals = enc_day in self.vitals
        if cond == "t2dm":
            row = self.latest_lab("hba1c", day)
            if not row:
                return ""
            goal = "at goal <7.0" if row["value"] < 7.0 else "above goal <7.0"
            prev = self.latest_lab("hba1c", row["date"], before=True)
            when = "today" if row["date"] == enc_day else f"on {row['date']}"
            trend = f" (prior {w.lab(prev)} on {prev['date']})" if prev else ""
            return f"{w.lab(row)} {when}, {goal}{trend}."
        if cond == "htn":
            if not has_vitals:
                return "Home BP readings reported at goal."
            v = self.vitals[enc_day]
            goal = "at goal" if v["systolic"] < 130 and v["diastolic"] < 80 else "above goal <130/80"
            return f"BP {v['systolic']}/{v['diastolic']} today, {goal}."
        if cond in ("hld", "cad"):
            row = self.latest_lab("ldl", day)
            if not row:
                return ""
            target = 70 if "cad" in self.conds else 100
            goal = "at goal" if row["value"] < target else "above goal"
            return f"{w.lab(row)} ({row['date']}), {goal} (target <{target})."
        if cond == "ckd3":
            e, c = self.latest_lab("egfr", day), self.latest_lab("creatinine", day)
            if not e:
                return ""
            return f"{w.lab(e)}, {w.lab(c)} ({e['date']}), stable for stage 3a."
        if cond == "hfref":
            row = self.latest_lab("bnp", day)
            return f"LVEF {self.ef}%; {w.lab(row)} ({row['date']})." if row else f"LVEF {self.ef}%."
        if cond == "afib":
            score = self.cha2ds2_vasc(_date(day))
            inr = self.lab_at("inr", enc_day)
            extra = ""
            if inr is not None and not any(s.key == "warfarin_labile" for s in w.plan.scenarios):
                state = "therapeutic" if 2.0 <= inr["value"] <= 3.0 else "out of range"
                extra = f" {w.lab(inr)} today, {state}."
            return f"CHA2DS2-VASc {score}.{extra}"
        if cond == "hypothyroid":
            row = self.latest_lab("tsh", day)
            if not row:
                return ""
            return f"{w.lab(row)} ({row['date']}), {'within range' if not row['flag'] else 'out of range'}."
        if cond == "mdd":
            return f"PHQ-9 {self._phq(enc_day)} today."
        if cond == "gad":
            return f"GAD-7 {self._phq(enc_day, gad=True)} today."
        if cond == "obesity" and has_vitals:
            return f"BMI {self.vitals[enc_day]['bmi']:.1f} today."
        return ""

    def _phq(self, day: str, gad: bool = False) -> int:
        key = f"{'gad' if gad else 'phq'}:{day}"
        if key not in self.phq:
            self.phq[key] = random.Random(f"{self.seed}:{self.pid}:{key}").randint(5, 13)
        return self.phq[key]

    def cha2ds2_vasc(self, on: date) -> int:
        age = self._age_at(on)
        has = self.conds.__contains__
        return (has("hfref") + has("htn") + (2 if age >= 75 else 1 if age >= 65 else 0) + has("t2dm")
                + (has("cad") or has("prior_mi")) + (self.sex == "female"))

    def _interval_history(self, idx: int) -> str:
        if idx == 0:
            return ""
        prev = self.plans[idx - 1]
        if prev.kind == "inpatient":
            label = _INPATIENT_REASONS[prev.reason_key]["label"]
            return (f"Hospitalized {prev.enc['date']} to {prev.enc['discharge_date']} for {label}; "
                    f"recovering well since discharge.")
        if prev.kind == "ed":
            return f"Seen in the ED on {prev.enc['date']} for {_ED_REASONS[prev.reason_key]['label'].lower()}; " \
                   f"symptoms resolved."
        last = f"Last seen {prev.enc['date']}" + (" by telehealth" if prev.kind == "tele" else "")
        return f"{last}; no ED visits or hospitalizations since."

    def _opener(self, w: "_NoteWriter", day: str, visit: str) -> str:
        age = self._age_at(_date(day))
        return w.rng.choice(V.HPI_OPENERS).format(
            age=age, sex_short="F" if self.sex == "female" else "M", sex_letter="F" if self.sex == "female" else "M",
            sex_word="woman" if self.sex == "female" else "man", pmh=self._short_pmh(w, day), visit=visit,
            article="an" if str(age).startswith("8") or age in (11, 18) else "a")

    def _scenario_text(self, w: "_NoteWriter", s: V.SwitchScenario, template: str, day: str) -> str:
        old, new = self.scenario_meds[s.key]
        out = template
        if "{lab}" in out:
            out = out.replace("{lab}", w.lab(self.lab_at(s.lab_override[0], day)))
        for token, value in (("{from}", lambda: w.med(old)), ("{to}", lambda: w.med(new)),
                             ("{from_name}", lambda: w.med(old, full=False)),
                             ("{to_name}", lambda: w.med(new, full=False)), ("{cond}", lambda: w.cond(s.condition))):
            if token in out:
                out = out.replace(token, value())
        return out

    # ---- note writers
    def _progress_note(self, w: "_NoteWriter", plan: _Plan, detail: int) -> None:
        rng = w.rng
        day = plan.enc["date"]
        idx = self.plans.index(plan)
        tele = plan.kind == "tele"
        conds = [c for c in self.conds_on(day) if c != "cap"]
        scen = plan.scenarios[0] if plan.scenarios else None
        focus = ([scen.condition] if scen else []) + [c for c in rng.sample(conds, len(conds))
                                                       if not scen or c != scen.condition]

        if scen:
            cc = scen.complaint
        elif tele:
            cc = f"Telehealth follow-up of {w.cond(focus[0])}."
        elif plan.kind == "first":
            cc = rng.choice(("Annual visit; chronic disease management.", "Establishing care; annual physical."))
        else:
            cc = rng.choice((f"Follow-up of {w.cond(focus[0])} and {w.cond(focus[1])}." if len(focus) > 1
                             else f"Follow-up of {w.cond(focus[0])}.",
                             "Routine chronic disease follow-up.", "3-month follow-up of chronic conditions."))
        w.add("CHIEF COMPLAINT", cc)

        visit = {"first": "an annual visit", "tele": "a video telehealth follow-up"}.get(
            plan.kind, rng.choice(("routine follow-up", "a 3-month follow-up", "chronic disease management")))
        hpi = [self._opener(w, day, visit)]
        if tele:
            hpi.append("Visit conducted by secure video; patient at home with no home vitals available today.")
        interval = self._interval_history(idx)
        if interval:
            hpi.append(interval)
        if scen:
            hpi.append(self._scenario_text(w, scen, scen.hpi, day))
        n_hpi = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}[detail]
        for c in focus[: n_hpi + (1 if scen else 0)]:
            if scen and c == scen.condition:
                continue
            hpi.append(self._cond_phrase(w, c, day, "hpi"))
        w.add("HISTORY OF PRESENT ILLNESS", " ".join(hpi))
        w.add("PAST MEDICAL HISTORY", self._pmh(w, day))
        if detail >= 4 or plan.kind == "first":
            w.add("SOCIAL HISTORY", self.social)
        w.add("MEDICATIONS", self._med_section(w, day))
        w.add("ALLERGIES", self._allergy_text())
        if tele:
            w.add("PHYSICAL EXAM", "Limited video exam: well-appearing, speaking in full sentences, no increased "
                                   "work of breathing, normal affect.")
        else:
            w.add("VITAL SIGNS", self._vitals_text(day))
            w.add("PHYSICAL EXAM", self._exam(w, day, detail))
        labs = self._labs_section(w, day, f"Drawn today ({day}):")
        w.add("LABS", labs or self._recent_labs_text(w, day))

        style = rng.choice(("num", "hash", "dash"))
        items: list[str] = []
        n_plan = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}[detail]
        addressed = focus[:n_plan]
        for c in addressed:
            if scen and c == scen.condition:
                body = self._scenario_text(w, scen, scen.plan, day)
            else:
                status = self._status(w, c, day, day)
                body = (status + " " if status else "") + self._cond_phrase(w, c, day, "plan")
            items.append(_upper_first(f"{w.cond(c)}: {body}"))
        others = [c for c in conds if c not in addressed]
        if others:
            items.append(f"Other problems ({_join([w.cond(c) for c in others])}) stable; continue current regimen.")
        w.add("ASSESSMENT AND PLAN", _format_items(items, style) + "\n" + rng.choice(V.PLAN_CLOSERS))

    def _ed_note(self, w: "_NoteWriter", plan: _Plan, detail: int) -> None:
        day = plan.enc["date"]
        info = _ED_REASONS[plan.reason_key]
        conds = [c for c in self.conds_on(day) if c != "cap"]
        w.add("CHIEF COMPLAINT", info["cc"])
        hpi = [self._opener(w, day, "evaluation in the ED"), info["hpi"]]
        n_ctx = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}[detail]
        presenting = {"asthma_exac": "asthma", "copd_flare": "copd", "palpitations": "afib", "chest_pain": "cad",
                      "knee_fall": "oa", "epigastric": "gerd"}.get(plan.reason_key)
        context = [c for c in conds if c != presenting]
        for c in w.rng.sample(context, len(context))[:n_ctx]:
            hpi.append(self._cond_phrase(w, c, day, "hpi"))
        w.add("HISTORY OF PRESENT ILLNESS", " ".join(hpi))
        w.add("PAST MEDICAL HISTORY", self._pmh(w, day))
        w.add("MEDICATIONS", "Home medications:\n" + self._med_section(w, day, strictly_before=True))
        w.add("ALLERGIES", self._allergy_text())
        w.add("VITAL SIGNS", "On arrival: " + self._vitals_text(day))
        extra = {"knee_fall": "Right knee: mild effusion, tenderness over medial joint line, ligaments stable.",
                 "asthma_exac": "Diffuse expiratory wheezes, mild accessory muscle use.",
                 "copd_flare": "Scattered expiratory wheezes."}.get(plan.reason_key, "")
        w.add("PHYSICAL EXAM", self._exam(w, day, detail, extra))
        labs = self._labs_section(w, day, f"ED labs ({day}):")
        if labs:
            w.add("LABS", labs)
        course = info["course"]
        courses = self.course_meds[plan.enc["encounter_id"]]
        if courses:
            course += " Discharged on " + _join([w.med(m) for m in courses]) + \
                      f" for {(_date(courses[0]['end_date']) - _date(courses[0]['start_date'])).days} days."
        w.add("ED COURSE", course)
        items = [f"{info['dx']}: {info['plan']}"]
        chronic = [w.cond(c) for c in conds[: 2 + detail]]
        if chronic:
            items.append(f"Chronic {_join(chronic)}: continue home regimen.")
        items.append("Disposition: discharged home in stable condition; PCP follow-up within 1 week; return "
                     "precautions reviewed.")
        w.add("ASSESSMENT AND PLAN", _format_items(items, "num"))

    def _discharge_summary(self, w: "_NoteWriter", plan: _Plan, detail: int) -> None:
        admit, dc = plan.enc["date"], plan.enc["discharge_date"]
        info = _INPATIENT_REASONS[plan.reason_key]
        reason = plan.reason_key
        w.add("CHIEF COMPLAINT", info["cc"])
        w.add("HISTORY OF PRESENT ILLNESS", self._opener(w, admit, "evaluation and was admitted") + " " + info["hpi"])
        if detail >= 2:
            w.add("PAST MEDICAL HISTORY", self._pmh(w, admit))
        v = self.vitals[admit]
        los = (_date(dc) - _date(admit)).days
        course = [f"Admitted {admit}, discharged {dc} (LOS {los} days)."]
        scen = plan.scenarios[0] if plan.scenarios else None
        if reason == "hf_exac":
            old, new = self.scenario_meds[scen.key]
            course.append(f"On arrival SpO2 {v['spo2']}% and weight {v['weight_kg']:.1f} kg; "
                          f"{w.lab(self.lab_at('bnp', admit))}.")
            course.append(f"CXR with pulmonary vascular congestion and small bilateral effusions. Diuresed with IV "
                          f"{w.med(old, full=False)} with net negative fluid balance of about "
                          f"{w.rng.randint(3, 6)} L; DOE and edema improved.")
            course.append(f"{w.lab(self.lab_at('creatinine', admit))} on admission, monitored daily during "
                          f"diuresis. Home GDMT continued; {_lower_first(self._scenario_text(w, scen, scen.plan, dc))}")
        elif reason == "copd_exac":
            pred, azi = self.course_meds[plan.enc["encounter_id"]]
            course.append(f"SpO2 {v['spo2']}% on RA at presentation, placed on 2 L NC. CXR hyperinflated without "
                          f"consolidation. {w.lab(self.lab_at('wbc', admit))}.")
            course.append(f"Treated with scheduled nebulized bronchodilators, {w.med(pred)} and {w.med(azi)}; "
                          f"supplemental O2 weaned to RA by hospital day {max(1, los - 1)}.")
        elif reason == "cap":
            (abx,) = self.course_meds[plan.enc["encounter_id"]]
            course.append(f"Febrile to {v['temp_c']:.1f} C with SpO2 {v['spo2']}% on arrival; "
                          f"{w.lab(self.lab_at('wbc', admit))}. CXR with RLL consolidation.")
            course.append(f"Started on empiric IV antibiotics, defervesced within 48 hours and transitioned to "
                          f"{w.med(abx)} to complete a 7-day course.")
        elif reason == "afib_rvr":
            anticoag = [m for m in self.active_meds(admit) if V.MEDICATIONS[m["name"]].drug_class == "anticoagulant"]
            course.append(f"HR {v['heart_rate']} on arrival. Rate controlled with IV diltiazem, then "
                          f"{_lower_first(self._scenario_text(w, scen, scen.plan, dc))}")
            tsh = self.lab_at("tsh", admit)
            if tsh:
                course.append(f"{w.lab(tsh)}, {'within range' if not tsh['flag'] else 'abnormal'}.")
            if anticoag:
                course.append(f"Anticoagulation continued with {w.med(anticoag[0])}.")
            course.append("TTE deferred to outpatient follow-up.")
        if detail >= 1:
            course.append("Ambulating independently and tolerating diet at discharge.")
        w.add("HOSPITAL COURSE", " ".join(course))
        labs = self._labs_section(w, admit, f"Admission labs ({admit}):")
        if labs and detail >= 1:
            w.add("LABS", labs)
        dx = [info["dx"]] + [_upper_first(w.cond(c)) for c in self.conds_on(admit)[: 1 + detail]
                             if c not in ("cap",) and not (reason == "hf_exac" and c == "hfref")
                             and not (reason == "copd_exac" and c == "copd")
                             and not (reason == "afib_rvr" and c == "afib")]
        w.add("DISCHARGE DIAGNOSES", _format_items(dx, "num"))
        lines = []
        for m in self.active_meds(dc):
            tag = ""
            if m["start_date"] >= admit:
                tag = " (NEW)"
                if scen:
                    old, new = self.scenario_meds[scen.key]
                    if new is m:
                        tag = f" (CHANGED: increased from {old['dose']})"
                        w.med_id_only(old)
            lines.append(f"- {w.med(m)}{tag}")
        w.add("DISCHARGE MEDICATIONS", "\n".join(lines) or "None.")
        w.add("ALLERGIES", self._allergy_text())
        follow = {"hf_exac": "PCP within 7 days with BMP; cardiology clinic in 2 weeks; daily weights.",
                  "copd_exac": "PCP within 7 days; pulmonology clinic in 4 weeks.",
                  "cap": "PCP within 7-10 days; repeat CXR in 6-8 weeks.",
                  "afib_rvr": "Cardiology clinic in 2 weeks; PCP within 7 days."}[reason]
        w.add("FOLLOW-UP", follow)

    def _consult_note(self, w: "_NoteWriter", plan: _Plan, detail: int) -> None:
        day = w.day
        inpatient = plan.kind == "inpatient"
        specialty = {"hf_exac": "cardiology", "afib_rvr": "cardiology", "copd_exac": "pulmonology"}.get(
            plan.reason_key, plan.reason_key) if inpatient else plan.reason_key
        enc_day = plan.enc["date"]
        focus = [c for c in self.conds if V.SPECIALTY_FOR_CONDITION.get(c) == specialty] or self.conds[:1]
        if inpatient:
            reason = (f"Inpatient {V.SPECIALTY_TITLES[specialty].lower()} input on "
                      f"{_INPATIENT_REASONS[plan.reason_key]['dx']}.")
        else:
            reason = f"Referred by {self.pcp} for {_join([w.cond(c) for c in focus])}."
        w.add("REASON FOR CONSULTATION", reason)
        hpi = [self._opener(w, day, "inpatient consultation" if inpatient else "specialty consultation")]
        if inpatient:
            hpi.append(_INPATIENT_REASONS[plan.reason_key]["hpi"])
        for c in focus[: 1 + detail]:
            hpi.append(self._cond_phrase(w, c, day, "hpi"))
        w.add("HISTORY OF PRESENT ILLNESS", " ".join(hpi))
        w.add("PAST MEDICAL HISTORY", self._pmh(w, day))
        w.add("SOCIAL HISTORY", self.social)
        w.add("MEDICATIONS", self._med_section(w, day))
        w.add("ALLERGIES", self._allergy_text())
        if enc_day in self.vitals:
            w.add("VITAL SIGNS", ("On admission: " if inpatient else "") + self._vitals_text(enc_day))
        labs = self._labs_section(w, enc_day, f"Labs ({enc_day}):")
        trend = self._trend_text(w, specialty, day) if detail >= 1 else ""
        w.add("LABS", "\n".join(x for x in (labs or self._recent_labs_text(w, day), trend) if x))
        recs = []
        for c in focus:
            status = self._status(w, c, day, enc_day)
            recs.append(_upper_first(f"{w.cond(c)}: {(status + ' ') if status else ''}"
                                     f"{self._cond_phrase(w, c, day, 'plan')}"))
        recs += {
            "nephrology": ["Check urine albumin-to-creatinine ratio; BMP every 6 months.",
                           "Avoid NSAIDs and IV contrast when possible.",
                           "Renally dose new medications; review OTC supplements at each visit.",
                           "Nephrology follow-up in 6 months, sooner if eGFR declines."],
            "endocrinology": ["Continue home glucose monitoring; bring log to next visit.",
                              "Annual dilated eye exam and foot exam; urine albumin screening.",
                              "Endocrinology follow-up in 4 months with repeat labs."],
            "cardiology": ["Low-sodium diet; daily weights.", "Cardiology follow-up in 3 months.",
                           "Continue to titrate GDMT as BP and renal function allow.",
                           "Call for chest pain, syncope or weight gain over 1 kg in a day."],
            "pulmonology": ["Full PFTs at next visit; continue inhaler technique review.",
                            "Pneumococcal and annual influenza vaccines up to date.",
                            "Pulmonary rehab referral placed; written action plan provided."],
            "behavioral_health": ["Continue weekly psychotherapy.", "Crisis resources reviewed; no SI today.",
                                  "Sleep hygiene and behavioral activation strategies reviewed.",
                                  "Behavioral health follow-up in 6 weeks; call sooner if symptoms worsen."],
        }[specialty][: 1 + detail]
        if detail >= 4:
            recs.append("Discussed findings and plan with the patient, who agrees; all questions answered.")
        w.add("ASSESSMENT AND PLAN", "Recommendations:\n" + _format_items(recs, "num") +
              "\nThank you for this consult; will follow.")

    def _trend_text(self, w: "_NoteWriter", specialty: str, day: str) -> str:
        key = {"nephrology": "egfr", "endocrinology": "hba1c" if "t2dm" in self.conds else "tsh",
               "cardiology": "bnp"}.get(specialty)
        if not key:
            return ""
        rows = sorted((r for r in self.labs if r["test_key"] == key and r["date"] <= day), key=lambda r: r["date"])
        if len(rows) < 2:
            return ""
        return "Trend: " + "; ".join(f"{w.lab(r)} on {r['date']}" for r in rows[-4:]) + "."

    def _radiology_report(self, w: "_NoteWriter", plan: _Plan, kind: str, detail: int) -> None:
        day = plan.enc["date"]
        age = self._age_at(_date(day))
        sx = "F" if self.sex == "female" else "M"
        reason = plan.reason_key if plan.kind in ("ed", "inpatient") else ""
        prior = [p.enc["date"] for p in self.plans if kind in p.imaging and p.enc["date"] < day]
        comparison = f"Prior study of the same type dated {max(prior)}." if prior else "None available."
        has = self.conds.__contains__
        if kind == "cxr":
            variant = {"hf_exac": "congestion", "cap": "cap", "copd_exac": "copd", "copd_flare": "copd"}.get(
                reason, "normal")
            indication = {
                "congestion": f"SOB and LE edema; history of {w.cond('hfref')}.",
                "cap": "Fever, productive cough and pleuritic chest pain; evaluate for pneumonia.",
                "copd": f"Increased dyspnea; history of {w.cond('copd')}.",
                "normal": {"chest_pain": "Chest pain.", "palpitations": "Palpitations and DOE.",
                           "asthma_exac": "Wheezing; history of "
                                          f"{w.cond('asthma') if has('asthma') else 'reactive airways'}.",
                           "afib_rvr": f"{w.cond('afib') if has('afib') else 'Arrhythmia'} with RVR, DOE."}.get(
                    reason, "SOB."),
            }[variant]
            heart = ("Cardiomegaly with cardiothoracic ratio approximately 0.6." if has("hfref")
                     else "Heart size within normal limits.")
            lungs = {
                "congestion": "Cephalization of the pulmonary vasculature with diffuse interstitial opacities and "
                              "peribronchial cuffing, consistent with interstitial edema. Kerley B lines at the bases.",
                "cap": "Focal airspace consolidation in the right lower lobe with air bronchograms, obscuring the "
                       "medial right hemidiaphragm. The left lung is clear.",
                "copd": "Hyperinflated lungs with flattened hemidiaphragms and increased retrosternal clear space. "
                        "No focal consolidation.",
                "normal": "Lungs are clear without focal consolidation, edema or mass.",
            }[variant]
            pleura = ("Small bilateral pleural effusions, right greater than left, with blunting of the "
                      "costophrenic angles. No pneumothorax." if variant == "congestion"
                      else "No pleural effusion or pneumothorax.")
            impression = {
                "congestion": [f"Interstitial pulmonary edema and small bilateral effusions, compatible with "
                               f"decompensated {w.cond('hfref')}.", "Stable cardiomegaly."],
                "cap": ["Right lower lobe consolidation compatible with pneumonia in the appropriate clinical "
                        "setting.", "Recommend follow-up radiograph in 6-8 weeks to document resolution."],
                "copd": [f"Hyperinflation compatible with known {w.cond('copd')}.", "No acute cardiopulmonary "
                         "process."],
                "normal": ["No acute cardiopulmonary process."] + (["Cardiomegaly."] if has("hfref") else []),
            }[variant]
            w.add("EXAM", "XR chest, PA and lateral (2 views)." if plan.kind != "inpatient"
                  else "XR chest, single portable AP view.")
            w.add("CLINICAL INDICATION", f"{age}{sx}. {indication}")
            w.add("TECHNIQUE", "Frontal and lateral radiographs of the chest were obtained. Image quality is adequate "
                               "with satisfactory inspiration and no significant rotation."
                  if plan.kind != "inpatient" else
                  "Single portable upright AP radiograph of the chest was obtained at the bedside. Positioning and "
                  "inspiratory effort are adequate for interpretation.")
            w.add("COMPARISON", comparison)
            w.add("FINDINGS", f"Lines and tubes: none. Airways: trachea midline; central airways patent. Heart: "
                              f"{heart} Mediastinum: normal mediastinal and hilar contours; aorta mildly tortuous "
                              f"with atherosclerotic calcification of the arch. Lungs: {lungs} Pleura: {pleura} "
                              f"Bones and soft tissues: mild degenerative changes of the thoracic spine; no acute "
                              f"osseous abnormality. Upper abdomen: no free air beneath the diaphragm; visualized "
                              f"bowel gas pattern nonspecific.")
            w.add("IMPRESSION", _format_items(impression, "num"))
        elif kind == "knee_xr":
            w.add("EXAM", "XR right knee, 3 views (AP, lateral, sunrise).")
            w.add("CLINICAL INDICATION", f"{age}{sx}. " + ("Fall onto right knee with pain." if reason == "knee_fall"
                                                         else f"Chronic knee pain; history of {w.cond('oa')}."))
            w.add("TECHNIQUE", "Three radiographic views of the right knee were obtained with weight bearing on the "
                               "AP view where tolerated.")
            w.add("COMPARISON", comparison)
            w.add("FINDINGS", "No acute fracture or dislocation. Moderate medial tibiofemoral joint space narrowing "
                              "with marginal osteophytes and subchondral sclerosis. Mild lateral compartment and "
                              "patellofemoral narrowing with patellar osteophytes. "
                              + ("Small suprapatellar joint effusion. " if reason == "knee_fall" else
                                 "No joint effusion. ")
                              + "No lytic or blastic lesion. Vascular calcifications in the popliteal region. Soft "
                                "tissues otherwise unremarkable without radiopaque foreign body.")
            w.add("IMPRESSION", _format_items(["No acute fracture or dislocation.",
                                               f"Moderate tricompartmental degenerative change, medial predominant, "
                                               f"consistent with {w.cond('oa')}."]
                                              + (["Small joint effusion."] if reason == "knee_fall" else []), "num"))
        elif kind == "tte":
            regional = (f"Akinesis of the {'anterior wall and apex' if self.mi_vessel == 'LAD' else 'inferior wall'} "
                        f"with hypokinesis elsewhere." if has("prior_mi") else "Global hypokinesis.")
            w.add("EXAM", "Transthoracic echocardiogram (TTE), complete with Doppler.")
            w.add("CLINICAL INDICATION", f"{age}{sx}. Reassessment of LV function; history of {w.cond('hfref')}.")
            w.add("TECHNIQUE", "Two-dimensional, M-mode, color and spectral Doppler imaging from standard windows. "
                               "Image quality fair. No contrast agent used.")
            w.add("COMPARISON", comparison)
            w.add("FINDINGS", f"Left ventricle: mildly dilated, LVEDD 6.1 cm, normal wall thickness. {regional} LVEF "
                              f"estimated {self.ef}% by biplane method. Grade II diastolic dysfunction. Right "
                              f"ventricle: normal size with mildly reduced systolic function (TAPSE 1.6 cm). Atria: "
                              f"left atrium moderately dilated; right atrium normal. Valves: trileaflet aortic valve "
                              f"without stenosis; mild-to-moderate functional mitral regurgitation; mild tricuspid "
                              f"regurgitation with estimated RVSP 38 mmHg. IVC normal in size with normal "
                              f"respiratory variation. Pericardium: no effusion.")
            w.add("IMPRESSION", _format_items([f"Reduced LV systolic function, LVEF {self.ef}%, consistent with "
                                               f"known {w.cond('hfref')}.", "Grade II diastolic dysfunction.",
                                               "Mild-to-moderate functional mitral regurgitation."], "num"))
        elif kind == "ldct":
            contrast = ("No IV contrast administered (documented iodinated contrast allergy)."
                        if any(a["substance"] == "iodinated contrast" for a in self.allergy_rows)
                        else "No IV contrast administered.")
            w.add("EXAM", "CT chest without contrast, low-dose lung cancer screening.")
            w.add("CLINICAL INDICATION", f"{age}{sx}. Annual lung cancer screening; former smoker, "
                                         f"{self.pack_years} pack-years; history of {w.cond('copd')}.")
            w.add("TECHNIQUE", f"Low-dose helical CT of the chest from the thoracic inlet to the adrenal glands with "
                               f"thin-section reconstructions. {contrast}")
            w.add("COMPARISON", comparison)
            w.add("FINDINGS", "Lungs: moderate centrilobular emphysema, upper lobe predominant. A 4 mm solid "
                              "nodule in the right upper lobe is benign in appearance. No consolidation or mass. "
                              "Airways: mild bronchial wall thickening; central airways patent. Pleura: no effusion "
                              "or pneumothorax. Mediastinum: no lymphadenopathy by size criteria. Heart: normal "
                              "size; moderate coronary artery calcification. Upper abdomen: unremarkable within the "
                              "limits of a noncontrast study. Bones: degenerative changes of the thoracic spine.")
            w.add("IMPRESSION", _format_items(["Lung-RADS 2: benign-appearing 4 mm right upper lobe nodule; "
                                               "continue annual screening.",
                                               f"Centrilobular emphysema compatible with known {w.cond('copd')}.",
                                               "Coronary artery calcification."], "num"))
        elif kind == "renal_us":
            w.add("EXAM", "US retroperitoneum (renal), complete.")
            w.add("CLINICAL INDICATION", f"{age}{sx}. Evaluation of {w.cond('ckd3')}.")
            w.add("TECHNIQUE", "Grayscale and color Doppler sonographic images of both kidneys and the urinary "
                               "bladder were obtained.")
            w.add("COMPARISON", comparison)
            right, left = w.rng.uniform(9.4, 10.4), w.rng.uniform(9.6, 10.6)
            w.add("FINDINGS", f"Right kidney: {right:.1f} cm in length with mildly increased cortical echogenicity "
                              f"and mild cortical thinning. No hydronephrosis, calculus or solid mass. Left kidney: "
                              f"{left:.1f} cm in length with similar increased cortical echogenicity. A 1.2 cm "
                              f"simple cyst in the lower pole. No hydronephrosis or calculus. Bladder: partially "
                              f"distended, normal wall thickness, no intraluminal lesion. Color Doppler demonstrates "
                              f"normal intrarenal flow bilaterally.")
            w.add("IMPRESSION", _format_items(["Bilateral increased renal cortical echogenicity with mild cortical "
                                               f"thinning, compatible with medical renal disease in the setting of "
                                               f"{w.cond('ckd3')}.", "No hydronephrosis.",
                                               "Simple left lower pole renal cyst (Bosniak 1), no follow-up needed."],
                                              "num"))
        self._radiology_tail(w, kind)

    _RAD_RECS = {
        "cxr": "Correlate clinically. Follow-up imaging as clinically indicated; no additional imaging is required "
               "on the basis of this study alone.",
        "knee_xr": "Orthopedic or sports medicine follow-up as clinically indicated. Consider weight-bearing "
                   "standing views for longitudinal comparison.",
        "tte": "Continue guideline-directed therapy per the treating team. Repeat TTE in 3-6 months after "
               "medication optimization to reassess LV function.",
        "ldct": "Continue annual low-dose CT screening. Smoking cessation resources remain available. Coronary "
                "calcification may warrant cardiovascular risk assessment by the primary team.",
        "renal_us": "No urologic follow-up needed. Continue nephrology care; findings support chronic parenchymal "
                    "change rather than an obstructive process.",
    }

    def _radiology_tail(self, w: "_NoteWriter", kind: str) -> None:
        tech = w.sections[2]
        tech[1] += (" Protocol and radiation dose per departmental standards. Images were reviewed on a diagnostic "
                    "workstation.")
        w.add("RECOMMENDATIONS", self._RAD_RECS[kind])
        w.add("ATTESTATION", "I have personally reviewed the images and agree with the findings above. "
                             "Electronically signed by the attending radiologist.")

    def radiologist(self, kind: str, day: str) -> str:
        options = V.PROVIDERS["radiology"]
        return options[(self.index + len(kind) + int(day[-2:])) % len(options)]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _format_items(items: list[str], style: str) -> str:
    if style == "hash":
        return "\n".join(f"# {x}" for x in items)
    if style == "dash":
        return "\n".join(f"- {x}" for x in items)
    return "\n".join(f"{i}. {x}" for i, x in enumerate(items, start=1))


class _NoteWriter:
    """Accumulates note sections and records exactly which meds and labs the text names."""

    _TITLES = {"progress_note": "Primary Care Progress Note", "ed_note": "Emergency Department Note",
               "discharge_summary": "Discharge Summary"}
    _RADIOLOGY_TITLES = {"cxr": "Radiology Report: Chest X-ray", "knee_xr": "Radiology Report: Right Knee X-ray",
                         "tte": "Radiology Report: Transthoracic Echocardiogram", "ldct": "Radiology Report: CT Chest",
                         "renal_us": "Radiology Report: Renal Ultrasound"}

    def __init__(self, pb: _PatientBuilder, note_type: str, plan: _Plan, day: str, rng: random.Random,
                 kind: str = "") -> None:
        self.pb, self.note_type, self.plan, self.day, self.rng, self.kind = pb, note_type, plan, day, rng, kind
        self.sections: list[list[str]] = []
        self._med_ids: list[str] = []
        self._labs: list[list[str]] = []

    def cond(self, key: str) -> str:
        return V.CONDITIONS[key].abbreviation

    def med(self, rec: dict, full: bool = True) -> str:
        self.med_id_only(rec)
        return render_medication(rec) if full else rec["name"]

    def med_id_only(self, rec: dict) -> None:
        if rec["med_id"] not in self._med_ids:
            self._med_ids.append(rec["med_id"])

    def lab(self, row: dict) -> str:
        key = [row["test_key"], row["date"]]
        if key not in self._labs:
            self._labs.append(key)
        return _PatientBuilder.lab_text(row)

    def add(self, name: str, text: str) -> None:
        self.sections.append([name, text.strip()])

    def text(self) -> str:
        return render_note_text({"sections": self.sections})

    def finish(self) -> dict:
        pb, plan = self.pb, self.plan
        text = self.text()
        if self.note_type == "radiology_report":
            title, author = self._RADIOLOGY_TITLES[self.kind], pb.radiologist(self.kind, self.day)
        elif self.note_type == "consult_note":
            specialty = ({"hf_exac": "cardiology", "afib_rvr": "cardiology", "copd_exac": "pulmonology"}
                         .get(plan.reason_key) if plan.kind == "inpatient" else plan.reason_key)
            title = f"{V.SPECIALTY_TITLES[specialty]} Consult Note"
            author = plan.enc["provider"] if plan.kind != "inpatient" else V.PROVIDERS[specialty][0]
        else:
            title = self._TITLES[self.note_type]
            if plan.kind == "tele":
                title = "Telehealth Progress Note"
            author = plan.enc["provider"]
        mentions = {
            "conditions": mentioned_conditions(text, pb.all_conds),
            "medications": list(self._med_ids),
            "labs": [list(x) for x in self._labs],
        }
        return {"patient_id": pb.pid, "note_id": "", "encounter_id": plan.enc["encounter_id"], "date": self.day,
                "note_type": self.note_type, "author": author, "title": title, "sections": self.sections,
                "mentions": mentions}

