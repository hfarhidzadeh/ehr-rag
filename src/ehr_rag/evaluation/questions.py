"""Gold evaluation questions built from the synthetic ground truth (PLAN.md section 5).

Per patient (deterministic from ``seed``; ``qid = f"{pid}-{category}-{n}"``):

* ``fact_lookup`` x2 - two different kinds out of: current dose of an active med, reaction to an allergen,
  reason a med was discontinued, a lab value on a given date. Keywords are the values as rendered in the
  documents; gold = the structured doc holding the fact plus notes whose ``mentions`` name it (and that
  actually contain the value; for discontinuations: the note dated ``end_date`` that names the med).
* ``temporal`` - "most recent <lab>" for a test measured at least twice; keyword = the latest value only.
* ``semantic_synonym`` - phrased only with a condition's ``lay_terms``: meds for it, when it was diagnosed,
  or whether the patient has it. The question never contains the abbreviation, display name or note terms.
* ``verbatim`` (positive control, needs ``chunks``) - quotes a >=8-word sentence of a note chunk that occurs
  in no other document of the patient; gold = that chunk's doc; keywords = its 2 rarest words.
* ``absent_fact`` (negative control) - a med, allergen or lab that occurs nowhere in the patient's documents.
* ``cross_patient`` (negative control) - another patient's distinctive med or allergen, absent from every
  document of the asked patient; ``metadata.source_patient_id`` names the other patient.

``gold_doc_ids[0]`` is always the canonical document (the structured doc or lab panel holding the fact, the
problem list for onset/has questions, or the quoted note); later entries are supporting notes.
Negative questions reuse the positive question templates, so their wording does not give them away.
``build_questions`` validates the result (``validate_questions``) and raises ``ValueError`` on any violation.
Absence is checked by a case-insensitive, word-boundary scan of every document text of the asked patient.
"""
from __future__ import annotations

import itertools
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ehr_rag.evaluation.metrics import answer_correct
from ehr_rag.ingest.chunking import split_sentences
from ehr_rag.synth import vocab as V
from ehr_rag.synth.generator import SyntheticDataset, format_lab_value, render_medication
from ehr_rag.types import (
    NEGATIVE_CATEGORIES,
    POSITIVE_CATEGORIES,
    Chunk,
    Document,
    EvalQuestion,
    allergies_doc_id,
    lab_panel_doc_id,
    medications_doc_id,
    problems_doc_id,
)

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES
FACT_KINDS: tuple[str, ...] = ("med_dose", "allergy_reaction", "stop_reason", "lab_value")
VERBATIM_MIN_WORDS = 8
VERBATIM_MAX_JACCARD = 0.6   # max token overlap with any sentence of another doc

# Templates shared by positive and negative questions.
Q_MED_DOSE = "What dose of {name} is the patient currently taking?"
Q_ALLERGY = "What reaction does the patient have to {substance}?"
Q_STOP = "Why was {name} {dose} discontinued?"
Q_LAB_ON = "What was the {lab} result on {date}?"
Q_LAB_LATEST = "What is the most recent {lab} result?"
Q_SYN_MEDS = "What medications is the patient taking for their {lay}?"
Q_SYN_ONSET = "When was the patient's {lay} first diagnosed?"
Q_SYN_HAS = "Does the patient have a history of {lay}?"
Q_VERBATIM = 'Where in the chart does it say: "{sentence}"?'

# How a lab is named in a question (lower-case where natural).
LAB_QUESTION_NAMES: dict[str, str] = {
    "hba1c": "HbA1c", "glucose": "glucose", "creatinine": "creatinine", "egfr": "eGFR", "bun": "BUN",
    "potassium": "potassium", "sodium": "sodium", "ldl": "LDL cholesterol", "hdl": "HDL cholesterol",
    "triglycerides": "triglycerides", "total_cholesterol": "total cholesterol", "tsh": "TSH", "bnp": "BNP",
    "hemoglobin": "hemoglobin", "wbc": "WBC count", "platelets": "platelet count", "alt": "ALT", "ast": "AST",
    "inr": "INR",
}

# Extra answer keywords for "does the patient have <lay term>" (medical wording an answer may use).
MEDICAL_TERMS: dict[str, tuple[str, ...]] = {
    "t2dm": ("type 2 diabetes", "diabetes mellitus"), "htn": ("hypertension",), "hld": ("hyperlipidemia",),
    "ckd3": ("chronic kidney disease", "renal insufficiency"),
    "cad": ("coronary artery disease", "atherosclerotic heart disease"),
    "prior_mi": ("myocardial infarction", "NSTEMI"), "hfref": ("heart failure", "reduced ejection fraction"),
    "afib": ("atrial fibrillation",), "copd": ("chronic obstructive pulmonary disease",), "asthma": ("asthma",),
    "hypothyroid": ("hypothyroidism",), "mdd": ("major depressive disorder",),
    "gad": ("generalized anxiety disorder",), "gerd": ("gastroesophageal reflux", "gastro-esophageal reflux"),
    "oa": ("osteoarthritis",), "osa": ("obstructive sleep apnea",), "obesity": ("obesity",), "cap": ("pneumonia",),
}

# Distinctive terms that, when present in a discontinue reason, become answer keywords.
REASON_TERMS: tuple[str, ...] = (
    "dry cough", "ACE inhibitor", "ankle edema", "hyponatremia", "hyperkalemia", "myalgias", "hypoglycemia",
    "GI intolerance", "diarrhea", "labile INR", "AFib with RVR", "HFrEF exacerbation", "persistent DOE",
    "vivid dreams", "neuropsychiatric", "partial response", "sexual side effects", "nausea", "insomnia",
    "symptoms controlled", "stepped down", "dyspepsia", "titration schedule", "uptitrated", "above goal",
    "day course",
)
_REASON_LAB_RE = re.compile(r"\(([A-Za-z0-9]+ \d+(?:\.\d+)?)\)")

# Conditions excluded from the "medications for <lay term>" template (no meds or awkward wording).
_NO_MEDS_TEMPLATE = frozenset({"osa", "obesity", "cap"})

_WORD_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as into about than then so is are was were be
    been being has have had do does did this that these those it its there here what which who when where why
    how no not without per over under after before since while patient patients today daily continue
    continued reports denies noted also well recent""".split()
)
_WORD_RE = re.compile(r"(?<![\w-])[A-Za-z]+(?:-[A-Za-z]+)*(?![\w-])")   # whole letter-only words


# --------------------------------------------------------------------------- text helpers


def _term_re(term: str) -> re.Pattern:
    return re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])", re.IGNORECASE)


def term_in_text(term: str, text: str) -> bool:
    """Case-insensitive, word-boundary occurrence of ``term`` in ``text``."""
    return bool(_term_re(term).search(text))


def _scan_terms(term: str) -> list[str]:
    """``term`` plus the parts of a hyphenated combination (``fluticasone-salmeterol`` -> both drugs)."""
    parts = [term]
    if "-" in term and not term[0].isdigit():
        parts += [p for p in term.split("-") if len(p) >= 4]
    return parts


def _lab_surface_forms(test_key: str) -> list[str]:
    spec = V.LAB_TESTS[test_key]
    forms = {spec.short, spec.name, spec.name.split(" (")[0], LAB_QUESTION_NAMES[test_key]}
    forms.update(re.findall(r"\(([^)]+)\)", spec.name))
    if test_key == "hba1c":
        forms.update({"HbA1c", "A1c"})
    return sorted(f for f in forms if f)


def _norm_ws(text: str) -> str:
    return " ".join(text.split()).lower()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def reason_keywords(reason: str) -> list[str]:
    """Answer keywords for a discontinue reason: lab phrase ("K 5.7"), known terms, short main clause."""
    kws = _REASON_LAB_RE.findall(reason)
    kws += [t for t in REASON_TERMS if term_in_text(t, reason)]
    main = re.sub(r"\s*\([^)]*\)", "", reason.split(";")[0]).strip()
    if main and (len(main.split()) <= 6 or not kws):
        kws.append(main)
    return _dedupe(kws)


# --------------------------------------------------------------------------- per-patient context


class _PatientCtx:
    def __init__(self, pid: str, recs: dict[str, list[dict]], docs: list[Document]) -> None:
        self.pid = pid
        self.recs = recs
        self.docs = docs
        self.doc_by_id = {d.doc_id: d for d in docs}
        self.notes = [d for d in docs if d.doc_type == "clinical_note"]
        self.all_text = "\n".join(d.text for d in docs)

    def absent(self, term: str) -> bool:
        return not any(term_in_text(t, self.all_text) for t in _scan_terms(term))

    def notes_where(self, pred: Callable[[Document, dict], bool]) -> list[str]:
        return [n.doc_id for n in self.notes if pred(n, n.metadata.get("mentions") or {})]


def _rng(seed: int, pid: str, category: str) -> random.Random:
    # String seeds are hashed with SHA-512 by ``random`` -> stable across processes and PYTHONHASHSEED.
    return random.Random(f"ehr-rag-eval|{seed}|{pid}|{category}")


# --------------------------------------------------------------------------- fact_lookup


def _fact_med_dose(ctx: _PatientCtx, rng: random.Random) -> Optional[dict]:
    active = [m for m in ctx.recs["medications"] if m["status"] == "active"]
    names = Counter(m["name"] for m in active)
    cands = [m for m in active if names[m["name"]] == 1]
    if not cands:
        return None
    med = rng.choice(cands)
    full = f"{med['name']} {med['dose']}"
    notes = ctx.notes_where(lambda n, mt: med["med_id"] in mt.get("medications", []) and answer_correct(n.text, [full]))
    return {
        "question": Q_MED_DOSE.format(name=med["name"]), "expected": render_medication(med),
        "keywords": [med["dose"]], "gold": [medications_doc_id(ctx.pid)] + notes,
        "meta": {"fact_kind": "med_dose", "med_id": med["med_id"], "key_term": med["name"]},
    }


def _fact_allergy(ctx: _PatientCtx, rng: random.Random) -> Optional[dict]:
    if not ctx.recs["allergies"]:
        return None
    a = rng.choice(ctx.recs["allergies"])
    notes = ctx.notes_where(lambda n, mt: term_in_text(a["substance"], n.text) and term_in_text(a["reaction"], n.text))
    return {
        "question": Q_ALLERGY.format(substance=a["substance"]), "expected": f"{a['reaction']} ({a['severity']})",
        "keywords": [a["reaction"]], "gold": [allergies_doc_id(ctx.pid)] + notes,
        "meta": {"fact_kind": "allergy_reaction", "substance": a["substance"], "key_term": a["substance"]},
    }


def _fact_stop_reason(ctx: _PatientCtx, rng: random.Random) -> Optional[dict]:
    stopped = [m for m in ctx.recs["medications"] if m["status"] == "discontinued" and m["discontinue_reason"]]
    key = Counter((m["name"], m["dose"]) for m in stopped)
    stopped = [m for m in stopped if key[(m["name"], m["dose"])] == 1]
    if not stopped:
        return None
    preferred = [m for m in stopped if not m["discontinue_reason"].startswith("completed")]
    med = rng.choice(preferred or stopped)
    notes = ctx.notes_where(lambda n, mt: n.date == med["end_date"] and med["med_id"] in mt.get("medications", []))
    return {
        "question": Q_STOP.format(name=med["name"], dose=med["dose"]), "expected": med["discontinue_reason"],
        "keywords": reason_keywords(med["discontinue_reason"]), "gold": [medications_doc_id(ctx.pid)] + notes,
        "meta": {"fact_kind": "stop_reason", "med_id": med["med_id"], "end_date": med["end_date"],
                 "key_term": med["name"]},
    }


def _lab_gold(ctx: _PatientCtx, row: dict) -> list[str]:
    pair = [row["test_key"], row["date"]]
    notes = ctx.notes_where(lambda n, mt: pair in [list(x) for x in mt.get("labs", [])])
    return [lab_panel_doc_id(ctx.pid, row["date"])] + notes


def _lab_expected(row: dict) -> str:
    unit = f" {row['unit']}" if row["unit"] else ""
    return f"{format_lab_value(row['test_key'], row['value'])}{unit} on {row['date']}"


def _fact_lab_value(ctx: _PatientCtx, rng: random.Random) -> Optional[dict]:
    if not ctx.recs["labs"]:
        return None
    row = rng.choice(sorted(ctx.recs["labs"], key=lambda r: (r["date"], r["test_key"])))
    return {
        "question": Q_LAB_ON.format(lab=LAB_QUESTION_NAMES[row["test_key"]], date=row["date"]),
        "expected": _lab_expected(row), "keywords": [format_lab_value(row["test_key"], row["value"])],
        "gold": _lab_gold(ctx, row),
        "meta": {"fact_kind": "lab_value", "test_key": row["test_key"], "date": row["date"],
                 "key_term": row["test_key"]},
    }


_FACT_BUILDERS: dict[str, Callable[[_PatientCtx, random.Random], Optional[dict]]] = {
    "med_dose": _fact_med_dose, "allergy_reaction": _fact_allergy, "stop_reason": _fact_stop_reason,
    "lab_value": _fact_lab_value,
}


_FACT_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.combinations(FACT_KINDS, 2))


def _fact_lookups(ctx: _PatientCtx, rng: random.Random, offset: int, n: int = 2) -> list[dict]:
    """``n`` facts of different kinds; the preferred pair of kinds cycles with ``offset`` for balance."""
    first = list(_FACT_PAIRS[offset % len(_FACT_PAIRS)])
    rest = [k for k in FACT_KINDS if k not in first]
    rng.shuffle(rest)
    out = []
    for kind in first + rest:
        if len(out) >= n:
            break
        spec = _FACT_BUILDERS[kind](ctx, rng)
        if spec is not None:
            out.append(spec)
    return out


# --------------------------------------------------------------------------- temporal


def _temporal(ctx: _PatientCtx, rng: random.Random, avoid: set[str]) -> Optional[dict]:
    by_test: dict[str, list[dict]] = defaultdict(list)
    for r in ctx.recs["labs"]:
        by_test[r["test_key"]].append(r)
    cands, distinct = [], []
    for key in sorted(by_test):
        rows = sorted(by_test[key], key=lambda r: r["date"])
        if len(rows) < 2:
            continue
        cands.append(key)
        latest = format_lab_value(key, rows[-1]["value"])
        if all(format_lab_value(key, r["value"]) != latest for r in rows[:-1]):
            distinct.append(key)
    if not cands:
        return None
    related = {t for c in ctx.recs["conditions"] for t in V.CONDITIONS[c["condition_key"]].labs}
    for pool in ([k for k in distinct if k in related and k not in avoid], [k for k in distinct if k not in avoid],
                 distinct, cands):
        if pool:
            key = rng.choice(pool)
            break
    rows = sorted(by_test[key], key=lambda r: r["date"])
    latest = rows[-1]
    return {
        "question": Q_LAB_LATEST.format(lab=LAB_QUESTION_NAMES[key]), "expected": _lab_expected(latest),
        "keywords": [format_lab_value(key, latest["value"])], "gold": _lab_gold(ctx, latest),
        "meta": {"test_key": key, "date": latest["date"], "n_measurements": len(rows),
                 "prior_values": [format_lab_value(key, r["value"]) for r in rows[:-1]], "key_term": key},
    }


# --------------------------------------------------------------------------- semantic_synonym


def condition_surface_forms(cond_key: str) -> list[str]:
    """Abbreviation, display name and note terms of a condition (must not appear in a lay question)."""
    spec = V.CONDITIONS[cond_key]
    return _dedupe([spec.abbreviation, spec.display, *spec.note_terms])


def _semantic(ctx: _PatientCtx, rng: random.Random, variant_offset: int) -> Optional[dict]:
    conds = [c for c in ctx.recs["conditions"] if V.CONDITIONS[c["condition_key"]].lay_terms]
    if not conds:
        return None
    active_meds = [m for m in ctx.recs["medications"] if m["status"] == "active"]
    variants = ("meds", "onset", "has")
    order = variants[variant_offset % 3:] + variants[:variant_offset % 3]
    for variant in order:
        pool = conds
        if variant == "meds":
            pool = [c for c in conds if c["condition_key"] not in _NO_MEDS_TEMPLATE and c["status"] == "active"
                    and any(m["indication"] == c["condition_key"] for m in active_meds)]
        if not pool:
            continue
        cond = rng.choice(pool)
        key = cond["condition_key"]
        spec = V.CONDITIONS[key]
        lay = spec.lay_terms[0]
        meta = {"variant": variant, "condition_key": key, "lay_term": lay, "abbreviation": spec.abbreviation}
        mentioned = lambda n, mt: key in mt.get("conditions", [])  # noqa: E731
        if variant == "meds":
            meds = sorted({m["name"] for m in active_meds if m["indication"] == key})
            notes = ctx.notes_where(lambda n, mt: mentioned(n, mt) and answer_correct(n.text, meds))
            return {"question": Q_SYN_MEDS.format(lay=lay), "expected": ", ".join(meds), "keywords": meds,
                    "gold": [medications_doc_id(ctx.pid)] + notes, "meta": meta}
        if variant == "onset":
            year = cond["onset_date"][:4]
            kws = [cond["onset_date"], year]
            notes = ctx.notes_where(lambda n, mt: mentioned(n, mt) and answer_correct(n.text, kws))
            return {"question": Q_SYN_ONSET.format(lay=lay),
                    "expected": f"{spec.abbreviation} onset {cond['onset_date']}", "keywords": kws,
                    "gold": [problems_doc_id(ctx.pid)] + notes, "meta": meta}
        kws = _dedupe([spec.abbreviation, *spec.note_terms, *MEDICAL_TERMS.get(key, ())])
        notes = ctx.notes_where(mentioned)
        return {"question": Q_SYN_HAS.format(lay=lay),
                "expected": f"Yes: {spec.display} ({spec.abbreviation}), {cond['status']}", "keywords": kws,
                "gold": [problems_doc_id(ctx.pid)] + notes, "meta": meta}
    return None


# --------------------------------------------------------------------------- verbatim


def _chunk_body(chunk: Chunk) -> str:
    head, sep, body = chunk.text.partition("\n")
    return body if sep and head.startswith("[") else chunk.text


_PROSE_WORD_RE = re.compile(r"^[A-Za-z][a-z]+[,.;]?$")
# Sections whose sentences are free prose (not lab lists, med lists or boilerplate).
VERBATIM_SECTIONS: frozenset[str] = frozenset(
    {"HISTORY OF PRESENT ILLNESS", "HOSPITAL COURSE", "ED COURSE", "ASSESSMENT AND PLAN", "FINDINGS",
     "IMPRESSION", "SOCIAL HISTORY", "PHYSICAL EXAM", "REASON FOR CONSULTATION", "CLINICAL INDICATION"}
)


def _is_prose(sentence: str) -> bool:
    words = sentence.split()
    return len(words) >= VERBATIM_MIN_WORDS and sum(1 for w in words if _PROSE_WORD_RE.match(w)) >= 6


def _token_set(text: str) -> frozenset[str]:
    # ISO dates stay one token (as in retrieval and metrics): otherwise "Last seen 2026-01-24; no ED visits ..."
    # counts as 3 distinct tokens against "Last seen 2025-07-10; ..." and a templated sentence that differs only
    # by its date passes the near-duplicate filter.
    return frozenset(re.findall(r"\d{4}-\d{2}-\d{2}|[a-z0-9]+(?:\.\d+)?", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _verbatim(ctx: _PatientCtx, chunks: list[Chunk], rng: random.Random) -> Optional[dict]:
    doc_texts = {d.doc_id: _norm_ws(d.text) for d in ctx.docs}
    other_sentences: dict[str, list[frozenset[str]]] = {
        d.doc_id: [_token_set(s) for s, _ in split_sentences(d.text)] for d in ctx.docs
    }
    cands: list[tuple[Chunk, str]] = []
    for ch in sorted(chunks, key=lambda c: c.chunk_id):
        if ch.doc_type != "clinical_note" or ch.doc_id not in ctx.doc_by_id or ch.section not in VERBATIM_SECTIONS:
            continue
        for sentence, _ in split_sentences(_chunk_body(ch)):
            s = sentence.strip()
            if s.startswith(("-", "*", "•")) or '"' in s or not _is_prose(s):
                continue
            ns = _norm_ws(s)
            if any(ns in t for did, t in doc_texts.items() if did != ch.doc_id):
                continue   # sentence also occurs in another document -> gold would be ambiguous
            toks = _token_set(s)
            if any(_jaccard(toks, o) >= VERBATIM_MAX_JACCARD
                   for did, sents in other_sentences.items() if did != ch.doc_id for o in sents):
                continue   # near-duplicate of a sentence elsewhere (templated closers, repeated plans)
            cands.append((ch, s))
    if not cands:
        return None
    ch, sentence = rng.choice(cands)
    words = _dedupe(w for w in _WORD_RE.findall(sentence) if len(w) >= 4 and w.lower() not in _WORD_STOPWORDS)
    df = {w: sum(1 for d in ctx.docs if term_in_text(w, d.text)) for w in words}
    keywords = sorted(words, key=lambda w: (df[w], -len(w), w.lower()))[:2]
    if not keywords:
        return None
    return {
        "question": Q_VERBATIM.format(sentence=sentence), "expected": sentence, "keywords": keywords,
        "gold": [ch.doc_id],
        "meta": {"chunk_id": ch.chunk_id, "section": ch.section, "doc_title": ch.title, "doc_date": ch.date},
    }


# --------------------------------------------------------------------------- negatives


def _least_used(pool: list[str], usage: Counter, rng: random.Random) -> str:
    """Random choice among the pool items used least often so far (spreads terms across patients)."""
    low = min(usage[x] for x in pool)
    return rng.choice([x for x in pool if usage[x] == low])


def _absent(ctx: _PatientCtx, rng: random.Random, offset: int, usage: Counter) -> Optional[dict]:
    kinds = ("med", "allergen", "lab")
    order = kinds[offset % 3:] + kinds[:offset % 3]
    for kind in order:
        if kind == "med":
            pool = [m for m in sorted(V.MEDICATIONS) if ctx.absent(m)]
            if pool:
                name = _least_used(pool, usage, rng)
                return {"question": Q_MED_DOSE.format(name=name), "key_terms": _scan_terms(name),
                        "meta": {"absent_kind": "med", "key_term": name}}
        elif kind == "allergen":
            pool = [a.substance for a in V.ALLERGENS if ctx.absent(a.substance)]
            if pool:
                sub = _least_used(pool, usage, rng)
                return {"question": Q_ALLERGY.format(substance=sub), "key_terms": _scan_terms(sub),
                        "meta": {"absent_kind": "allergen", "key_term": sub}}
        else:
            pool = [LAB_QUESTION_NAMES[k] for k in sorted(V.LAB_TESTS)
                    if all(ctx.absent(f) for f in _lab_surface_forms(k))]
            if pool:
                name = _least_used(pool, usage, rng)
                key = next(k for k, v in LAB_QUESTION_NAMES.items() if v == name)
                return {"question": Q_LAB_LATEST.format(lab=name), "key_terms": _lab_surface_forms(key),
                        "meta": {"absent_kind": "lab", "key_term": name, "test_key": key}}
    return None


def _cross_patient(ctx: _PatientCtx, others: list[_PatientCtx], holders: Counter, rng: random.Random,
                   offset: int, exclude: set[str], usage: Counter) -> Optional[dict]:
    """Another patient's distinctive med/allergen (held by <= 3 patients) never mentioned in this chart.

    Ranked by: least used so far in this question set, preferred kind (alternates allergen/med with
    ``offset``), fewest holders, then a seeded shuffle of the source patients.
    """
    cands: list[tuple[tuple, str, str, _PatientCtx, dict]] = []
    others = list(others)
    rng.shuffle(others)
    prefer = "allergen" if offset % 2 == 0 else "med"
    for order, src in enumerate(others):
        recs = [("allergen", a["substance"], a) for a in src.recs["allergies"]]
        recs += [("med", m["name"], m) for m in src.recs["medications"]]
        for kind, term, rec in recs:
            if term in exclude or not ctx.absent(term):
                continue
            rank = (holders[term] > V.MAX_PATIENTS_PER_ALLERGEN, usage[term], kind != prefer, holders[term],
                    order, term)
            cands.append((rank, kind, term, src, rec))
    if not cands:
        return None
    _, kind, term, src, rec = min(cands, key=lambda c: c[0])
    if kind == "allergen":
        question, source_kw, source_docs = Q_ALLERGY.format(substance=term), [rec["reaction"]], [
            allergies_doc_id(src.pid)]
    else:
        question, source_kw, source_docs = Q_MED_DOSE.format(name=term), [rec["dose"]], [medications_doc_id(src.pid)]
    return {
        "question": question, "key_terms": _scan_terms(term),
        "meta": {"source_patient_id": src.pid, "source_kind": kind, "key_term": term, "n_holders": holders[term],
                 "source_answer_keywords": source_kw, "source_doc_ids": source_docs},
    }


# --------------------------------------------------------------------------- build


def _group_records(dataset: SyntheticDataset) -> dict[str, dict[str, list[dict]]]:
    tables = ("encounters", "conditions", "medications", "allergies", "labs", "vitals", "notes")
    out: dict[str, dict[str, list[dict]]] = {p["patient_id"]: {t: [] for t in tables} for p in dataset.patients}
    for t in tables:
        for rec in getattr(dataset, t):
            out[rec["patient_id"]][t].append(rec)
    return out


def _make(pid: str, category: str, n: int, spec: dict, negative: bool) -> EvalQuestion:
    meta = dict(spec.get("meta", {}))
    if negative:
        meta["key_terms"] = list(spec["key_terms"])
    return EvalQuestion(
        qid=f"{pid}-{category}-{n}",
        patient_id=pid,
        question=spec["question"],
        category=category,
        expected_answer=None if negative else spec["expected"],
        answer_keywords=[] if negative else list(spec["keywords"]),
        gold_doc_ids=[] if negative else _dedupe(spec["gold"]),
        metadata={"synthetic": True, **meta},
    )


def build_questions(dataset: SyntheticDataset, chunks: list[Chunk] | None = None, seed: int = 42) -> list[EvalQuestion]:
    """Build the gold question set (see module docstring). ``verbatim`` questions need ``chunks``."""
    docs = dataset.to_documents()
    docs_by_pid: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        docs_by_pid[d.patient_id].append(d)
    recs = _group_records(dataset)
    pids = [p["patient_id"] for p in dataset.patients]
    ctxs = {pid: _PatientCtx(pid, recs[pid], docs_by_pid[pid]) for pid in pids}
    chunks_by_pid: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks or []:
        chunks_by_pid[c.patient_id].append(c)
    if chunks is None:
        logger.warning("build_questions: no chunks given, skipping verbatim questions")

    holders: Counter = Counter()
    for pid in pids:
        terms = {a["substance"] for a in recs[pid]["allergies"]} | {m["name"] for m in recs[pid]["medications"]}
        holders.update(terms)

    absent_usage: Counter = Counter()
    cross_usage: Counter = Counter()
    questions: list[EvalQuestion] = []
    for idx, pid in enumerate(pids):
        ctx = ctxs[pid]
        made: list[tuple[str, dict]] = []
        facts = _fact_lookups(ctx, _rng(seed, pid, "fact_lookup"), idx)
        made += [("fact_lookup", f) for f in facts]
        avoid = {f["meta"]["test_key"] for f in facts if f["meta"].get("fact_kind") == "lab_value"}
        temporal = _temporal(ctx, _rng(seed, pid, "temporal"), avoid)
        if temporal:
            made.append(("temporal", temporal))
        semantic = _semantic(ctx, _rng(seed, pid, "semantic_synonym"), idx)
        if semantic:
            made.append(("semantic_synonym", semantic))
        if chunks is not None:
            verbatim = _verbatim(ctx, chunks_by_pid.get(pid, []), _rng(seed, pid, "verbatim"))
            if verbatim:
                made.append(("verbatim", verbatim))
        absent = _absent(ctx, _rng(seed, pid, "absent_fact"), idx, absent_usage)
        if absent:
            absent_usage[absent["meta"]["key_term"]] += 1
            made.append(("absent_fact", absent))
        cross = _cross_patient(ctx, [ctxs[o] for o in pids if o != pid], holders, _rng(seed, pid, "cross_patient"),
                               idx, {absent["meta"]["key_term"]} if absent else set(), cross_usage)
        if cross:
            cross_usage[cross["meta"]["key_term"]] += 1
            made.append(("cross_patient", cross))

        counters: Counter = Counter()
        for category, spec in made:
            counters[category] += 1
            questions.append(_make(pid, category, counters[category], spec, category in NEGATIVE_CATEGORIES))
        missing = [c for c in CATEGORIES if c not in counters and (c != "verbatim" or chunks is not None)]
        if missing:
            logger.warning("Patient %s: no question could be built for %s", pid, ", ".join(missing))

    validate_questions(questions, docs, chunks=chunks)
    logger.info("Built %d evaluation questions for %d patients (%s)", len(questions), len(pids),
                ", ".join(f"{k}={v}" for k, v in sorted(Counter(q.category for q in questions).items())))
    return questions


# --------------------------------------------------------------------------- validation


def validate_questions(questions: list[EvalQuestion], documents: list[Document],
                       chunks: list[Chunk] | None = None) -> None:
    """Raise ``ValueError`` listing every violated invariant (see module docstring)."""
    errors: list[str] = []
    by_id = {d.doc_id: d for d in documents}
    text_by_pid: dict[str, str] = defaultdict(str)
    for d in documents:
        text_by_pid[d.patient_id] += d.text + "\n"
    chunk_ids = {c.chunk_id for c in chunks or []}

    counts = Counter(q.qid for q in questions)
    errors += [f"duplicate qid {qid}" for qid, n in counts.items() if n > 1]
    for q in questions:
        tag = f"{q.qid}:"
        if q.category not in CATEGORIES:
            errors.append(f"{tag} unknown category {q.category!r}")
            continue
        if q.patient_id not in text_by_pid:
            errors.append(f"{tag} unknown patient {q.patient_id}")
            continue
        if q.category in POSITIVE_CATEGORIES:
            if q.expected_answer is None or not q.answer_keywords or not q.gold_doc_ids:
                errors.append(f"{tag} positive question needs expected_answer, keywords and gold docs")
                continue
            missing = [g for g in q.gold_doc_ids if g not in by_id]
            if missing:
                errors.append(f"{tag} gold doc ids do not exist: {missing}")
                continue
            if any(by_id[g].patient_id != q.patient_id for g in q.gold_doc_ids):
                errors.append(f"{tag} gold doc belongs to another patient")
            if not any(answer_correct(by_id[g].text, q.answer_keywords) for g in q.gold_doc_ids):
                errors.append(f"{tag} no keyword {q.answer_keywords} occurs in any gold doc")
        else:
            if q.expected_answer is not None or q.gold_doc_ids:
                errors.append(f"{tag} negative question must have expected_answer=None and no gold docs")
            terms = q.metadata.get("key_terms") or [q.metadata.get("key_term")]
            if not all(terms):
                errors.append(f"{tag} negative question has no key term")
            present = [t for t in terms if t and term_in_text(t, text_by_pid[q.patient_id])]
            if present:
                errors.append(f"{tag} key term(s) {present} occur in patient {q.patient_id}'s documents")
        if q.category == "cross_patient":
            src = q.metadata.get("source_patient_id")
            if not src or src == q.patient_id or src not in text_by_pid:
                errors.append(f"{tag} cross_patient needs a different, existing source_patient_id")
            elif not term_in_text(q.metadata.get("key_term", ""), text_by_pid[src]):
                errors.append(f"{tag} key term not found in source patient {src}'s documents")
        if q.category == "semantic_synonym":
            key = q.metadata.get("condition_key")
            if key not in V.CONDITIONS:
                errors.append(f"{tag} semantic question without a valid condition_key")
            else:
                leaked = [t for t in condition_surface_forms(key) if term_in_text(t, q.question)]
                if leaked:
                    errors.append(f"{tag} lay question contains clinical term(s) {leaked}")
                if not any(term_in_text(t, q.question) for t in V.CONDITIONS[key].lay_terms):
                    errors.append(f"{tag} lay question contains no lay term")
        if q.category == "verbatim":
            if len(q.gold_doc_ids) != 1:
                errors.append(f"{tag} verbatim question must have exactly one gold doc")
            elif _norm_ws(q.expected_answer or "") not in _norm_ws(by_id[q.gold_doc_ids[0]].text):
                errors.append(f"{tag} quoted sentence not found verbatim in its gold doc")
            if chunks is not None and q.metadata.get("chunk_id") not in chunk_ids:
                errors.append(f"{tag} verbatim chunk id not among the given chunks")
    if errors:
        raise ValueError(f"{len(errors)} invalid evaluation question(s):\n" + "\n".join(errors[:50]))


# --------------------------------------------------------------------------- io


def write_questions(questions: list[EvalQuestion], path: Path) -> None:
    """Write one JSON object per line (``EvalQuestion.to_dict``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d evaluation questions to %s", len(questions), path)


def read_questions(path: Path) -> list[EvalQuestion]:
    """Read questions written by ``write_questions``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the questions first")
    with path.open(encoding="utf-8") as fh:
        return [EvalQuestion.from_dict(json.loads(line)) for line in fh if line.strip()]


def question_stats(questions: list[EvalQuestion]) -> dict[str, Any]:
    """Counts per category and per patient (for reports)."""
    return {
        "n": len(questions),
        "by_category": dict(Counter(q.category for q in questions)),
        "patients": len({q.patient_id for q in questions}),
    }
