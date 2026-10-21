"""Clinical vocabulary for the synthetic EHR generator (plain data, no logic beyond lookups).

Contents
--------
* ``CONDITIONS``  - chronic (and one acute) conditions with ICD-10, display name, note abbreviation,
  ``note_terms`` (every surface form a note may use), ``lay_terms``, typical meds and labs.
* ``PROFILES``    - coherent comorbidity profiles a patient is drawn from.
* ``MEDICATIONS`` - generic name, dose options, route, frequency, class, indications.
* ``REGIMENS``    - per-condition medication "slots" (one drug per slot, one drug per class per patient).
* ``SWITCHES``    - realistic medication changes (dose change / switch / stop) with reasons.
* ``LAB_TESTS``   - LOINC, units, reference ranges, number format (decimals), plausible ranges.
* ``ALLERGENS``   - substances with reactions/severities and the medications they exclude.
* ``NOTE_PHRASES``- HPI / plan phrasing variants per condition (``{cond}`` and ``{meds}`` placeholders).
* Fixed lists of invented patient and provider names.

``lay_terms`` (e.g. "heart attack", "high blood pressure") must never appear in generated text: they exist so
evaluation can ask lay-language questions against abbreviated chart text. Nothing in ``display``,
``note_terms``, phrases or reasons may contain a lay term (enforced by tests).

All names are invented. Providers carry a "(synthetic)" suffix. No real patient data lives here.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    display: str
    icd10: str
    abbreviation: str                 # preferred surface form in notes
    note_terms: tuple[str, ...]       # every surface form that counts as a mention (word-boundary, case-insensitive)
    lay_terms: tuple[str, ...]        # NEVER emitted in generated text
    meds: tuple[str, ...]             # typical medications (keys of MEDICATIONS)
    labs: tuple[str, ...]             # related lab tests (keys of LAB_TESTS)
    chronic: bool = True


CONDITIONS: dict[str, ConditionSpec] = {
    c.key: c
    for c in (
        ConditionSpec(
            "t2dm", "Type 2 diabetes mellitus without complications", "E11.9", "T2DM", ("T2DM",),
            ("sugar diabetes", "high blood sugar", "blood sugar disease"),
            ("metformin", "empagliflozin", "glipizide", "sitagliptin", "insulin glargine", "semaglutide"),
            ("hba1c", "glucose", "creatinine", "egfr"),
        ),
        ConditionSpec(
            "htn", "Essential (primary) hypertension", "I10", "HTN", ("HTN",),
            ("high blood pressure", "blood pressure problem"),
            ("lisinopril", "losartan", "amlodipine", "hydrochlorothiazide"),
            ("creatinine", "potassium", "sodium"),
        ),
        ConditionSpec(
            "hld", "Hyperlipidemia, unspecified", "E78.5", "HLD", ("HLD",),
            ("high cholesterol", "fatty blood"),
            ("atorvastatin", "rosuvastatin", "simvastatin"),
            ("ldl", "hdl", "triglycerides", "total_cholesterol", "alt"),
        ),
        ConditionSpec(
            "ckd3", "Chronic renal insufficiency, stage 3a", "N18.31", "CKD3a", ("CKD3a", "CKD"),
            ("kidney disease", "weak kidneys", "kidney problems"),
            ("losartan",),
            ("creatinine", "egfr", "bun", "potassium", "hemoglobin"),
        ),
        ConditionSpec(
            "cad", "Atherosclerotic heart disease of native coronary artery", "I25.10", "CAD", ("CAD",),
            ("clogged heart arteries", "hardened heart arteries"),
            ("aspirin", "atorvastatin", "metoprolol succinate"),
            ("ldl", "total_cholesterol"),
        ),
        ConditionSpec(
            "prior_mi", "Old myocardial infarction", "I25.2", "s/p MI", ("s/p MI", "MI", "NSTEMI"),
            ("heart attack", "previous heart attack"),
            ("clopidogrel", "metoprolol succinate"),
            ("ldl",),
        ),
        ConditionSpec(
            "hfref", "Heart failure with reduced ejection fraction", "I50.22", "HFrEF", ("HFrEF", "HF"),
            ("weak heart", "weak heart muscle", "heart pump problem"),
            ("sacubitril-valsartan", "carvedilol", "metoprolol succinate", "furosemide", "spironolactone",
             "empagliflozin"),
            ("bnp", "creatinine", "potassium", "sodium"),
        ),
        ConditionSpec(
            "afib", "Atrial fibrillation, unspecified", "I48.91", "AFib", ("AFib",),
            ("irregular heartbeat", "heart flutter", "fluttering heart"),
            ("apixaban", "warfarin", "metoprolol succinate"),
            ("inr", "tsh", "hemoglobin"),
        ),
        ConditionSpec(
            "copd", "Chronic obstructive pulmonary disease, unspecified", "J44.9", "COPD", ("COPD",),
            ("smoker's lung", "lung damage from smoking"),
            ("tiotropium", "umeclidinium-vilanterol", "albuterol", "fluticasone-salmeterol"),
            ("wbc", "hemoglobin"),
        ),
        ConditionSpec(
            "asthma", "Moderate persistent asthma, uncomplicated", "J45.40", "asthma", ("asthma",),
            ("wheezy chest", "twitchy airways"),
            ("fluticasone-salmeterol", "albuterol", "montelukast"),
            ("wbc",),
        ),
        ConditionSpec(
            "hypothyroid", "Hypothyroidism, unspecified", "E03.9", "hypothyroidism", ("hypothyroidism",),
            ("underactive thyroid", "slow thyroid", "low thyroid"),
            ("levothyroxine",),
            ("tsh",),
        ),
        ConditionSpec(
            "mdd", "Major depressive disorder, recurrent, moderate", "F33.1", "MDD", ("MDD",),
            ("depression", "feeling down", "the blues"),
            ("sertraline", "bupropion", "escitalopram"),
            (),
        ),
        ConditionSpec(
            "gad", "Generalized anxiety disorder", "F41.1", "GAD", ("GAD",),
            ("constant worrying", "nervous disorder"),
            ("escitalopram", "sertraline", "buspirone"),
            (),
        ),
        ConditionSpec(
            "gerd", "Gastro-esophageal reflux disease without esophagitis", "K21.9", "GERD", ("GERD",),
            ("heartburn", "acid reflux"),
            ("omeprazole", "pantoprazole", "famotidine"),
            ("hemoglobin",),
        ),
        ConditionSpec(
            "oa", "Primary osteoarthritis of bilateral knees", "M17.0", "knee OA", ("knee OA", "OA"),
            ("wear-and-tear arthritis", "worn-out joints"),
            ("acetaminophen", "diclofenac", "meloxicam"),
            ("creatinine",),
        ),
        ConditionSpec(
            "osa", "Obstructive sleep apnea", "G47.33", "OSA", ("OSA",),
            ("snoring problem", "stopping breathing at night"),
            (),
            (),
        ),
        ConditionSpec(
            "obesity", "Obesity, unspecified", "E66.9", "obesity", ("obesity",),
            ("overweight", "being heavy"),
            ("semaglutide",),
            ("hba1c", "triglycerides"),
        ),
        ConditionSpec(
            "cap", "Pneumonia, unspecified organism", "J18.9", "CAP", ("CAP", "pneumonia"),
            ("chest infection", "lung infection"),
            ("amoxicillin-clavulanate", "levofloxacin", "doxycycline"),
            ("wbc",),
            chronic=False,
        ),
    )
}

CHRONIC_CONDITIONS: tuple[str, ...] = tuple(k for k, c in CONDITIONS.items() if c.chronic)

# Order in which conditions claim medication classes (first claim wins).
CONDITION_PRIORITY: tuple[str, ...] = (
    "hfref", "afib", "cad", "prior_mi", "t2dm", "ckd3", "htn", "hld", "copd", "asthma",
    "hypothyroid", "mdd", "gad", "gerd", "oa", "obesity", "osa", "cap",
)


@dataclass(frozen=True)
class Profile:
    key: str
    core: tuple[str, ...]
    optional: tuple[tuple[str, float], ...]   # (condition key, probability)
    weight: float


PROFILES: dict[str, Profile] = {
    p.key: p
    for p in (
        Profile("cardiometabolic", ("t2dm", "htn", "hld"), (("ckd3", 0.45), ("obesity", 0.5), ("osa", 0.25)), 0.30),
        Profile("cardiac", ("cad", "prior_mi", "hfref", "htn", "hld"), (("afib", 0.6), ("t2dm", 0.3), ("ckd3", 0.25)),
                0.20),
        Profile("pulmonary_copd", ("copd",), (("htn", 0.6), ("hld", 0.45), ("gerd", 0.4), ("afib", 0.2)), 0.14),
        Profile("pulmonary_asthma", ("asthma",), (("gerd", 0.5), ("obesity", 0.45), ("gad", 0.35)), 0.12),
        Profile("endocrine_mood", ("hypothyroid", "mdd"), (("gad", 0.45), ("hld", 0.4), ("obesity", 0.3)), 0.12),
        Profile("musculoskeletal", ("oa", "htn"), (("obesity", 0.55), ("gerd", 0.4), ("osa", 0.3), ("hld", 0.3)),
                0.12),
    )
}

ADD_ON_CONDITIONS: tuple[str, ...] = ("hypothyroid", "mdd", "gad", "gerd", "oa", "osa")
INCOMPATIBLE_CONDITIONS: tuple[frozenset[str], ...] = (frozenset({"copd", "asthma"}),)
MAX_CHRONIC_CONDITIONS = 6
MIN_CHRONIC_CONDITIONS = 2

# --------------------------------------------------------------------------- medications


@dataclass(frozen=True)
class MedicationSpec:
    name: str                        # generic, lowercase (also the key)
    doses: tuple[str, ...]           # realistic dose options, e.g. "1000 mg"
    route: str                       # PO | SC | inhaled | topical
    frequency: str                   # daily | BID | at bedtime | weekly | q4-6h PRN ...
    drug_class: str                  # at most one drug per class per patient
    indications: tuple[str, ...]     # condition keys
    avoid_with: tuple[str, ...] = ()  # condition keys / drug classes that contraindicate it


MEDICATIONS: dict[str, MedicationSpec] = {
    m.name: m
    for m in (
        # diabetes / weight
        MedicationSpec("metformin", ("500 mg", "1000 mg"), "PO", "BID", "biguanide", ("t2dm",)),
        MedicationSpec("empagliflozin", ("10 mg", "25 mg"), "PO", "daily", "sglt2", ("t2dm", "hfref")),
        MedicationSpec("glipizide", ("5 mg",), "PO", "daily", "sulfonylurea", ("t2dm",)),
        MedicationSpec("sitagliptin", ("100 mg", "50 mg"), "PO", "daily", "dpp4", ("t2dm",), ("glp1",)),
        MedicationSpec("insulin glargine", ("14 units", "20 units", "26 units"), "SC", "at bedtime", "insulin",
                       ("t2dm",)),
        MedicationSpec("semaglutide", ("0.25 mg", "0.5 mg", "1 mg"), "SC", "weekly", "glp1", ("t2dm", "obesity"),
                       ("dpp4",)),
        # hypertension / renal
        MedicationSpec("lisinopril", ("10 mg", "20 mg"), "PO", "daily", "raas", ("htn",)),
        MedicationSpec("losartan", ("50 mg", "100 mg"), "PO", "daily", "raas", ("htn", "ckd3")),
        MedicationSpec("amlodipine", ("5 mg", "10 mg"), "PO", "daily", "ccb", ("htn",), ("hfref",)),
        MedicationSpec("hydrochlorothiazide", ("12.5 mg", "25 mg"), "PO", "daily", "thiazide", ("htn",)),
        # lipids / coronary
        MedicationSpec("atorvastatin", ("20 mg", "40 mg", "80 mg"), "PO", "at bedtime", "statin", ("hld", "cad")),
        MedicationSpec("rosuvastatin", ("10 mg", "20 mg"), "PO", "daily", "statin", ("hld", "cad")),
        MedicationSpec("simvastatin", ("20 mg",), "PO", "at bedtime", "statin", ("hld",)),
        MedicationSpec("aspirin", ("81 mg",), "PO", "daily", "antiplatelet", ("cad",), ("anticoagulant",)),
        MedicationSpec("clopidogrel", ("75 mg",), "PO", "daily", "p2y12", ("prior_mi",), ("anticoagulant",)),
        # heart failure / rhythm
        MedicationSpec("metoprolol succinate", ("25 mg", "50 mg", "100 mg"), "PO", "daily", "beta_blocker",
                       ("cad", "prior_mi", "afib", "hfref")),
        MedicationSpec("carvedilol", ("6.25 mg", "12.5 mg"), "PO", "BID", "beta_blocker", ("hfref",)),
        MedicationSpec("sacubitril-valsartan", ("49-51 mg", "24-26 mg"), "PO", "BID", "raas", ("hfref",)),
        MedicationSpec("furosemide", ("20 mg", "40 mg"), "PO", "daily", "loop", ("hfref",)),
        MedicationSpec("spironolactone", ("25 mg",), "PO", "daily", "mra", ("hfref",)),
        MedicationSpec("apixaban", ("5 mg",), "PO", "BID", "anticoagulant", ("afib",)),
        MedicationSpec("warfarin", ("5 mg", "4 mg"), "PO", "daily", "anticoagulant", ("afib",)),
        # pulmonary
        MedicationSpec("tiotropium", ("18 mcg",), "inhaled", "daily", "lama", ("copd",)),
        MedicationSpec("umeclidinium-vilanterol", ("62.5-25 mcg",), "inhaled", "daily", "lama", ("copd",)),
        MedicationSpec("fluticasone-salmeterol", ("250-50 mcg", "100-50 mcg"), "inhaled", "BID", "ics_laba",
                       ("asthma", "copd")),
        MedicationSpec("albuterol", ("90 mcg 2 puffs",), "inhaled", "q4-6h PRN", "saba", ("asthma", "copd")),
        MedicationSpec("montelukast", ("10 mg",), "PO", "at bedtime", "ltra", ("asthma",)),
        # endocrine / psych / GI / MSK
        MedicationSpec("levothyroxine", ("50 mcg", "75 mcg", "88 mcg", "100 mcg"), "PO", "daily", "thyroid",
                       ("hypothyroid",)),
        MedicationSpec("sertraline", ("50 mg", "100 mg"), "PO", "daily", "ssri", ("mdd", "gad")),
        MedicationSpec("escitalopram", ("10 mg",), "PO", "daily", "ssri", ("gad", "mdd")),
        MedicationSpec("bupropion", ("150 mg",), "PO", "daily (extended-release)", "ndri", ("mdd",)),
        MedicationSpec("buspirone", ("10 mg",), "PO", "BID", "azapirone", ("gad",)),
        MedicationSpec("omeprazole", ("20 mg", "40 mg"), "PO", "daily before breakfast", "ppi", ("gerd",)),
        MedicationSpec("pantoprazole", ("40 mg",), "PO", "daily before breakfast", "ppi", ("gerd",)),
        MedicationSpec("famotidine", ("20 mg",), "PO", "BID", "h2", ("gerd",), ("ppi",)),
        MedicationSpec("acetaminophen", ("650 mg",), "PO", "q6h PRN pain", "analgesic", ("oa",)),
        MedicationSpec("diclofenac", ("1% gel 4 g",), "topical", "QID to knees", "topical_nsaid", ("oa",)),
        MedicationSpec("meloxicam", ("7.5 mg", "15 mg"), "PO", "daily", "nsaid", ("oa",),
                       ("ckd3", "hfref", "cad", "anticoagulant")),
        # short courses (inpatient / ED)
        MedicationSpec("prednisone", ("40 mg",), "PO", "daily", "steroid", ("copd", "asthma")),
        MedicationSpec("azithromycin", ("250 mg",), "PO", "daily", "macrolide", ("copd",)),
        MedicationSpec("amoxicillin-clavulanate", ("875-125 mg",), "PO", "BID", "antibiotic", ("cap",)),
        MedicationSpec("levofloxacin", ("750 mg",), "PO", "daily", "antibiotic", ("cap",)),
        MedicationSpec("doxycycline", ("100 mg",), "PO", "BID", "antibiotic", ("cap",)),
    )
}

# Per-condition regimen slots. Each slot is a tuple of (med, dose) options; ``None`` = slot may stay empty.
REGIMENS: dict[str, tuple[tuple[tuple[str, str] | None, ...], ...]] = {
    "hfref": (
        (("sacubitril-valsartan", "49-51 mg"), ("sacubitril-valsartan", "24-26 mg"), ("losartan", "50 mg")),
        (("carvedilol", "12.5 mg"), ("metoprolol succinate", "50 mg")),
        (("furosemide", "20 mg"), ("furosemide", "40 mg")),
        (("spironolactone", "25 mg"), None),
        (("empagliflozin", "10 mg"), None),
    ),
    "afib": (
        (("apixaban", "5 mg"), ("apixaban", "5 mg"), ("warfarin", "5 mg")),
        (("metoprolol succinate", "50 mg"), ("metoprolol succinate", "25 mg")),
    ),
    "cad": (
        (("aspirin", "81 mg"),),
        (("atorvastatin", "80 mg"), ("rosuvastatin", "20 mg")),
        (("metoprolol succinate", "25 mg"), ("metoprolol succinate", "50 mg")),
    ),
    "prior_mi": ((("clopidogrel", "75 mg"), None),),
    "t2dm": (
        (("metformin", "1000 mg"), ("metformin", "500 mg")),
        (("empagliflozin", "10 mg"), ("sitagliptin", "100 mg"), ("insulin glargine", "20 units"),
         ("semaglutide", "0.5 mg"), ("glipizide", "5 mg"), None),
    ),
    "ckd3": ((("losartan", "50 mg"), None),),
    "htn": (
        (("losartan", "50 mg"), ("losartan", "100 mg"), ("lisinopril", "20 mg"), ("amlodipine", "5 mg")),
        (("amlodipine", "5 mg"), ("hydrochlorothiazide", "12.5 mg"), None, None),
    ),
    "hld": ((("atorvastatin", "40 mg"), ("atorvastatin", "20 mg"), ("rosuvastatin", "10 mg")),),
    "copd": (
        (("tiotropium", "18 mcg"), ("umeclidinium-vilanterol", "62.5-25 mcg")),
        (("albuterol", "90 mcg 2 puffs"),),
        (("fluticasone-salmeterol", "250-50 mcg"), None, None),
    ),
    "asthma": (
        (("fluticasone-salmeterol", "250-50 mcg"), ("fluticasone-salmeterol", "100-50 mcg")),
        (("albuterol", "90 mcg 2 puffs"),),
        (("montelukast", "10 mg"), None, None),
    ),
    "hypothyroid": ((("levothyroxine", "75 mcg"), ("levothyroxine", "88 mcg"), ("levothyroxine", "100 mcg")),),
    "mdd": ((("sertraline", "100 mg"), ("bupropion", "150 mg"), ("escitalopram", "10 mg")),),
    "gad": ((("escitalopram", "10 mg"), ("buspirone", "10 mg"), ("sertraline", "50 mg")),),
    "gerd": ((("omeprazole", "20 mg"), ("pantoprazole", "40 mg"), ("famotidine", "20 mg")),),
    "oa": (
        (("acetaminophen", "650 mg"),),
        (("diclofenac", "1% gel 4 g"), ("meloxicam", "7.5 mg"), None),
    ),
    "obesity": ((("semaglutide", "1 mg"), None, None),),
    "osa": (),
}

# --------------------------------------------------------------------------- medication changes


@dataclass(frozen=True)
class SwitchScenario:
    """A medication change at one encounter: ``from`` record is discontinued, ``to`` record (if any) starts.

    ``reason`` may contain ``{lab}`` (rendered lab value, e.g. "K 5.7"). ``lab_override`` forces the named
    lab test into an abnormal range at that encounter so the reason is supported by the lab table.
    ``hpi`` / ``plan`` may use ``{from}``, ``{to}``, ``{from_name}``, ``{to_name}``, ``{lab}``, ``{cond}``.
    """

    key: str
    condition: str
    from_med: tuple[str, str]
    to_med: tuple[str, str] | None
    reason: str
    complaint: str
    hpi: str
    plan: str
    lab_override: tuple[str, float, float] | None = None
    setting: str = "outpatient"      # outpatient | hf_exac | afib_rvr (inpatient-linked)


SWITCHES: tuple[SwitchScenario, ...] = (
    SwitchScenario(
        "acei_cough", "htn", ("lisinopril", "10 mg"), ("losartan", "50 mg"),
        "persistent dry cough (likely ACE inhibitor-related); switched to losartan",
        "Dry cough for 2 months.",
        "Reports a persistent nonproductive dry cough for about 2 months on {from}; no fever, SOB or "
        "postnasal drip.",
        "Dry cough most consistent with ACE inhibitor effect. Stop {from}; start {to}. BMP in 2 weeks.",
    ),
    SwitchScenario(
        "ccb_edema", "htn", ("amlodipine", "10 mg"), ("amlodipine", "5 mg"),
        "bilateral ankle edema; dose reduced to 5 mg",
        "Ankle swelling.",
        "Notes bilateral ankle swelling by evening on {from}; no orthopnea or PND.",
        "Dependent edema likely CCB-related. Reduce to {to} (from {from}).",
    ),
    SwitchScenario(
        "thiazide_hyponatremia", "htn", ("hydrochlorothiazide", "25 mg"), None,
        "hyponatremia ({lab}); stopped",
        "Fatigue and HTN follow-up.",
        "Reports mild fatigue over several weeks; taking {from} as prescribed.",
        "Hyponatremia ({lab}) attributed to thiazide. Stop {from}; recheck BMP in 1 week.",
        lab_override=("sodium", 129.0, 132.0),
    ),
    SwitchScenario(
        "statin_myalgia", "hld", ("simvastatin", "20 mg"), ("rosuvastatin", "10 mg"),
        "statin-associated myalgias; switched to rosuvastatin",
        "Muscle aches.",
        "Describes symmetric thigh aching for about 6 weeks on {from}; no weakness or dark urine.",
        "Statin-associated myalgias. Stop {from}; start {to} and reassess symptoms.",
    ),
    SwitchScenario(
        "statin_titration", "hld", ("atorvastatin", "20 mg"), ("atorvastatin", "40 mg"),
        "LDL above goal ({lab}); dose increased to 40 mg",
        "HLD follow-up.",
        "Reports adherence to {from}; diet has been inconsistent over the holidays.",
        "{lab}, above goal. Increase to {to} (from {from}); repeat lipid panel in 3 months.",
        lab_override=("ldl", 118.0, 142.0),
    ),
    SwitchScenario(
        "metformin_titration", "t2dm", ("metformin", "500 mg"), ("metformin", "1000 mg"),
        "A1c above goal ({lab}); dose increased to 1000 mg BID",
        "T2DM follow-up.",
        "Tolerating {from} without GI upset; fasting fingersticks remain in the 150s-170s.",
        "{lab}, above goal <7.0. Increase to {to} (from {from}).",
        lab_override=("hba1c", 7.5, 8.3),
    ),
    SwitchScenario(
        "sulfonylurea_hypoglycemia", "t2dm", ("glipizide", "5 mg"), ("empagliflozin", "10 mg"),
        "recurrent hypoglycemia (home fingersticks in the 50s); switched to empagliflozin",
        "Low fingerstick readings.",
        "Reports 3 episodes of shakiness and diaphoresis with home fingersticks in the 50s while on {from}.",
        "Recurrent hypoglycemia on sulfonylurea. Stop {from}; start {to}. Reviewed hypoglycemia precautions.",
    ),
    SwitchScenario(
        "metformin_gi", "t2dm", ("metformin", "1000 mg"), ("sitagliptin", "100 mg"),
        "GI intolerance (persistent diarrhea); switched to sitagliptin",
        "Diarrhea.",
        "Reports loose stools 3-4 times daily on {from} despite taking it with meals.",
        "GI intolerance to biguanide. Stop {from}; start {to}.",
    ),
    SwitchScenario(
        "mra_hyperkalemia", "hfref", ("spironolactone", "25 mg"), None,
        "hyperkalemia ({lab}); stopped",
        "HFrEF follow-up.",
        "Euvolemic by report; adherent to {from} and the rest of GDMT.",
        "Hyperkalemia ({lab}) on MRA. Stop {from}; low-potassium diet; recheck BMP in 1 week.",
        lab_override=("potassium", 5.6, 5.9),
    ),
    SwitchScenario(
        "bb_uptitration", "hfref", ("carvedilol", "6.25 mg"), ("carvedilol", "12.5 mg"),
        "uptitrated per GDMT; dose increased to 12.5 mg BID",
        "HFrEF follow-up.",
        "Tolerating {from} without dizziness; resting HR 80s at home.",
        "Room to uptitrate beta-blocker. Increase to {to} (from {from}).",
    ),
    SwitchScenario(
        "diuretic_increase", "hfref", ("furosemide", "20 mg"), ("furosemide", "40 mg"),
        "dose increased to 40 mg after HFrEF exacerbation admission",
        "", "", "Home {from} increased to {to} at discharge.",
        setting="hf_exac",
    ),
    SwitchScenario(
        "warfarin_labile", "afib", ("warfarin", "5 mg"), ("apixaban", "5 mg"),
        "labile INR ({lab}); switched to apixaban",
        "AFib anticoagulation follow-up.",
        "INR has been difficult to keep in range on {from} despite stable diet; no bleeding.",
        "Labile INR ({lab}) on vitamin K antagonist. Stop {from}; start {to} once INR <2.0.",
        lab_override=("inr", 3.6, 4.3),
    ),
    SwitchScenario(
        "rate_control_increase", "afib", ("metoprolol succinate", "50 mg"), ("metoprolol succinate", "100 mg"),
        "dose increased to 100 mg after AFib with RVR admission",
        "", "", "Home {from} increased to {to} for rate control.",
        setting="afib_rvr",
    ),
    SwitchScenario(
        "lama_escalation", "copd", ("tiotropium", "18 mcg"), ("umeclidinium-vilanterol", "62.5-25 mcg"),
        "persistent DOE on LAMA monotherapy; escalated to umeclidinium-vilanterol",
        "Worsening DOE.",
        "Reports DOE after one flight of stairs despite adherence to {from}; rescue inhaler use 4-5 times weekly.",
        "Persistent symptoms on LAMA. Stop {from}; start {to} (LAMA/LABA).",
    ),
    SwitchScenario(
        "montelukast_neuro", "asthma", ("montelukast", "10 mg"), None,
        "vivid dreams and irritability (neuropsychiatric effects); stopped",
        "Sleep disturbance.",
        "Reports vivid dreams and irritability while taking {from}.",
        "Possible neuropsychiatric effects of leukotriene antagonist. Stop {from}.",
    ),
    SwitchScenario(
        "levothyroxine_titration", "hypothyroid", ("levothyroxine", "50 mcg"), ("levothyroxine", "75 mcg"),
        "TSH above goal ({lab}); dose increased to 75 mcg",
        "Fatigue.",
        "Reports fatigue and cold intolerance; takes {from} on an empty stomach.",
        "{lab}, above goal. Increase to {to} (from {from}); repeat TSH in 6-8 weeks.",
        lab_override=("tsh", 6.2, 9.4),
    ),
    SwitchScenario(
        "ssri_titration", "mdd", ("sertraline", "50 mg"), ("sertraline", "100 mg"),
        "partial response; dose increased to 100 mg",
        "MDD follow-up.",
        "Partial improvement in mood and sleep on {from}; no SI.",
        "Partial response. Increase to {to} (from {from}).",
    ),
    SwitchScenario(
        "ssri_sexual", "mdd", ("sertraline", "100 mg"), ("bupropion", "150 mg"),
        "sexual side effects; switched to bupropion",
        "Medication side effects.",
        "Mood improved on {from} but reports sexual side effects affecting adherence.",
        "SSRI sexual dysfunction. Cross-taper off {from}; start {to}.",
    ),
    SwitchScenario(
        "escitalopram_gi", "gad", ("escitalopram", "10 mg"), ("buspirone", "10 mg"),
        "nausea and insomnia; switched to buspirone",
        "Medication side effects.",
        "Reports persistent nausea and insomnia on {from}.",
        "Intolerance to SSRI. Taper off {from}; start {to}.",
    ),
    SwitchScenario(
        "ppi_stepdown", "gerd", ("omeprazole", "40 mg"), ("omeprazole", "20 mg"),
        "symptoms controlled; stepped down to 20 mg",
        "GERD follow-up.",
        "Retrosternal burning resolved on {from}; no dysphagia or melena.",
        "Symptoms controlled. Step down to {to} (from {from}).",
    ),
    SwitchScenario(
        "nsaid_dyspepsia", "oa", ("meloxicam", "15 mg"), ("diclofenac", "1% gel 4 g"),
        "dyspepsia on oral NSAID; switched to topical diclofenac",
        "Knee pain and stomach upset.",
        "Knee pain partly controlled on {from} but reports epigastric discomfort after doses.",
        "Dyspepsia on oral NSAID. Stop {from}; start {to}.",
    ),
    SwitchScenario(
        "glp1_titration", "obesity", ("semaglutide", "0.25 mg"), ("semaglutide", "0.5 mg"),
        "dose escalation per titration schedule; increased to 0.5 mg weekly",
        "Weight management follow-up.",
        "Tolerating {from} with mild early satiety, no vomiting.",
        "Tolerating GLP-1 RA. Increase to {to} (from {from}) per titration schedule.",
    ),
)

# --------------------------------------------------------------------------- labs


@dataclass(frozen=True)
class LabTestSpec:
    key: str
    name: str                         # display name in lab panel documents
    short: str                        # label used in notes, always followed by a space and the value
    loinc: str
    unit: str
    panel: str
    decimals: int                     # number format: f"{value:.{decimals}f}"
    ref_low: float | None             # None = no lower bound
    ref_high: float | None            # None = no upper bound
    normal: tuple[float, float]       # plausible value range in a healthy adult
    abnormal: tuple[tuple[str, float, float], ...] = ()   # (condition/context, low, high)
    ref_female: tuple[float, float] | None = None          # sex-specific override of (ref_low, ref_high)


LAB_TESTS: dict[str, LabTestSpec] = {
    t.key: t
    for t in (
        LabTestSpec("hba1c", "Hemoglobin A1c (HbA1c)", "A1c", "4548-4", "%", "HbA1c", 1, 4.0, 5.6, (4.8, 5.6),
                    (("t2dm", 6.4, 9.6), ("obesity", 5.4, 6.2))),
        LabTestSpec("glucose", "Glucose", "glucose", "2345-7", "mg/dL", "BMP", 0, 70, 99, (80, 99),
                    (("t2dm", 110, 240),)),
        LabTestSpec("creatinine", "Creatinine (Cr)", "Cr", "2160-0", "mg/dL", "BMP", 2, 0.74, 1.35, (0.6, 1.1),
                    (("ckd3", 1.1, 1.8),), ref_female=(0.59, 1.04)),
        LabTestSpec("egfr", "eGFR (CKD-EPI 2021)", "eGFR", "98979-8", "mL/min/1.73m2", "BMP", 0, 60, None, (65, 110),
                    (("ckd3", 45, 59),)),
        LabTestSpec("bun", "Blood urea nitrogen (BUN)", "BUN", "3094-0", "mg/dL", "BMP", 0, 7, 20, (9, 19),
                    (("ckd3", 21, 34), ("hf_exac", 25, 40))),
        LabTestSpec("potassium", "Potassium (K)", "K", "2823-3", "mmol/L", "BMP", 1, 3.5, 5.1, (3.7, 4.8),
                    (("mra", 4.6, 5.9),)),
        LabTestSpec("sodium", "Sodium (Na)", "Na", "2951-2", "mmol/L", "BMP", 0, 136, 145, (136, 143),
                    (("thiazide", 129, 136), ("hfref", 132, 139))),
        LabTestSpec("ldl", "LDL cholesterol, calculated (LDL)", "LDL", "13457-7", "mg/dL", "Lipid panel", 0, 0, 99,
                    (75, 125), (("hld", 140, 190), ("statin", 55, 110))),
        LabTestSpec("hdl", "HDL cholesterol (HDL)", "HDL", "2085-9", "mg/dL", "Lipid panel", 0, 40, None, (40, 70),
                    (("t2dm", 32, 45),)),
        LabTestSpec("triglycerides", "Triglycerides (TG)", "TG", "2571-8", "mg/dL", "Lipid panel", 0, 0, 149,
                    (70, 145), (("t2dm", 140, 260),)),
        LabTestSpec("total_cholesterol", "Total cholesterol", "total cholesterol", "2093-3", "mg/dL", "Lipid panel",
                    0, 0, 199, (150, 199), (("hld", 210, 270),)),
        LabTestSpec("tsh", "Thyroid stimulating hormone (TSH)", "TSH", "3016-3", "mIU/L", "TSH", 2, 0.45, 4.50,
                    (0.8, 3.0), (("hypothyroid", 6.0, 9.5),)),
        LabTestSpec("bnp", "B-type natriuretic peptide (BNP)", "BNP", "30934-4", "pg/mL", "BNP", 0, 0, 99, (15, 85),
                    (("hfref", 250, 650), ("hf_exac", 1100, 2200), ("afib", 90, 190))),
        LabTestSpec("hemoglobin", "Hemoglobin (Hgb)", "Hgb", "718-7", "g/dL", "CBC", 1, 13.5, 17.5, (12.5, 16.0),
                    (("ckd3", 10.8, 12.6),), ref_female=(12.0, 15.5)),
        LabTestSpec("wbc", "White blood cell count (WBC)", "WBC", "6690-2", "K/uL", "CBC", 1, 4.5, 11.0, (5.0, 9.0),
                    (("cap", 13.5, 17.8),)),
        LabTestSpec("platelets", "Platelet count (Plt)", "Plt", "777-3", "K/uL", "CBC", 0, 150, 400, (180, 340)),
        LabTestSpec("alt", "Alanine aminotransferase (ALT)", "ALT", "1742-6", "U/L", "Hepatic function panel", 0, 7,
                    56, (14, 38)),
        LabTestSpec("ast", "Aspartate aminotransferase (AST)", "AST", "1920-8", "U/L", "Hepatic function panel", 0,
                    10, 40, (15, 34)),
        LabTestSpec("inr", "INR", "INR", "6301-6", "", "Coagulation", 1, 0.8, 1.1, (0.9, 1.1),
                    (("warfarin", 2.0, 3.0),)),
    )
}

PANEL_ORDER: tuple[str, ...] = ("BMP", "HbA1c", "Lipid panel", "TSH", "BNP", "CBC", "Hepatic function panel",
                                "Coagulation")
PANEL_TESTS: dict[str, tuple[str, ...]] = {
    panel: tuple(k for k, t in LAB_TESTS.items() if t.panel == panel) for panel in PANEL_ORDER
}

# --------------------------------------------------------------------------- allergens


@dataclass(frozen=True)
class AllergenSpec:
    substance: str
    reactions: tuple[tuple[str, str], ...]   # (reaction, severity)
    excludes: tuple[str, ...] = ()           # medication names that must not be prescribed


ALLERGENS: tuple[AllergenSpec, ...] = (
    AllergenSpec("penicillin", (("hives", "moderate"), ("anaphylaxis", "severe")), ("amoxicillin-clavulanate",)),
    AllergenSpec("sulfamethoxazole-trimethoprim", (("diffuse maculopapular rash", "moderate"),)),
    AllergenSpec("codeine", (("nausea and vomiting", "mild"),)),
    AllergenSpec("morphine", (("pruritus", "mild"), ("hypotension", "moderate"))),
    AllergenSpec("iodinated contrast", (("hives", "moderate"),)),
    AllergenSpec("latex", (("contact dermatitis", "mild"),)),
    AllergenSpec("ciprofloxacin", (("Achilles tendinopathy", "moderate"),), ("levofloxacin",)),
    AllergenSpec("cephalexin", (("rash", "mild"),)),
    AllergenSpec("ibuprofen", (("angioedema", "severe"), ("bronchospasm", "severe")),
                 ("meloxicam", "diclofenac", "aspirin")),
    AllergenSpec("doxycycline", (("photosensitivity rash", "mild"),), ("doxycycline",)),
    AllergenSpec("vancomycin", (("infusion reaction with flushing", "moderate"),)),
    AllergenSpec("gabapentin", (("dizziness and sedation", "mild"),)),
)
NKDA_PROBABILITY = 0.30
MAX_PATIENTS_PER_ALLERGEN = 3      # keeps allergies distinctive (shared by at most 2 other patients)

# --------------------------------------------------------------------------- people (all invented)

FEMALE_FIRST_NAMES: tuple[str, ...] = (
    "Bettina", "Delphine", "Florentina", "Henrietta", "Josefina", "Marisol", "Ophelia", "Rosalind", "Theodora",
    "Valentina", "Ximena", "Zinnia", "Alma", "Celestine", "Evangeline", "Liesel", "Maribel", "Odessa",
    "Perpetua", "Wilhelmina",
)
MALE_FIRST_NAMES: tuple[str, ...] = (
    "Aurelio", "Cassius", "Ezekiel", "Gideon", "Ignatius", "Lysander", "Nikolai", "Percival", "Silvio", "Ulysses",
    "Wendell", "Yusuf", "Bertram", "Dashiell", "Fitzroy", "Horatio", "Leopold", "Osric", "Quentin", "Thaddeus",
)
LAST_NAMES: tuple[str, ...] = (
    "Ambersley", "Brindlecourt", "Cindermoss", "Drummelow", "Eskerwood", "Fenwhistle", "Gorsebrook", "Hallowmere",
    "Ivelstrand", "Juniperson", "Kettlewick", "Larkspurr", "Mossenden", "Nettlebrook", "Orrinsdale", "Pebbleford",
    "Quillfeather", "Rookmanor", "Saltershaw", "Thistlecombe", "Umberlay", "Vantrellis", "Willowmarsh",
    "Yarrowdeen", "Zennorick", "Brackwater", "Dovecrest", "Emberlyn", "Foxtaverne", "Glimmerholt",
)

PROVIDERS: dict[str, tuple[str, ...]] = {
    "primary_care": (
        "Dr. Imani Okafor (synthetic)", "Dr. Teodor Vashti (synthetic)", "Dr. Linnea Brask (synthetic)",
        "Dr. Rafferty Quon (synthetic)", "Dr. Selene Marwick (synthetic)", "Dr. Oduya Pellinore (synthetic)",
    ),
    "cardiology": ("Dr. Caspian Ilori (synthetic)", "Dr. Henrike Saldana-Moor (synthetic)"),
    "pulmonology": ("Dr. Mireille Haddow (synthetic)",),
    "nephrology": ("Dr. Anselm Tkachenko-Ruiz (synthetic)",),
    "endocrinology": ("Dr. Priya Velanne (synthetic)",),
    "behavioral_health": ("Dr. Joaquin Restrepo-Lind (synthetic)",),
    "emergency": ("Dr. Beatrix Onwuachi (synthetic)", "Dr. Harlan Mbeki-Ross (synthetic)"),
    "hospital_medicine": ("Dr. Yusra Kettleby (synthetic)", "Dr. Emeric Dunmore (synthetic)"),
    "radiology": ("Dr. Soraya Lindqvist-Amari (synthetic)", "Dr. Thaddeus Okonkwo-Vale (synthetic)"),
}

SOCIAL_TOBACCO: tuple[str, ...] = ("never smoker", "never smoker", "former smoker, quit decades ago")
SOCIAL_ALCOHOL: tuple[str, ...] = ("none", "rare", "1-2 drinks per week", "socially")
SOCIAL_LIVING: tuple[str, ...] = ("with spouse", "alone", "with adult daughter", "with partner",
                                  "with son and his family")
SOCIAL_OCCUPATION: tuple[str, ...] = ("a retired teacher", "a bus mechanic", "an accountant",
                                      "a retired postal carrier", "a home health aide", "a librarian", "a carpenter",
                                      "a retired chef")

SPECIALTY_FOR_CONDITION: dict[str, str] = {
    "ckd3": "nephrology", "t2dm": "endocrinology", "hypothyroid": "endocrinology", "hfref": "cardiology",
    "afib": "cardiology", "cad": "cardiology", "copd": "pulmonology", "asthma": "pulmonology",
    "osa": "pulmonology", "mdd": "behavioral_health", "gad": "behavioral_health",
}
SPECIALTY_TITLES: dict[str, str] = {
    "nephrology": "Nephrology", "endocrinology": "Endocrinology", "cardiology": "Cardiology",
    "pulmonology": "Pulmonology", "behavioral_health": "Behavioral Health",
}

# --------------------------------------------------------------------------- note phrasing

# Per-condition phrase variants. ``{cond}`` -> abbreviation, ``{meds}`` -> that condition's active meds with dose.
# Variants containing ``{meds}`` are only used when the condition has an active medication.
NOTE_PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "t2dm": {
        "hpi": (
            "{cond} on {meds}; checks fasting fingersticks most mornings, mostly 110s-150s, no hypoglycemic episodes.",
            "For {cond}, reports good adherence to {meds} and denies polyuria, polydipsia or blurred vision.",
            "{cond} managed with {meds}; follows a carbohydrate-controlled diet with variable success.",
        ),
        "plan": (
            "Continue {meds}. Reinforced diet and exercise; retinal exam and monofilament foot exam due.",
            "Continue {meds}; recheck A1c in 3 months.",
            "Continue {meds}. Discussed carbohydrate counting and post-meal walks.",
        ),
    },
    "htn": {
        "hpi": (
            "{cond} on {meds}; home BP log mostly 120s-130s systolic.",
            "{cond}: tolerating {meds} without lightheadedness or orthostasis.",
            "{cond}: home readings variable, 130s-140s systolic; no headaches or vision changes.",
        ),
        "plan": (
            "Continue {meds}; low-sodium diet and home BP log.",
            "Continue {meds}. DASH diet reviewed; bring home cuff to next visit.",
            "Continue current regimen; recheck BP at next visit.",
        ),
    },
    "hld": {
        "hpi": (
            "{cond} on {meds}, no myalgias.",
            "{cond}: adherent to {meds}; trying to limit saturated fat.",
        ),
        "plan": (
            "Continue {meds}.",
            "Continue {meds}; lipids annually.",
        ),
    },
    "ckd3": {
        "hpi": (
            "{cond}: avoids NSAIDs; no edema or change in urine output.",
            "{cond} followed with periodic BMP; no recent nephrotoxin exposure.",
        ),
        "plan": (
            "Avoid nephrotoxins and NSAIDs; renally dose medications.",
            "Continue BP control; urine albumin-to-creatinine ratio with next labs.",
        ),
    },
    "cad": {
        "hpi": (
            "{cond}: denies exertional chest pain or pressure; walks about 20 minutes most days.",
            "{cond}: no angina, DOE or palpitations; adherent to {meds}.",
        ),
        "plan": (
            "No anginal symptoms. Continue {meds}.",
            "Stable. Continue {meds}; encouraged regular aerobic activity.",
        ),
    },
    "prior_mi": {
        "hpi": (
            "{cond} with PCI, completed cardiac rehab; no recurrent chest pain.",
            "{cond}; no recurrent angina since PCI.",
        ),
        "plan": (
            "Secondary prevention reviewed; continue {meds}.",
            "Secondary prevention: BP, lipid and glucose targets reviewed.",
        ),
    },
    "hfref": {
        "hpi": (
            "{cond}: NYHA class II, stable 2-pillow orthopnea, no PND; weighs self daily.",
            "{cond}: DOE with brisk walking only, no orthopnea or PND; adherent to {meds}.",
        ),
        "plan": (
            "Euvolemic on exam. Continue {meds}; daily weights and 2 g sodium restriction.",
            "Compensated. Continue {meds}; call for weight gain over 1 kg in a day.",
        ),
    },
    "afib": {
        "hpi": (
            "{cond}: occasional palpitations, no syncope or bleeding; on {meds}.",
            "{cond}: denies palpitations, presyncope or bleeding on {meds}.",
        ),
        "plan": (
            "Rate controlled. Continue {meds}.",
            "Asymptomatic, rate controlled. Continue {meds}; bleeding precautions reviewed.",
        ),
    },
    "copd": {
        "hpi": (
            "{cond}: chronic DOE with one flight of stairs, rescue inhaler about twice weekly; former smoker.",
            "{cond}: stable cough with scant white sputum; on {meds}.",
        ),
        "plan": (
            "Continue {meds}; pulmonary rehab discussed; vaccines up to date.",
            "Continue {meds}; reviewed inhaler technique and COPD action plan.",
        ),
    },
    "asthma": {
        "hpi": (
            "{cond}: nighttime symptoms about once weekly; rescue inhaler 2-3 times weekly.",
            "{cond}: well controlled on {meds}; no nocturnal awakenings.",
        ),
        "plan": (
            "Continue {meds}; reviewed inhaler technique and asthma action plan.",
            "Continue {meds}; avoid known triggers; flu vaccine this fall.",
        ),
    },
    "hypothyroid": {
        "hpi": (
            "{cond} on {meds}; energy stable, no palpitations or tremor.",
            "{cond}: takes {meds} on an empty stomach; no cold intolerance.",
        ),
        "plan": (
            "Continue {meds}.",
            "Continue {meds}; separate from calcium and iron supplements.",
        ),
    },
    "mdd": {
        "hpi": (
            "{cond}: mood improved, sleeping 6-7 hours; denies SI or HI.",
            "{cond} on {meds}; engaged with therapy, denies SI.",
        ),
        "plan": (
            "Continue {meds}; continue psychotherapy.",
            "Stable on {meds}; safety plan reviewed.",
        ),
    },
    "gad": {
        "hpi": (
            "{cond}: worry interfering with sleep at times; no panic episodes.",
            "{cond}: symptoms manageable on {meds}.",
        ),
        "plan": (
            "Continue {meds}; CBT referral offered.",
            "Continue {meds}; sleep hygiene reviewed.",
        ),
    },
    "gerd": {
        "hpi": (
            "{cond}: occasional postprandial retrosternal burning, worse lying flat; on {meds}.",
            "{cond}: symptoms controlled on {meds}; no dysphagia.",
        ),
        "plan": (
            "Continue {meds}; avoid late meals and elevate head of bed.",
            "Controlled on {meds}.",
        ),
    },
    "oa": {
        "hpi": (
            "{cond}: bilateral knee pain with prolonged walking, morning stiffness under 30 minutes.",
            "{cond}: knee pain controlled with {meds}.",
        ),
        "plan": (
            "Continue {meds}; PT home exercise program.",
            "Continue {meds}; quadriceps strengthening and activity pacing.",
        ),
    },
    "osa": {
        "hpi": (
            "{cond}: uses CPAP nightly with good adherence; less daytime somnolence.",
            "{cond} on CPAP, mask fit comfortable.",
        ),
        "plan": (
            "Continue CPAP nightly.",
            "CPAP adherence good; continue.",
        ),
    },
    "obesity": {
        "hpi": (
            "{cond}: working on portion control and walking.",
            "{cond}: enrolled in a nutrition program.",
        ),
        "plan": (
            "Lifestyle counseling provided; goal 5-10% weight loss.",
            "Continue {meds}; lifestyle counseling provided.",
        ),
    },
    "cap": {"hpi": ("{cond}, completed antibiotics.",), "plan": ("Resolved.",)},
}

HPI_OPENERS: tuple[str, ...] = (
    "{age} y/o {sex_short} with {pmh} presents for {visit}.",
    "{age}-year-old {sex_word} with PMH of {pmh} seen today for {visit}.",
    "Patient is {article} {age} y/o {sex_word} with {pmh}, here for {visit}.",
    "{age}{sex_letter} with {pmh} returns for {visit}.",
)
PLAN_CLOSERS: tuple[str, ...] = (
    "RTC in 3 months with labs prior.",
    "Follow up in 3-4 months, sooner PRN.",
    "Return visit in 3 months; call with new symptoms.",
    "Next visit in 4 months with fasting labs.",
)
GENERAL_EXAM: tuple[str, ...] = (
    "Gen: well-appearing, NAD.", "General: alert, conversant, NAD.", "Gen: A&O x3, NAD.",
)
NEURO_EXAM: tuple[str, ...] = (
    "Neuro: CN II-XII grossly intact, strength 5/5 throughout.", "Neuro: nonfocal, gait steady.",
    "Neuro: no focal deficits.",
)
PSYCH_EXAM: tuple[str, ...] = ("Psych: normal mood and affect.", "Psych: appropriate affect, linear thought process.")


def all_lay_terms() -> list[str]:
    """Every lay term across conditions (lower-case, de-duplicated, sorted)."""
    return sorted({t.lower() for c in CONDITIONS.values() for t in c.lay_terms})


def med_spec(name: str) -> MedicationSpec:
    return MEDICATIONS[name]


def lab_spec(key: str) -> LabTestSpec:
    return LAB_TESTS[key]


def ref_range(test_key: str, sex: str) -> tuple[float | None, float | None]:
    """Reference range for a test, honouring sex-specific ranges."""
    t = LAB_TESTS[test_key]
    if sex == "female" and t.ref_female is not None:
        return t.ref_female
    return t.ref_low, t.ref_high
