"""Guards that keep non-synthetic or PHI-looking data out of the index and away from external LLMs.

``assert_synthetic`` is a hard gate on the ``synthetic: True`` metadata flag. ``scan_phi`` is a
regex screen for identifiers that the synthetic generator never produces (SSNs, phone numbers, e-mail
addresses, URLs, street addresses, MRNs without the ``SYN-`` prefix). It is a tripwire, not a
de-identification tool: a clean scan does not prove a text is free of PHI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Union

from ehr_rag.types import Chunk, Document, RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_IDS_IN_MESSAGE = 20


class NonSyntheticDataError(ValueError):
    """Raised when an item is not explicitly flagged ``synthetic: True``."""

    def __init__(self, offending_ids: list[str]):
        self.offending_ids = list(offending_ids)
        shown = ", ".join(self.offending_ids[:_MAX_IDS_IN_MESSAGE])
        more = len(self.offending_ids) - _MAX_IDS_IN_MESSAGE
        suffix = f" (+{more} more)" if more > 0 else ""
        super().__init__(
            f"{len(self.offending_ids)} item(s) lack metadata synthetic=True: {shown}{suffix}. "
            "EHR-RAG only accepts synthetic data from ehr_rag.synth."
        )


@dataclass
class PHIFinding:
    kind: str     # "ssn" | "phone" | "email" | "url" | "street_address" | "mrn"
    match: str
    start: int
    end: int


# --------------------------------------------------------------------------- synthetic gate


def _item_id(item: object) -> str:
    for attr in ("chunk_id", "doc_id"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return repr(item)[:80]


def assert_synthetic(items: Iterable[Union[Document, Chunk, RetrievedChunk]]) -> None:
    """Raise :class:`NonSyntheticDataError` naming every item whose metadata lacks ``synthetic=True``.

    Accepts Documents, Chunks, and (for the generation layer's convenience) RetrievedChunks, which are
    checked via their ``.chunk``. Only the boolean ``True`` passes; ``"true"``/``1`` do not.
    """
    offending: list[str] = []
    for item in items:
        target = item.chunk if isinstance(item, RetrievedChunk) else item
        metadata = getattr(target, "metadata", None)
        if not isinstance(metadata, dict) or metadata.get("synthetic") is not True:
            offending.append(_item_id(target))
    if offending:
        logger.error("Blocked %d non-synthetic item(s)", len(offending))
        raise NonSyntheticDataError(offending)


# --------------------------------------------------------------------------- PHI regexes

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|"
    r"Circle|Cir|Parkway|Pkwy|Highway|Hwy|Square|Sq|Trail|Trl)"
)

# Order matters only for reporting; overlapping findings are resolved in ``scan_phi``.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 123-45-6789 / 123 45 6789 with the same separator; excludes impossible area/group/serial numbers.
    ("ssn", re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}([- ])(?!00)\d{2}\1(?!0000)\d{4}(?![\w-])")),
    # NANP numbers: (713) 555-0142, 713-555-0142, 713.555.0142, +1 713 555 0142. Bare space-separated
    # digit runs are only accepted with a +1 prefix to avoid flagging tables of numbers.
    (
        "phone",
        re.compile(
            r"(?<![\w.-])(?:"
            r"(?:\+?1[-. ]?)?\([2-9]\d{2}\)\s?[2-9]\d{2}[-. ]\d{4}"
            r"|(?:\+?1[-.])?[2-9]\d{2}(?P<sep>[-.])[2-9]\d{2}(?P=sep)\d{4}"
            r"|\+1 [2-9]\d{2} [2-9]\d{2} \d{4}"
            r")(?![\w-])"
        ),
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"\b(?:https?://|ftp://|www\.)[^\s<>\"']+", re.IGNORECASE)),
    # "123 Main St", "4500 Oak Hollow Drive" — house number + 1-4 capitalised words + street suffix.
    (
        "street_address",
        re.compile(rf"\b\d{{1,6}}(?:\s+(?:[A-Z][a-z]+|[NSEW]\.?|\d+(?:st|nd|rd|th))){{1,4}}\s+{_STREET_SUFFIX}\b"),
    ),
    # Labelled record numbers that are not synthetic ("MRN: 00482913", "MR# A-123456").
    (
        "mrn",
        re.compile(
            r"\b(?:MRN|MR\s?#|MR\s?No\.?|Medical\s+Record\s+(?:Number|No\.?|#))\s*[:#]?\s*"
            r"(?!SYN-)[A-Z]{0,4}-?\d{5,}\b",
            re.IGNORECASE,
        ),
    ),
    # Unlabelled MRN-like ids: upper-case prefix + dash + 6-10 digits, except the synthetic SYN- prefix.
    ("mrn", re.compile(r"(?<![\w-])(?!SYN-)[A-Z]{2,5}-\d{6,10}(?![\w-])")),
)

_TRAILING_PUNCT = ".,;:!?)]}'\""


def scan_phi(text: str) -> list[PHIFinding]:
    """Return PHI-looking matches in ``text`` ordered by position.

    A match fully contained in a longer match (e.g. a phone number inside a URL) is dropped.
    ISO dates, lab values, doses, blood pressures, LOINC/ICD-10 codes and ``SYN-`` ids are not flagged.
    """
    raw: list[PHIFinding] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if kind == "url":
                while end > start and text[end - 1] in _TRAILING_PUNCT:
                    end -= 1
            raw.append(PHIFinding(kind=kind, match=text[start:end], start=start, end=end))

    raw.sort(key=lambda f: (f.start, -(f.end - f.start)))
    findings: list[PHIFinding] = []
    for f in raw:
        if any(o.start <= f.start and f.end <= o.end for o in findings):
            continue
        findings.append(f)
    return findings


def check_documents(docs: Iterable[Document]) -> dict[str, list[PHIFinding]]:
    """Scan each document's text; return ``{doc_id: findings}`` for documents with at least one finding."""
    results: dict[str, list[PHIFinding]] = {}
    for doc in docs:
        findings = scan_phi(doc.text)
        if findings:
            results[doc.doc_id] = findings
    if results:
        logger.warning("PHI-like patterns found in %d document(s): %s", len(results), sorted(results)[:10])
    return results
