"""Section-aware chunking of :class:`~ehr_rag.types.Document` objects into :class:`~ehr_rag.types.Chunk` objects.

* Clinical notes (``doc_type == "clinical_note"``) are split on upper-case section header lines
  (``SECTION NAME:``). Each section yields one or more chunks; long sections are split on sentence
  boundaries and ``overlap_sentences`` trailing sentences are carried into the next chunk of the *same*
  section only. Empty sections are skipped.
* Structured documents (lists, lab panels, vitals, demographics) are rendered one record per line; whole
  lines are grouped up to the size budget and a line is never split (a line longer than the budget becomes
  its own chunk).

Size budget: ``max_chars`` bounds the whole chunk text (header line + newline + body) wherever possible.
A single sentence or record line that does not fit on its own is emitted as an oversized chunk rather
than being cut.

Title line repetition (structured docs): when the first non-empty line of a structured document is a
short, non-bullet descriptive line (e.g. ``"Active and past medications as of 2026-06-30:"``) and the
remaining lines are bullet records, that title line is repeated at the top of every continuation chunk so
each chunk still says what the list is. Documents without such a line (or whose first line is itself a
record) are grouped as-is; the header line already carries patient, title and date.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Iterable

from ehr_rag.types import NOTE_TYPES, Chunk, Document, chunk_id_for

logger = logging.getLogger(__name__)

SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 /&(),'-]+:$")

# Words kept upper-case when prettifying a section name for the header line.
_SECTION_ACRONYMS = frozenset(
    {"HPI", "ROS", "ED", "ICU", "EKG", "ECG", "CT", "MRI", "CXR", "PMH", "PSH", "ID", "HEENT", "GI", "GU",
     "CV", "MDM", "PE", "ICD", "CBC", "BMP", "CMP", "A/P", "S/P", "PRN", "NKDA", "LV", "EF"}
)
_SECTION_SMALL_WORDS = frozenset({"and", "or", "of", "the", "for", "to", "in", "on", "with", "a", "an", "at"})

# Tokens (lower-cased, without the trailing period) after which a period never ends a sentence.
_ABBREVIATIONS = frozenset(
    {
        # titles / general
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no", "pt", "pts",
        "fig", "hx", "dx", "tx", "rx", "sx", "fx", "yr", "yrs", "y.o", "wk", "wks", "mo", "mos", "min", "hr",
        "hrs", "sec", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        # dosing frequencies / routes
        "q", "q.d", "qd", "b.i.d", "bid", "t.i.d", "tid", "q.i.d", "qid", "q.h.s", "qhs", "h.s", "hs", "p.o",
        "po", "p.r.n", "prn", "a.c", "p.c", "q.a.m", "q.p.m", "q.o.d", "i.v", "i.m", "s.c", "subq", "s.l",
        "q4h", "q6h", "q8h", "q12h",
        # units
        "mg", "mcg", "g", "kg", "lb", "lbs", "ml", "l", "dl", "mmol", "meq", "u", "iu", "oz", "cm", "mm", "in",
        "ft", "tab", "tabs", "cap", "caps",
    }
)

# Candidate boundary: terminal punctuation (+ optional closing quote/paren) followed by whitespace.
_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]*\s+")
_PREV_TOKEN_RE = re.compile(r"(\S+)$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


# --------------------------------------------------------------------------- header


def _display_section(section: str) -> str:
    """``"HOSPITAL COURSE"`` -> ``"Hospital Course"``; mixed-case names are returned unchanged."""
    name = section.strip().rstrip(":").strip()
    if name != name.upper():
        return name
    words = []
    for i, word in enumerate(name.split()):
        if word in _SECTION_ACRONYMS:
            words.append(word)
        elif i > 0 and word.lower() in _SECTION_SMALL_WORDS:
            words.append(word.lower())
        else:
            words.append("/".join(part.capitalize() for part in word.split("/")))
    return " ".join(words)


def header_for(doc: Document, section: str | None) -> str:
    """Context line prepended to every chunk.

    ``"[Patient P001 | Discharge Summary | 2025-03-14 | Section: Hospital Course]"``; the section part is
    omitted when ``section`` is None. Upper-case section names are shown in title case.
    """
    parts = [f"Patient {doc.patient_id}", doc.title, doc.date]
    if section:
        parts.append(f"Section: {_display_section(section)}")
    return "[" + " | ".join(p for p in parts if p) + "]"


# --------------------------------------------------------------------------- sentences


def _is_boundary(line: str, punct_pos: int, next_pos: int) -> bool:
    """Decide whether the terminal punctuation at ``punct_pos`` ends a sentence."""
    if next_pos >= len(line):
        return False
    nxt = line[next_pos]
    # Next sentence must start with an upper-case letter, digit, or opening bracket/quote.
    if not (nxt.isupper() or nxt.isdigit() or nxt in "([\"'"):
        return False
    if line[punct_pos] != ".":
        return True
    m = _PREV_TOKEN_RE.search(line[:punct_pos])
    if not m:
        return True
    token = m.group(1).lstrip("([\"'").lower()
    if token.isdigit() and not line[: m.start(1)].strip():  # list marker: "1. HFrEF ..."
        return False
    if token in _ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha():  # initials: "J. Smith"
        return False
    if re.fullmatch(r"(?:[a-z]\.)+[a-z]", token):  # dotted abbreviations not listed: "a.m", "u.s"
        return False
    return True


def _split_line(line: str) -> list[str]:
    """Split one line into sentences without breaking decimals, abbreviations or dates."""
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(line):
        if _is_boundary(line, m.start(), m.end()):
            piece = line[start:m.end()].strip()
            if piece:
                sentences.append(piece)
            start = m.end()
    tail = line[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[tuple[str, bool]]:
    """Split text into ``(sentence, starts_new_line)`` pairs.

    Line breaks are always boundaries (list items stay separate); within a line, sentences end at
    ``. ! ?`` followed by whitespace and an upper-case letter/digit, except after known abbreviations
    (``Dr.``, ``q.d.``, ``b.i.d.``, ``mg.``) and single-letter initials. Decimals (``7.2``) and dates
    contain no whitespace after the period and are never split.
    """
    out: list[tuple[str, bool]] = []
    for line in text.splitlines():
        for j, sentence in enumerate(_split_line(line)):
            out.append((sentence, j == 0))
    return out


def _join(units: list[tuple[str, bool]]) -> str:
    parts: list[str] = []
    for i, (text, newline) in enumerate(units):
        if i > 0:
            parts.append("\n" if newline else " ")
        parts.append(text)
    return "".join(parts)


def _joined_len(units: list[tuple[str, bool]]) -> int:
    return sum(len(t) for t, _ in units) + max(len(units) - 1, 0)


def _pack_sentences(
    units: list[tuple[str, bool]], budget: int, overlap_sentences: int
) -> list[str]:
    """Greedy sentence packing with ``overlap_sentences`` carried between consecutive bodies."""
    bodies: list[str] = []
    current: list[tuple[str, bool]] = []
    n_new = 0  # sentences in ``current`` not carried over from the previous body
    for unit in units:
        if current and n_new > 0 and _joined_len(current + [unit]) > budget:
            bodies.append(_join(current))
            carry = current[-overlap_sentences:] if overlap_sentences > 0 else []
            # Drop carried sentences (oldest first) until the new sentence fits.
            while carry and _joined_len(carry + [unit]) > budget:
                carry = carry[1:]
            current = list(carry)
            n_new = 0
        current.append(unit)
        n_new += 1
    if current and n_new > 0:
        bodies.append(_join(current))
    return bodies


# --------------------------------------------------------------------------- lines


def _pack_lines(lines: list[str], budget: int, title_line: str | None) -> list[str]:
    """Group whole lines into bodies of at most ``budget`` chars; never split a line."""
    bodies: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = current + [line]
        if current and len("\n".join(candidate)) > budget:
            bodies.append("\n".join(current))
            current = [title_line] if title_line else []
            if current and len("\n".join(current + [line])) > budget:
                current = []  # title + this line would not fit: emit the line on its own
        current.append(line)
    if current and not (title_line and current == [title_line]):
        bodies.append("\n".join(current))
    return bodies


def _title_line(lines: list[str], budget: int) -> str | None:
    """Return the first line if it is a short descriptive (non-record) line worth repeating."""
    if len(lines) < 2:
        return None
    first = lines[0]
    if _BULLET_RE.match(first) or len(first) > min(200, budget // 3):
        return None
    if not all(_BULLET_RE.match(line) for line in lines[1:]):
        return None
    return first


# --------------------------------------------------------------------------- documents


def _note_sections(text: str) -> list[tuple[str | None, str]]:
    """Split note text into ``(section_name, body)`` pairs; text before the first header has no section."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        stripped = line.strip()
        if SECTION_HEADER_RE.match(stripped):
            sections.append((stripped[:-1].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(body).strip()) for name, body in sections]


def _note_type(doc: Document) -> str | None:
    note_type = doc.metadata.get("note_type")
    if note_type:
        return str(note_type)
    if doc.doc_type != "clinical_note":
        return None
    guess = re.sub(r"[^a-z]+", "_", doc.title.lower()).strip("_")
    if guess in NOTE_TYPES:
        return guess
    if guess == "consult" or guess.startswith("consult"):
        return "consult_note"
    return None


def _budget(header: str, max_chars: int) -> int:
    return max(max_chars - len(header) - 1, 1)


def chunk_document(doc: Document, max_chars: int = 900, overlap_sentences: int = 1) -> list[Chunk]:
    """Chunk one document. Chunk ids are ``chunk_id_for(doc.doc_id, i)`` with ``i`` counting from 0."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    pieces: list[tuple[str | None, str]] = []  # (section, body)
    if doc.doc_type == "clinical_note":
        for section, body in _note_sections(doc.text):
            if not body:
                continue
            units = split_sentences(body)
            budget = _budget(header_for(doc, section), max_chars)
            pieces.extend((section, b) for b in _pack_sentences(units, budget, overlap_sentences))
    else:
        lines = [line.rstrip() for line in doc.text.splitlines() if line.strip()]
        if lines:
            budget = _budget(header_for(doc, None), max_chars)
            title = _title_line(lines, budget)
            pieces.extend((None, b) for b in _pack_lines(lines, budget, title))

    if not pieces:
        logger.warning("Document %s produced no chunks (empty text)", doc.doc_id)
        return []

    metadata = copy.deepcopy(doc.metadata)
    note_type = _note_type(doc)
    if note_type:
        metadata["note_type"] = note_type

    chunks = []
    for i, (section, body) in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(doc.doc_id, i),
                doc_id=doc.doc_id,
                patient_id=doc.patient_id,
                doc_type=doc.doc_type,
                date=doc.date,
                title=doc.title,
                section=section,
                text=header_for(doc, section) + "\n" + body,
                metadata=copy.deepcopy(metadata),
            )
        )
    return chunks


def chunk_documents(
    docs: Iterable[Document], max_chars: int = 900, overlap_sentences: int = 1
) -> list[Chunk]:
    """Chunk many documents; raises ``ValueError`` if two documents share a ``doc_id``."""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc.doc_id!r}: chunk ids would collide")
        seen.add(doc.doc_id)
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap_sentences=overlap_sentences))
    logger.info("Chunked %d documents into %d chunks", len(seen), len(chunks))
    return chunks
