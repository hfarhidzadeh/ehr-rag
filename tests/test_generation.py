"""Tests for ehr_rag.generation (no network unless credentials are present for the one live test)."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from ehr_rag.config import load_settings
from ehr_rag.generation import (
    ANSWER_SCHEMA,
    FALLBACK_BETA,
    SYSTEM_PROMPT,
    ClaudeGenerator,
    ExtractiveGenerator,
    GenerationError,
    GenerationResult,
    format_context,
    get_generator,
    has_claude_credentials,
)
from ehr_rag.ingest.chunking import chunk_documents
from ehr_rag.ingest.phi_guard import NonSyntheticDataError
from ehr_rag.synth.generator import generate_dataset
from ehr_rag.types import Chunk, RetrievedChunk

PID = "P001"


# --------------------------------------------------------------------------- fixtures / fakes


def make_chunk(cid: str, doc_type: str, date: str, title: str, body: str, *, pid: str = PID,
               section: str | None = None, synthetic: object = True) -> Chunk:
    header = f"[Patient {pid} | {title} | {date}" + (f" | Section: {section.title()}" if section else "") + "]"
    return Chunk(
        chunk_id=f"{cid}::000", doc_id=cid, patient_id=pid, doc_type=doc_type, date=date, title=title,
        section=section, text=f"{header}\n{body}", metadata={"synthetic": synthetic, "patient_id": pid},
    )


def rc(chunks: list[Chunk]) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=c, score=1.0 / (i + 1), rank=i + 1, source="hybrid") for i, c in enumerate(chunks)]


@pytest.fixture
def contexts() -> list[RetrievedChunk]:
    return rc([
        make_chunk("P001-labs-2025-03-03", "lab_panel", "2025-03-03", "Lab Panel 2025-03-03",
                   "Lab results drawn 2025-03-03 (outpatient encounter P001-E03); panels: A1c.\n"
                   "- Hemoglobin A1c (HbA1c): 8.1 % (ref 4.0-5.6) [H]"),
        make_chunk("P001-allergies", "allergy_list", "2026-06-06", "Allergy List",
                   "Allergy list as of 2026-06-06.\n- ibuprofen — bronchospasm (severe)"),
        make_chunk("P001-labs-2026-06-06", "lab_panel", "2026-06-06", "Lab Panel 2026-06-06",
                   "Lab results drawn 2026-06-06 (outpatient encounter P001-E07); panels: A1c.\n"
                   "- Hemoglobin A1c (HbA1c): 7.2 % (ref 4.0-5.6) [H]"),
        make_chunk("P001-N04", "clinical_note", "2025-07-10", "Primary Care Progress Note",
                   "T2DM: A1c 7.6, above goal <7.0. Continue metformin 1000 mg PO BID. " + "Diet reviewed. " * 20,
                   section="ASSESSMENT AND PLAN"),
    ])


class FakeMessages:
    def __init__(self, response=None, exc: Exception | None = None):
        self.response, self.exc, self.calls = response, exc, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, response=None, exc: Exception | None = None):
        self.messages = FakeMessages(response, exc)
        self.beta = SimpleNamespace(messages=FakeMessages(response, exc))

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls + self.beta.messages.calls


def fake_response(payload: dict | str | None, stop_reason: str = "end_turn", **extra):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    content = [SimpleNamespace(type="thinking", thinking="")]
    if payload is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, stop_reason=stop_reason, model="claude-opus-5-5",
                           stop_details=extra.get("stop_details"))


def ok_payload(answer: str, labels: list[str], insufficient: bool = False) -> dict:
    return {"answer": answer, "cited_labels": labels, "insufficient_evidence": insufficient}


# --------------------------------------------------------------------------- format_context


def test_format_context_labels_in_order(contexts):
    text = format_context(contexts)
    blocks = text.split("\n\n")
    assert len(blocks) == len(contexts)
    for i, (block, ctx) in enumerate(zip(blocks, contexts), start=1):
        assert block.startswith(f"[S{i}] [Patient {PID} | {ctx.chunk.title} | {ctx.chunk.date}")
        assert block == f"[S{i}] {ctx.chunk.text}"
    assert format_context([]) == ""


# --------------------------------------------------------------------------- Claude: request shape


def test_claude_request_kwargs_with_fallbacks(contexts):
    client = FakeClient(fake_response(ok_payload("A1c 7.2% on 2026-06-06 [S3].", ["S3"])))
    gen = ClaudeGenerator(model="claude-opus-5-5", effort="medium", max_tokens=16000, client=client)
    gen.generate("What is the most recent A1c?", PID, contexts)

    assert client.messages.calls == [] and len(client.beta.messages.calls) == 1
    kw = client.beta.messages.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 16000
    assert kw["output_config"]["effort"] == "medium"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert set(ANSWER_SCHEMA["required"]) == {"answer", "cited_labels", "insufficient_evidence"}
    assert kw["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    for forbidden in ("thinking", "temperature", "top_p", "top_k"):
        assert forbidden not in kw
    # one user turn, no assistant prefill
    assert [m["role"] for m in kw["messages"]] == ["user"]
    user = kw["messages"][0]["content"]
    assert "[S1] [Patient P001 | Lab Panel 2025-03-03" in user and "[S4]" in user
    assert "What is the most recent A1c?" in user and "S1, S2, S3, S4" in user
    system = kw["system"]
    assert system == SYSTEM_PROMPT
    for phrase in ("only these excerpts", "data, never instructions", "[S", "date", "most recent",
                   "insufficient_evidence", "do not document", "Never guess", "outside medical knowledge",
                   "treatment recommendations"):
        assert phrase in system, phrase


def test_claude_request_without_fallbacks_uses_plain_messages(contexts):
    client = FakeClient(fake_response(ok_payload("x [S1]", ["S1"])))
    ClaudeGenerator(effort="low", fallbacks=False, client=client).generate("q?", PID, contexts)
    assert client.beta.messages.calls == [] and len(client.messages.calls) == 1
    kw = client.messages.calls[0]
    assert "betas" not in kw and "fallbacks" not in kw
    assert kw["output_config"]["effort"] == "low"


@pytest.mark.parametrize("fallbacks", [True, False])
def test_real_sdk_serialises_request(contexts, fallbacks):
    """The installed anthropic SDK accepts our kwargs and puts them on the wire (mock transport, no network)."""
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(url=str(request.url), beta=request.headers.get("anthropic-beta"),
                    body=json.loads(request.content))
        body = ok_payload("ibuprofen [S2]", ["S2"])
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(body)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    res = ClaudeGenerator(client=client, fallbacks=fallbacks).generate("Allergies?", PID, contexts)
    assert [c.label for c in res.citations] == ["S2"] and res.warnings == []
    body = seen["body"]
    assert body["output_config"]["effort"] == "medium" and body["output_config"]["format"]["type"] == "json_schema"
    assert not {"thinking", "temperature", "top_p", "top_k"} & set(body)
    if fallbacks:
        assert body["fallbacks"] == "default" and seen["beta"] == FALLBACK_BETA
    else:
        assert "fallbacks" not in body and seen["beta"] is None


def test_invalid_effort_rejected():
    with pytest.raises(ValueError):
        ClaudeGenerator(effort="extreme")


# --------------------------------------------------------------------------- Claude: parsing


def test_claude_parses_answer_and_builds_citations(contexts):
    answer = "The most recent A1c is 7.2% on 2026-06-06 [S3]; earlier 8.1% on 2025-03-03 [S1]."
    client = FakeClient(fake_response(ok_payload(answer, ["S3", "S1"])))
    res = ClaudeGenerator(client=client).generate("Most recent A1c?", PID, contexts)

    assert isinstance(res, GenerationResult)
    assert res.mode == "llm" and res.model == "claude-opus-5-5"
    assert res.abstained is False and res.warnings == []
    assert res.text == answer
    assert [c.label for c in res.citations] == ["S3", "S1"]
    c3 = res.citations[0]
    assert (c3.chunk_id, c3.doc_id, c3.doc_type, c3.date) == (
        "P001-labs-2026-06-06::000", "P001-labs-2026-06-06", "lab_panel", "2026-06-06")
    assert not c3.snippet.startswith("[Patient") and c3.snippet.startswith("Lab results drawn 2026-06-06")


def test_claude_snippet_is_truncated_body(contexts):
    client = FakeClient(fake_response(ok_payload("Plan documented [S4].", ["S4"])))
    res = ClaudeGenerator(client=client).generate("plan?", PID, contexts)
    snip = res.citations[0].snippet
    assert snip.startswith("T2DM: A1c 7.6") and len(snip) <= 201 and snip.endswith("…")


def test_claude_drops_invalid_labels(contexts):
    answer = "A1c 7.2% [S3]. Also something [S9]. Allergy [S2, S7]."
    client = FakeClient(fake_response(ok_payload(answer, ["S3", "S9", "[s2]", "bogus"])))
    res = ClaudeGenerator(client=client).generate("q?", PID, contexts)
    assert [c.label for c in res.citations] == ["S3", "S2"]
    assert "[S9]" not in res.text and "S7" not in res.text
    assert "[S3]" in res.text and "[S2]" in res.text
    assert re.findall(r"\[S\d+\]", res.text) == ["[S3]", "[S2]"]
    joined = " ".join(res.warnings)
    assert "S9" in joined and "bogus" in joined and "S7" in joined


def test_claude_insufficient_evidence_abstains(contexts):
    payload = ok_payload("The provided record excerpts do not document an INR.", [], insufficient=True)
    res = ClaudeGenerator(client=FakeClient(fake_response(payload))).generate("Latest INR?", PID, contexts)
    assert res.abstained is True and res.citations == []
    assert "answer has no valid citations" not in res.warnings


def test_claude_uncited_answer_warns(contexts):
    res = ClaudeGenerator(client=FakeClient(fake_response(ok_payload("A1c is 7.2%.", [])))).generate(
        "q?", PID, contexts)
    assert res.abstained is False and "answer has no valid citations" in res.warnings


def test_claude_refusal_abstains_without_reading_content(contexts):
    resp = fake_response(None, stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))
    res = ClaudeGenerator(client=FakeClient(resp)).generate("q?", PID, contexts)
    assert res.abstained is True and res.citations == []
    assert any("refusal" in w and "bio" in w for w in res.warnings)


def test_claude_fallback_block_reported(contexts):
    resp = fake_response(ok_payload("x [S1]", ["S1"]))
    resp.content.insert(0, SimpleNamespace(type="fallback", from_=SimpleNamespace(model="claude-opus-5-5"),
                                           to=SimpleNamespace(model="claude-opus-5")))
    res = ClaudeGenerator(client=FakeClient(resp)).generate("q?", PID, contexts)
    assert any("fallback" in w and "claude-opus-5" in w for w in res.warnings)


def test_claude_max_tokens_warns(contexts):
    resp = fake_response(ok_payload("A1c 7.2% [S3].", ["S3"]), stop_reason="max_tokens")
    res = ClaudeGenerator(max_tokens=500, client=FakeClient(resp)).generate("q?", PID, contexts)
    assert any("max_tokens" in w for w in res.warnings)
    assert [c.label for c in res.citations] == ["S3"]


def test_claude_truncated_json_raises_generation_error(contexts):
    resp = fake_response('{"answer": "A1c 7.2% [S3', stop_reason="max_tokens")
    with pytest.raises(GenerationError) as ei:
        ClaudeGenerator(client=FakeClient(resp)).generate("q?", PID, contexts)
    assert any("max_tokens" in w for w in ei.value.warnings)


def test_non_synthetic_context_raises_before_client_called(contexts):
    bad = make_chunk("P001-N99", "clinical_note", "2025-01-01", "Progress Note", "text", synthetic="true")
    client = FakeClient(fake_response(ok_payload("x", [])))
    with pytest.raises(NonSyntheticDataError):
        ClaudeGenerator(client=client).generate("q?", PID, contexts + rc([bad]))
    assert client.calls == []


def test_foreign_patient_context_raises_before_client_called(contexts):
    other = make_chunk("P002-allergies", "allergy_list", "2026-06-01", "Allergy List", "- latex — rash",
                       pid="P002")
    client = FakeClient(fake_response(ok_payload("x", [])))
    with pytest.raises(ValueError, match="P002-allergies"):
        ClaudeGenerator(client=client).generate("q?", PID, contexts + rc([other]))
    assert client.calls == []


def test_no_contexts_abstains_without_call():
    client = FakeClient(fake_response(ok_payload("x", [])))
    res = ClaudeGenerator(client=client).generate("q?", PID, [])
    assert res.abstained and client.calls == []


@pytest.mark.parametrize("exc", [
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
    anthropic.RateLimitError(
        "rate limited",
        response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
        body=None,
    ),
])
def test_api_errors_wrapped_in_generation_error(contexts, exc):
    client = FakeClient(exc=exc)
    with pytest.raises(GenerationError) as ei:
        ClaudeGenerator(client=client).generate("q?", PID, contexts)
    assert isinstance(ei.value.__cause__, anthropic.AnthropicError)
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- extractive


def test_extractive_positive_with_valid_citations(contexts):
    res = ExtractiveGenerator().generate("What is the patient allergic to?", PID, contexts)
    assert res.abstained is False and res.mode == "extractive" and res.model is None
    assert "ibuprofen" in res.text and "[S2]" in res.text
    labels = {f"S{i}" for i in range(1, len(contexts) + 1)}
    by_label = {f"S{i}": c.chunk for i, c in enumerate(contexts, start=1)}
    assert res.citations and all(c.label in labels for c in res.citations)
    for c in res.citations:
        assert c.chunk_id == by_label[c.label].chunk_id
        assert f"[{c.label}]" in res.text
    assert set(re.findall(r"\[(S\d+)\]", res.text)) == {c.label for c in res.citations}
    assert len(res.text.splitlines()) <= 4  # intro line + at most 3 sentences


def test_extractive_abstains_when_key_term_absent(contexts):
    for q in ("Is the patient allergic to latex?", "What was the most recent INR?",
              "What dose of warfarin is the patient taking?"):
        res = ExtractiveGenerator().generate(q, PID, contexts)
        assert res.abstained is True, q
        assert "does not appear to document" in res.text
        assert res.citations == []


def test_extractive_abstains_on_generic_question(contexts):
    res = ExtractiveGenerator().generate("What is the most recent result?", PID, contexts)
    assert res.abstained is True


def test_extractive_temporal_prefers_latest_context(contexts):
    res = ExtractiveGenerator().generate("What was the most recent HbA1c?", PID, contexts)
    assert res.abstained is False
    first = res.text.splitlines()[0]
    assert "2026-06-06" in first and "7.2" in first and "[S3]" in first
    assert res.citations[0].label == "S3"


def test_extractive_non_temporal_uses_relevance_order(contexts):
    res = ExtractiveGenerator().generate("HbA1c values?", PID, contexts)
    # both lab lines match; S1 is ranked first by the retriever and is kept first
    assert res.citations[0].label == "S1"


def test_extractive_allergy_reaction_question_is_answered(contexts):
    # "reaction" names the kind of fact asked for; allergy records never contain the word itself.
    res = ExtractiveGenerator().generate("What reaction does the patient have to ibuprofen?", PID, contexts)
    assert res.abstained is False and "bronchospasm" in res.text
    neg = ExtractiveGenerator().generate("What reaction does the patient have to latex?", PID, contexts)
    assert neg.abstained is True


def test_extractive_temporal_prefers_value_line_over_panel_title_line():
    # The latest panel is split in two chunks: the title line of chunk ::001 names the HbA1c panel, but the
    # value is in chunk ::000, which the retriever ranked lower.
    title = "Lab results drawn 2026-06-23 (outpatient encounter P001-E07); panels: BMP, HbA1c, Hepatic function panel."
    later = make_chunk("P001-labs-2026-06-23", "lab_panel", "2026-06-23", "Lab Panel 2026-06-23",
                       f"{title}\n- Aspartate aminotransferase (AST): 19 U/L (ref 10-40)")
    later.chunk_id = "P001-labs-2026-06-23::001"
    value = make_chunk("P001-labs-2026-06-23", "lab_panel", "2026-06-23", "Lab Panel 2026-06-23",
                       f"{title}\n- Glucose: 101 mg/dL (ref 70-99) [H]\n- Hemoglobin A1c (HbA1c): 6.7 % (ref 4.0-5.6) [H]")
    older = make_chunk("P001-labs-2025-03-07", "lab_panel", "2025-03-07", "Lab Panel 2025-03-07",
                       "Lab results drawn 2025-03-07 (outpatient encounter P001-E02); panels: HbA1c.\n"
                       "- Hemoglobin A1c (HbA1c): 6.9 % (ref 4.0-5.6) [H]")
    res = ExtractiveGenerator().generate("What is the most recent HbA1c?", PID, rc([older, later, value]))
    first = res.text.splitlines()[0]
    assert "2026-06-23" in first and "6.7" in first and "[S3]" in first


def test_extractive_onset_question_with_expansion():
    # "first diagnosed" asks for the onset; the problem list says "onset", notes say "dx".
    ctx = rc([make_chunk("P001-problems", "problem_list", "2026-06-06", "Problem List",
                         "Problem list as of 2026-06-06.\n"
                         "- Essential (primary) hypertension (HTN) — ICD-10 I10 — onset 2011-12-01 — active")])
    q = "When was the patient's high blood pressure first diagnosed?"
    assert ExtractiveGenerator().generate(q, PID, ctx).abstained is True
    res = ExtractiveGenerator(term_expansions={"high blood pressure": ["HTN"]}).generate(q, PID, ctx)
    assert res.abstained is False and "2011-12-01" in res.text


def test_extractive_note_lead_in_line_brings_its_list():
    body = ("Home medications:\n- metformin 1000 mg PO BID\n- lisinopril 20 mg PO daily\n\n"
            "Drawn today (2025-04-14):\nBMP: glucose 92, Cr 0.81.\nFollow up in clinic.")
    ctx = rc([make_chunk("P001-N03", "clinical_note", "2025-04-14", "Emergency Department Note", body,
                         section="MEDICATIONS")])
    res = ExtractiveGenerator().generate("Which home medications?", PID, ctx)
    assert "Home medications: metformin 1000 mg PO BID; lisinopril 20 mg PO daily" in res.text
    lab = ExtractiveGenerator().generate("What was drawn on 2025-04-14?", PID, ctx)
    assert "Drawn today (2025-04-14): BMP: glucose 92, Cr 0.81." in lab.text
    assert "Follow up" not in lab.text


def test_extractive_temporal_same_date_prefers_sentence_with_value():
    note = make_chunk("P001-N05", "clinical_note", "2026-06-15", "Nephrology Consult Note",
                      "Continue BP control; urine albumin-to-creatinine ratio with next labs.",
                      section="ASSESSMENT AND PLAN")
    panel = make_chunk("P001-labs-2026-06-15", "lab_panel", "2026-06-15", "Lab Panel 2026-06-15",
                       "Lab results drawn 2026-06-15 (outpatient encounter P001-E08); panels: BMP.\n"
                       "- Creatinine (Cr): 1.41 mg/dL (ref 0.74-1.35) [H]")
    res = ExtractiveGenerator().generate("What is the most recent creatinine result?", PID, rc([note, panel]))
    assert "1.41" in res.text.splitlines()[0]


def test_extractive_ignores_question_verb_say():
    ctx = rc([make_chunk("P001-N07", "clinical_note", "2026-01-10", "Progress Note",
                         "Next visit in 4 months with fasting labs.", section="ASSESSMENT AND PLAN")])
    res = ExtractiveGenerator().generate('Where does it say: "Next visit in 4 months with fasting labs."?', PID, ctx)
    assert res.abstained is False and "say" not in ExtractiveGenerator().key_terms("Where does it say")


def test_extractive_term_expansions():
    ctx = rc([make_chunk("P001-N02", "clinical_note", "2025-02-01", "Cardiology Consult Note",
                         "Hx: s/p MI in 2019, CAD on aspirin.", section="HISTORY OF PRESENT ILLNESS")])
    q = "Has the patient had a heart attack?"
    assert ExtractiveGenerator().generate(q, PID, ctx).abstained is True
    res = ExtractiveGenerator(term_expansions={"heart attack": ["MI"]}).generate(q, PID, ctx)
    assert res.abstained is False and "s/p MI" in res.text


def test_extractive_rejects_foreign_patient(contexts):
    other = make_chunk("P002-allergies", "allergy_list", "2026-06-01", "Allergy List", "- latex — rash",
                       pid="P002")
    with pytest.raises(ValueError):
        ExtractiveGenerator().generate("latex?", PID, contexts + rc([other]))


def test_extractive_on_generated_data_matches_ground_truth():
    """Positive control on real generator output: latest creatinine among the patient's lab panels."""
    ds = generate_dataset(n_patients=3, seed=7)
    pid = ds.patients[0]["patient_id"]
    chunks = [c for c in chunk_documents(ds.to_documents()) if c.patient_id == pid and c.doc_type == "lab_panel"]
    creat = sorted((r for r in ds.labs if r["patient_id"] == pid and r["test_key"] == "creatinine"),
                   key=lambda r: r["date"])
    assert creat, "generator should produce creatinine values"
    latest = creat[-1]
    # Put the latest panel last so the temporal rule, not retrieval rank, has to find it.
    chunks.sort(key=lambda c: c.date)
    res = ExtractiveGenerator().generate("What was the most recent creatinine?", pid, rc(chunks))
    first = res.text.splitlines()[0]
    assert latest["date"] in first
    assert f"{latest['value']:.2f}" in first


# --------------------------------------------------------------------------- factory


def test_has_claude_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert has_claude_credentials() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    assert has_claude_credentials() is False
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert has_claude_credentials() is True


def test_get_generator(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert isinstance(get_generator(load_settings(llm="auto")), ExtractiveGenerator)
    assert isinstance(get_generator(load_settings(llm="extractive")), ExtractiveGenerator)
    gen = get_generator(load_settings(llm="claude", claude_effort="high", claude_max_tokens=1234,
                                      claude_fallbacks=False))
    assert isinstance(gen, ClaudeGenerator)
    assert (gen.model, gen.effort, gen.max_tokens, gen.fallbacks) == ("claude-opus-5-5", "high", 1234, False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(get_generator(load_settings(llm="auto")), ClaudeGenerator)
    with pytest.raises(ValueError):
        get_generator(load_settings(llm="gpt"))


# --------------------------------------------------------------------------- live (network)


@pytest.mark.skipif(not has_claude_credentials(), reason="needs ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
def test_live_claude_answer_and_abstention(contexts):
    settings = load_settings()
    gen = ClaudeGenerator(model=settings.claude_model, effort="low", max_tokens=settings.claude_max_tokens,
                          fallbacks=settings.claude_fallbacks)
    res = gen.generate("What was the most recent HbA1c?", PID, contexts)
    assert res.mode == "llm"
    if not res.abstained:
        assert "7.2" in res.text and "S3" in {c.label for c in res.citations}
    neg = gen.generate("What was the most recent INR?", PID, contexts)
    assert neg.abstained is True
