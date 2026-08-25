"""End-to-end checks that /chat returns structured citations (#15).

tests/test_citations.py covers the parser in isolation. These tests cover the
wiring: that the block is stripped from what the user is shown, that the parsed
citations arrive on the response body, and -- the point of the whole issue --
that a well-cited answer scores higher confidence than the same answer with no
citations at all.

The safety pipeline is disabled per-test via the environment variable the
handler already reads, so no test here needs a Gemini key or network.
"""

import re
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)

START = "<<<CITATIONS>>>"
END = "<<<END_CITATIONS>>>"


def block(payload: str) -> str:
    return f"\n\n{START}\n{payload}\n{END}"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Disable the safety classifier and start from an empty session store."""
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    main.active_chats.clear()
    yield
    main.active_chats.clear()


def mock_model_returning(text: str):
    """Build a Gemini stand-in whose single turn returns exactly ``text``."""
    session = MagicMock()

    async def send_message_async(message, **kwargs):
        response = MagicMock()
        response.text = text
        response.candidates = [MagicMock(finish_reason="STOP")]
        response.prompt_feedback = None
        return response

    session.send_message_async = send_message_async
    session.history = []

    model = MagicMock()
    model.start_chat.return_value = session
    return model


def ask(monkeypatch, answer_text: str, chat_id: str, prompt: str = "What does Islam say about patience?"):
    monkeypatch.setattr(main, "get_model", lambda: mock_model_returning(answer_text))
    valid_chat_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chat_id))
    response = client.post("/chat", json={"prompt": prompt, "chat_id": valid_chat_id})
    assert response.status_code == 200, response.text
    return response.json()


class TestCitationsOnTheResponse:
    def test_a_quran_citation_is_returned_as_structured_data(self, monkeypatch):
        answer = "Allah counsels patience." + block('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]}')
        body = ask(monkeypatch, answer, "c-quran")

        assert len(body["citations"]) == 1
        citation = body["citations"][0]
        assert citation["type"] == "quran"
        assert citation["surah"] == 2
        assert citation["ayah_start"] == 153
        # The name is authored by the surah index, never by the model.
        assert citation["surah_name"] == "Al-Baqarah"

    def test_the_block_is_never_shown_to_the_user(self, monkeypatch):
        answer = "Allah counsels patience." + block('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]}')
        body = ask(monkeypatch, answer, "c-hidden")

        assert START not in body["response"]
        assert END not in body["response"]
        assert "ayah_start" not in body["response"]
        assert body["response"].startswith("Allah counsels patience.")

    def test_an_answer_with_no_block_returns_an_empty_list(self, monkeypatch):
        body = ask(monkeypatch, "Patience is a virtue.", "c-none")

        assert body["citations"] == []
        assert body["response"].startswith("Patience is a virtue.")

    def test_a_hadith_citation_is_graded_from_the_dataset(self, monkeypatch):
        answer = "Actions are by intentions." + block(
            '{"citations": [{"type": "hadith", "collection": "bukhari", "number": "1"}]}'
        )
        body = ask(monkeypatch, answer, "c-hadith")

        assert len(body["citations"]) == 1
        citation = body["citations"][0]
        assert citation["type"] == "hadith"
        # Whatever alias the model used, the response carries the canonical name.
        assert citation["collection"] == "Sahih al-Bukhari"
        assert citation["number"] == "1"


class TestFailureIsAlwaysSurvivable:
    """A citation block can never cost the user their answer."""

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            "{",
            '{"citations": "a string, not a list"}',
            '{"citations": [{"type": "quran", "surah": 999, "ayah_start": 1}]}',
            '{"citations": [{"type": "quran", "surah": 2, "ayah_start": 900}]}',
            '{"citations": [{"type": "hadith", "collection": "Book of Made Up", "number": "3"}]}',
            "",
        ],
    )
    def test_malformed_or_invalid_blocks_yield_no_citations_and_still_answer(self, monkeypatch, payload):
        answer = "The prose survives." + block(payload)
        body = ask(monkeypatch, answer, f"c-bad-{abs(hash(payload))}")

        assert body["citations"] == []
        assert body["response"].startswith("The prose survives.")
        assert START not in body["response"]

    def test_a_truncated_block_is_stripped_rather_than_leaked(self, monkeypatch):
        # What a response cut off by max_output_tokens looks like: no end marker.
        answer = "Cut off mid-thought.\n\n" + START + '\n{"citations": [{"type": "qur'
        body = ask(monkeypatch, answer, "c-truncated")

        assert body["citations"] == []
        assert START not in body["response"]
        assert "citations" not in body["response"]
        # The half-written JSON must be gone, not merely unparsed.
        assert '"type": "qur' not in body["response"]
        # Assert on the prefix, not the whole string: a moderate-confidence answer
        # has a caution note appended after this handler returns, and that note is
        # the confidence layer's business, not this test's.
        assert body["response"].startswith("Cut off mid-thought.")


class TestConfidenceSignal:
    """The headline behavioural win: citations move the confidence score."""

    PROSE = "Allah counsels patience, and the Prophet taught intention."

    def test_validated_citations_raise_confidence_above_the_unverified_ceiling(self, monkeypatch):
        cited = self.PROSE + block(
            '{"citations": ['
            '{"type": "quran", "surah": 2, "ayah_start": 153},'
            '{"type": "quran", "surah": 94, "ayah_start": 5, "ayah_end": 6}'
            "]}"
        )
        with_citations = ask(monkeypatch, cited, "c-conf-cited")
        without = ask(monkeypatch, self.PROSE, "c-conf-plain")

        scored = with_citations["confidence"]
        plain = without["confidence"]

        assert "citation_verification" in scored["signals"]
        assert scored["signals"]["citation_verification"] == 1.0
        # An uncited answer must not be penalised - the signal simply drops out.
        assert "citation_verification" not in plain["signals"]
        assert scored["score"] > plain["score"]

    def test_fabricated_citations_lower_the_score(self, monkeypatch):
        half_bogus = self.PROSE + block(
            '{"citations": ['
            '{"type": "quran", "surah": 2, "ayah_start": 153},'
            '{"type": "quran", "surah": 2, "ayah_start": 9999}'
            "]}"
        )
        good = self.PROSE + block('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]}')

        mixed = ask(monkeypatch, half_bogus, "c-conf-mixed")
        clean = ask(monkeypatch, good, "c-conf-clean")

        assert mixed["confidence"]["signals"]["citation_verification"] == 0.5
        assert clean["confidence"]["signals"]["citation_verification"] == 1.0
        assert mixed["confidence"]["score"] < clean["confidence"]["score"]
        # The fabricated one is dropped; only the real reference is returned.
        assert len(mixed["citations"]) == 1


class TestSystemPrompt:
    def test_the_model_is_actually_told_about_the_format(self):
        """A format the model is never shown is a format it will never emit."""
        from citations import CITATION_BLOCK_CONTEXT

        assert START in CITATION_BLOCK_CONTEXT
        assert END in CITATION_BLOCK_CONTEXT
        # The instruction has to survive being concatenated after other blocks.
        assert CITATION_BLOCK_CONTEXT.startswith("\n")

    def test_the_response_model_declares_the_field(self):
        assert "citations" in main.ChatResponse.model_fields


def test_no_citation_markers_survive_into_any_visible_field(monkeypatch):
    """Belt and braces: no marker may appear anywhere the client renders."""
    answer = "Visible prose." + block('{"citations": [{"type": "quran", "surah": 1, "ayah_start": 1}]}')
    body = ask(monkeypatch, answer, "c-leak")

    rendered = [body.get("response") or "", body.get("text") or ""]
    rendered += [m.get("content") or "" for m in body.get("history", [])]
    marker = re.compile(r"<<<\s*/?\s*(END_)?CITATIONS\s*>>>")
    for field in rendered:
        assert not marker.search(field), f"citation marker leaked into: {field[:120]}"
