"""Contract tests for advertised languages, routers, and fallback metadata (#879).

These tests are offline and deterministic. They fail when the whitelist, the
documentation tables, the README, the support matrix, or the mounted language
routers drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from citations import CITATION_BLOCK_END, CITATION_BLOCK_START
from language_direction import KNOWN_LANGUAGES, direction_for
from main import SUPPORTED_LANGUAGES, ChatResponse, app, normalize_language

ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)


# ---------------------------------------------------------------------------
# Documentation agreement
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _readme_codes() -> set[str]:
    text = _read(ROOT / "README.md")
    section = text.split("### Multilingual support", 1)[1].split("\n### ", 1)[0]
    return set(re.findall(r"^\|\s*`([a-z]{2})`\s*\|", section, flags=re.MULTILINE))


def _matrix_codes() -> set[str]:
    text = _read(ROOT / "docs" / "language-support-matrix.md")
    return set(re.findall(r"^\|\s*[^|\n]+\|\s*`([a-z]{2})`\s*\|", text, flags=re.MULTILINE))


def test_readme_table_matches_whitelist() -> None:
    assert _readme_codes() == set(SUPPORTED_LANGUAGES)


def test_direction_helper_knows_the_whitelist() -> None:
    assert set(KNOWN_LANGUAGES) == set(SUPPORTED_LANGUAGES)


def test_support_matrix_matches_whitelist() -> None:
    assert _matrix_codes() == set(SUPPORTED_LANGUAGES)


def test_language_module_docs_do_not_claim_missing_routers() -> None:
    text = _read(ROOT / "docs" / "languages.md")
    for code, path in (("ms", "/malay"), ("ur", "/urdu")):
        assert code in SUPPORTED_LANGUAGES
        assert path in text
    malay_row = next(line for line in text.splitlines() if line.startswith("| Malay"))
    urdu_row = next(line for line in text.splitlines() if line.startswith("| Urdu"))
    assert "No (no FastAPI router)" not in malay_row
    assert "never imported" not in urdu_row
    assert "Yes" in malay_row
    assert "Yes" in urdu_row


# ---------------------------------------------------------------------------
# Routers are mounted and reachable
# ---------------------------------------------------------------------------


def test_advertised_language_routers_are_mounted() -> None:
    paths = {route.path for route in app.routes}
    assert {
        "/malay/analyze",
        "/malay/normalize",
        "/malay/dialects",
        "/malay/terms",
        "/malay/terms/{term_id}",
    } <= paths
    assert {"/urdu/process", "/urdu/terms"} <= paths


def test_malay_router_endpoints() -> None:
    analyze = client.post("/malay/analyze", json={"text": "Bagaimana cara shalat dan puasa?"})
    assert analyze.status_code == 200
    body = analyze.json()
    assert body["detected_dialect"] == "indonesia"
    assert body["islamic_terms_identified"]

    normalize = client.post("/malay/normalize", json={"text": "Shalat dan UANG", "dialect": "malaysia"})
    assert normalize.status_code == 200
    assert normalize.json()["normalized_text"] == "solat dan wang"

    dialects = client.get("/malay/dialects")
    assert dialects.status_code == 200
    assert {d["id"] for d in dialects.json()} == {"malaysia", "indonesia"}

    terms = client.get("/malay/terms", params={"query": "solat"})
    assert terms.status_code == 200
    assert terms.json()
    term_id = terms.json()[0]["id"]
    assert client.get(f"/malay/terms/{term_id}").status_code == 200
    assert client.get("/malay/terms/does-not-exist").status_code == 404


def test_urdu_router_is_reachable() -> None:
    response = client.post("/urdu/process", json={"text": "Quran میں صبر کے متعلق کیا حکم ہے؟"})
    assert response.status_code == 200
    assert response.json()["normalized_text"]


# ---------------------------------------------------------------------------
# Unsupported and partial language codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ur-PK", "ur"),
        ("AR", "ar"),
        ("  en  ", "en"),
        ("ms-SG", "ms"),
        ("zz", None),
        (" Klingon", None),
        ("", None),
        (None, None),
    ],
)
def test_language_normalization_is_deterministic(raw: str | None, expected: str | None) -> None:
    assert normalize_language(raw) == expected


class _FakeModel:
    def start_chat(self, history=None) -> _FakeSession:
        return _FakeSession()


class _FakeSession:
    def __init__(self) -> None:
        self.history: list[object] = []

    async def send_message_async(self, message: str, **_kwargs: object) -> SimpleNamespace:
        answer = "Sabr is patience."
        self.history.append(SimpleNamespace(role="model", parts=[SimpleNamespace(text=answer)]))
        return SimpleNamespace(text=answer, candidates=[], prompt_feedback=None)


@pytest.fixture
def fake_chat(monkeypatch: pytest.MonkeyPatch):
    async def _empty(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "tafsir_retriever", _empty)
    monkeypatch.setattr(main, "zakat_retriever", _empty)
    monkeypatch.setattr(main, "purchase_retriever", _empty)

    async def _enqueue(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(main, "enqueue_for_review", _enqueue)


def test_unknown_language_falls_back_to_auto_detection_not_422(fake_chat: None) -> None:
    response = client.post("/chat", json={"prompt": "What is sabr?", "language": "zz-ZZ"})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] is None
    assert body["requested_language"] is None
    assert body["direction"] == "ltr"


def test_base_code_fallback_preserves_effective_language(fake_chat: None) -> None:
    response = client.post("/chat", json={"prompt": "صبر کیا ہے؟", "language": "ur-PK"})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "ur"
    assert body["requested_language"] == "ur"
    assert body["direction"] == "rtl"


# ---------------------------------------------------------------------------
# An effective-language override keeps the requested code visible
# ---------------------------------------------------------------------------


def test_effective_language_override_preserves_requested_language() -> None:
    # The Swahili heuristic (and provider auto-detection) can change the
    # effective language after a caller asked for another one. Both codes must
    # stay visible, and direction follows the effective language (#879).
    response = ChatResponse(
        response="Subira ni ufunguo wa faraja",
        chat_id="chat-1",
        language="sw",
        requested_language="en",
    )
    assert response.requested_language == "en"
    assert response.language == "sw"
    assert response.direction == "ltr"

    rtl_response = ChatResponse(
        response="صبر کامیابی کی کنجی ہے",
        chat_id="chat-2",
        language="ur",
        requested_language="en",
    )
    assert rtl_response.requested_language == "en"
    assert rtl_response.direction == "rtl"


# ---------------------------------------------------------------------------
# Per-language direction and citation rules
# ---------------------------------------------------------------------------

RTL_CASES = [
    ("ar", "Arabic", "الصبر مفتاح الفرج"),
    ("ur", "Urdu", "صبر کامیابی کی کنجی ہے"),
    ("fa", "Persian", "صبر کلید پیروزی است"),
    ("ms", "Jawi Malay", "صلاة dan puasa"),
]
LTR_CASES = [
    ("en", "English", "Patience is the key to relief"),
    ("sw", "Swahili", "Subira ni ufunguo wa faraja"),
    ("sw", "Swahili-English", "Subira ni ufunguo, and patience is a virtue"),
]


@pytest.mark.parametrize(("code", "name", "text"), RTL_CASES)
def test_arabic_script_languages_are_rtl(code: str, name: str, text: str) -> None:
    assert code in SUPPORTED_LANGUAGES, name
    assert direction_for(code, text) == "rtl"


@pytest.mark.parametrize(("code", "name", "text"), LTR_CASES)
def test_latin_script_languages_are_ltr(code: str, name: str, text: str) -> None:
    assert code in SUPPORTED_LANGUAGES, name
    assert direction_for(code, text) == "ltr"


@pytest.mark.parametrize(("code", "_name", "quote"), [(c, n, t) for c, n, t in RTL_CASES])
def test_quran_quote_keeps_arabic_script_and_surah_ayah_citation(code: str, _name: str, quote: str) -> None:
    block = (
        f"{CITATION_BLOCK_START}\n"
        + json.dumps({"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]})
        + f"\n{CITATION_BLOCK_END}"
    )
    prose, extraction = main.extract_citations(f"{quote}\n\n{block}")
    assert quote in prose
    assert extraction.citations[0].reference == "2:153"
    assert direction_for(code, prose) == "rtl"


def test_jawi_malay_is_rtl_only_when_the_text_is_arabic_script() -> None:
    assert direction_for("ms", "صلاة") == "rtl"
    assert direction_for("ms", "Solat") == "ltr"


# ---------------------------------------------------------------------------
# Release matrix and mixed-language input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(SUPPORTED_LANGUAGES))
def test_every_advertised_language_resolves_to_a_documented_behaviour(code: str) -> None:
    name = SUPPORTED_LANGUAGES[code]
    assert name
    assert normalize_language(code) == code
    assert normalize_language(f"{code}-ZZ") == code
    assert direction_for(code, "test") in {"ltr", "rtl"}


@pytest.mark.parametrize(
    ("prompt", "language"),
    [
        ("Subira ni ufunguo wa faraja, and patience is a virtue.", "sw"),
        ("صبر کامیابی کی کنجی ہے, and patience is a virtue.", "ur"),
    ],
)
def test_mixed_language_prompt_keeps_requested_language(fake_chat: None, prompt: str, language: str) -> None:
    response = client.post("/chat", json={"prompt": prompt, "language": language})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == language
    assert body["requested_language"] == language
    assert body["direction"] == ("rtl" if language == "ur" else "ltr")


# ---------------------------------------------------------------------------
# Documentation cross-reference
# ---------------------------------------------------------------------------


def test_direction_helper_is_shared_with_docs() -> None:
    text = _read(ROOT / "docs" / "languages.md")
    assert "language_direction.py" in text
    assert "ChatResponse.direction" in text
