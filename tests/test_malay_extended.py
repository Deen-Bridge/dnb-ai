"""Extended coverage for the malay module (#445).

Complements tests/test_malay.py with unit tests for each helper class plus processing
edge cases. Everything is pure Python: no model, network or external API is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import malay
from malay.code_switching import CodeSwitchingHandler
from malay.cultural_context import CulturalContextManager
from malay.dialects import DialectHandler
from malay.generator import MalayContentGenerator
from malay.loanwords import LoanwordRecognizer
from malay.models import MalayQueryRequest, MalayQueryResponse
from malay.optimizer import QueryOptimizer
from malay.processor import MalayProcessor
from malay.router import MalayRouter
from malay.terminology import TerminologyLexicon

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMS_FILE = Path("data") / "malay_indonesian_islamic_terms.json"

JAWI_SOLAT = "\u0635\u0644\u0627\u0629"
JAWI_PUASA = "\u06a4\u0648\u0627\u0633"
JAWI_ALLAH = "\u0627\u0644\u0644\u0647"
JAWI_MUHAMMAD = "\u0645\u062d\u0645\u062f"
GREETING = "Assalamu'alaikum warahmatullahi wabarakatuh."


@pytest.fixture
def repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lexicon loads a relative data path, so run from the repo root."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def empty_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A working directory with no data/ folder, forcing the built-in fallback lexicon."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_terms_file(root: Path, content: str) -> None:
    target = root / TERMS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_request_defaults() -> None:
    request = MalayQueryRequest(text="solat")

    assert request.dialect is None
    assert request.target_script == "latin"
    assert request.context == {}


def test_request_requires_text() -> None:
    with pytest.raises(ValidationError):
        MalayQueryRequest.model_validate({})


def test_request_context_default_is_not_shared_between_instances() -> None:
    first = MalayQueryRequest(text="a")
    second = MalayQueryRequest(text="b")

    assert first.context is not None
    first.context["key"] = "value"

    assert second.context == {}


def test_response_defaults() -> None:
    response = MalayQueryResponse(
        original_text="a",
        normalized_text="a",
        detected_dialect="malaysia",
        detected_script="latin",
        optimized_response="b",
    )

    assert response.transliterated_text is None
    assert response.islamic_terms_identified == []
    assert response.confidence == 1.0


# ---------------------------------------------------------------------------
# QueryOptimizer and CodeSwitchingHandler
# ---------------------------------------------------------------------------


def test_optimizer_strips_and_lowercases() -> None:
    assert QueryOptimizer.optimize_query("  Apa Itu ZAKAT? \n") == "apa itu zakat?"


def test_optimizer_handles_empty_and_whitespace_only_input() -> None:
    assert QueryOptimizer.optimize_query("") == ""
    assert QueryOptimizer.optimize_query("   \t ") == ""


def test_code_switching_translates_known_arabic_phrases() -> None:
    result = CodeSwitchingHandler.process_code_switching("Allahu Akbar dan Subhanallah")

    assert result == "Allah Maha Besar dan Maha Suci Allah"


def test_code_switching_leaves_other_text_untouched() -> None:
    assert CodeSwitchingHandler.process_code_switching("selamat pagi") == "selamat pagi"


def test_code_switching_is_case_sensitive() -> None:
    assert CodeSwitchingHandler.process_code_switching("allahu akbar") == "allahu akbar"


# ---------------------------------------------------------------------------
# DialectHandler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bagaimana cara shalat?", "indonesia"),
        ("uang saya hilang", "indonesia"),
        ("Bagaimana cara solat?", "malaysia"),
        ("berapa wang zakat", "malaysia"),
        ("shalat dan solat", "indonesia"),  # an Indonesian marker wins when both appear
        ("awak kamu", "malaysia"),  # no hard markers, so the score decides
        ("bagaimana", "indonesia"),
        ("macam mana", "malaysia"),
        ("rumah dengan tidak", "indonesia"),  # tied score defaults to Indonesian
        ("", "indonesia"),
    ],
)
def test_detect_dialect(text: str, expected: str) -> None:
    assert DialectHandler.detect_dialect(text) == expected


def test_normalize_dialect_to_malaysia() -> None:
    assert DialectHandler.normalize_dialect("shalat dan uang", "malaysia") == "solat dan wang"


def test_normalize_dialect_to_indonesia() -> None:
    assert DialectHandler.normalize_dialect("solat dan wang", "indonesia") == "shalat dan uang"


def test_normalize_dialect_is_case_insensitive_and_replaces_with_lowercase() -> None:
    assert DialectHandler.normalize_dialect("Shalat UANG", "malaysia") == "solat wang"


def test_normalize_dialect_preserves_unrelated_words_and_collapses_whitespace() -> None:
    assert DialectHandler.normalize_dialect("Rumah   Saya", "malaysia") == "Rumah Saya"


def test_normalize_dialect_ignores_unknown_target_dialect() -> None:
    assert DialectHandler.normalize_dialect("shalat uang", "brunei") == "shalat uang"


def test_normalize_dialect_does_not_touch_words_with_attached_punctuation() -> None:
    # Documents current behaviour: matching is on the raw token, so "shalat," is not rewritten.
    assert DialectHandler.normalize_dialect("shalat,", "malaysia") == "shalat,"


def test_normalize_dialect_empty_text() -> None:
    assert DialectHandler.normalize_dialect("", "malaysia") == ""


# ---------------------------------------------------------------------------
# TerminologyLexicon
# ---------------------------------------------------------------------------

CUSTOM_TERMS = {
    "terms": [
        {"id": "t-solat", "malay_term": "Solat", "indonesian_term": "Shalat", "variants": ["sembahyang"]},
        {"id": "t-puasa", "malay_term": "Puasa", "indonesian_term": "Puasa", "variants": ["shaum"]},
    ]
}


def test_lexicon_uses_builtin_fallback_when_data_file_is_missing(empty_cwd: Path) -> None:
    lexicon = TerminologyLexicon()

    assert [t["id"] for t in lexicon.terms] == ["ms-id-ibada-solat"]


@pytest.mark.parametrize(
    "content",
    ["{not valid json", '{"terms": []}', '{"other": 1}', "[1, 2, 3]"],
)
def test_lexicon_falls_back_on_unusable_data_file(empty_cwd: Path, content: str) -> None:
    _write_terms_file(empty_cwd, content)

    lexicon = TerminologyLexicon()

    assert [t["id"] for t in lexicon.terms] == ["ms-id-ibada-solat"]


def test_lexicon_loads_terms_from_data_file(empty_cwd: Path) -> None:
    _write_terms_file(empty_cwd, json.dumps(CUSTOM_TERMS))

    lexicon = TerminologyLexicon()

    assert [t["id"] for t in lexicon.terms] == ["t-solat", "t-puasa"]


def test_repository_data_file_is_valid_and_loads(repo_cwd: None) -> None:
    lexicon = TerminologyLexicon()

    assert lexicon.terms
    assert all("id" in term and "malay_term" in term for term in lexicon.terms)


@pytest.mark.parametrize("word", ["Solat", "SHALAT", "sembahyang"])
def test_lookup_matches_malay_indonesian_and_variant_forms_case_insensitively(empty_cwd: Path, word: str) -> None:
    _write_terms_file(empty_cwd, json.dumps(CUSTOM_TERMS))

    found = TerminologyLexicon().lookup(word)

    assert found is not None
    assert found["id"] == "t-solat"


def test_lookup_returns_none_for_unknown_word(empty_cwd: Path) -> None:
    _write_terms_file(empty_cwd, json.dumps(CUSTOM_TERMS))

    assert TerminologyLexicon().lookup("kucing") is None


def test_identify_terms_strips_punctuation_and_deduplicates(empty_cwd: Path) -> None:
    _write_terms_file(empty_cwd, json.dumps(CUSTOM_TERMS))

    found = TerminologyLexicon().identify_terms("Solat, puasa dan solat!")

    assert [t["id"] for t in found] == ["t-solat", "t-puasa"]


def test_identify_terms_returns_empty_list_when_nothing_matches(empty_cwd: Path) -> None:
    lexicon = TerminologyLexicon()

    assert lexicon.identify_terms("selamat pagi semua") == []
    assert lexicon.identify_terms("") == []


# ---------------------------------------------------------------------------
# LoanwordRecognizer, MalayRouter, MalayContentGenerator, CulturalContextManager
# ---------------------------------------------------------------------------


def test_loanwords_recognizes_known_words_ignoring_case_and_punctuation() -> None:
    assert LoanwordRecognizer.recognize_loanwords("Ilmu dan Sabar, ZAKAT!") == ["ilmu", "sabar", "zakat"]


@pytest.mark.parametrize(("word", "expected"), [("al-Quran", "al-quran"), ("sunnah", "sunnah"), ("ibadat", "ibadat")])
def test_loanwords_recognizes_arabic_prefix_and_suffix_patterns(word: str, expected: str) -> None:
    assert LoanwordRecognizer.recognize_loanwords(word) == [expected]


def test_loanwords_deduplicates_repeated_words() -> None:
    assert LoanwordRecognizer.recognize_loanwords("sabar sabar Sabar") == ["sabar"]


def test_loanwords_ignores_ordinary_malay_words_and_empty_input() -> None:
    assert LoanwordRecognizer.recognize_loanwords("saya makan nasi") == []
    assert LoanwordRecognizer.recognize_loanwords("") == []


@pytest.mark.parametrize("text", ["Bagaimana cara berpuasa?", "Apa itu ZAKAT", "rukun haji", "hukum shalat jumaat"])
def test_router_sends_worship_questions_to_fiqh_ibadat(text: str) -> None:
    assert MalayRouter.route_request(text) == "fiqh_ibadat"


@pytest.mark.parametrize("text", ["Siapa nabi terakhir?", "selamat pagi", ""])
def test_router_defaults_to_general_islamic_qa(text: str) -> None:
    assert MalayRouter.route_request(text) == "general_islamic_qa"


def test_generator_produces_indonesian_content() -> None:
    expected = "Penjelasan mengenai zakat dalam konteks ajaran Islam yang rahmatan lil 'alamin di Indonesia."

    assert MalayContentGenerator.generate_islamic_content("zakat", "indonesia") == expected


@pytest.mark.parametrize("dialect", ["malaysia", "brunei", ""])
def test_generator_uses_malaysian_content_for_everything_else(dialect: str) -> None:
    expected = "Penerangan mengenai zakat menurut panduan Ahli Sunnah Wal Jamaah di Malaysia."

    assert MalayContentGenerator.generate_islamic_content("zakat", dialect) == expected


def test_cultural_context_prefixes_indonesian_responses() -> None:
    expected = f"{GREETING} Berdasarkan pemahaman Islam di Nusantara (Indonesia): isi"

    assert CulturalContextManager.enrich_response("isi", "indonesia") == expected


@pytest.mark.parametrize("dialect", ["malaysia", "singapore"])
def test_cultural_context_prefixes_other_responses_as_malaysian(dialect: str) -> None:
    expected = f"{GREETING} Berdasarkan panduan syariah di rantau Melayu (Malaysia): isi"

    assert CulturalContextManager.enrich_response("isi", dialect) == expected


# ---------------------------------------------------------------------------
# MalayProcessor
# ---------------------------------------------------------------------------


@pytest.fixture
def processor(empty_cwd: Path) -> MalayProcessor:
    """A processor backed by the built-in fallback lexicon, independent of the data file."""
    return MalayProcessor()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("solat puasa", "latin"),
        ("Bagaimana cara shalat?", "latin"),
        (JAWI_SOLAT, "jawi"),
        (f"hukum {JAWI_PUASA} ramadan", "jawi"),
        ("", "latin"),
    ],
)
def test_detect_script(processor: MalayProcessor, text: str, expected: str) -> None:
    assert processor.detect_script(text) == expected


def test_transliterate_maps_known_words_and_drops_surrounding_punctuation(processor: MalayProcessor) -> None:
    result = processor.transliterate_to_jawi("Solat, PUASA dan Allah!")

    assert result == f"{JAWI_SOLAT} {JAWI_PUASA} dan {JAWI_ALLAH}"


def test_transliterate_maps_indonesian_spelling_and_muhammad(processor: MalayProcessor) -> None:
    assert processor.transliterate_to_jawi("shalat Muhammad") == f"{JAWI_SOLAT} {JAWI_MUHAMMAD}"


def test_transliterate_leaves_unknown_words_unchanged(processor: MalayProcessor) -> None:
    assert processor.transliterate_to_jawi("Saya Makan") == "Saya Makan"


def test_transliterate_empty_text(processor: MalayProcessor) -> None:
    assert processor.transliterate_to_jawi("") == ""


def test_process_detects_dialect_and_enriches_response(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="Bagaimana cara shalat?"))

    assert response.original_text == "Bagaimana cara shalat?"
    assert response.normalized_text == "bagaimana cara shalat?"
    assert response.detected_dialect == "indonesia"
    assert response.detected_script == "latin"
    assert response.transliterated_text is None
    assert [t["id"] for t in response.islamic_terms_identified] == ["ms-id-ibada-solat"]
    assert response.optimized_response.startswith(f"{GREETING} Berdasarkan pemahaman Islam di Nusantara")
    assert "bagaimana cara shalat?" in response.optimized_response
    assert response.confidence == 0.98


def test_explicit_dialect_overrides_detection_and_normalizes_spelling(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="Solat", dialect="indonesia"))

    assert response.detected_dialect == "indonesia"
    assert response.normalized_text == "shalat"


def test_malaysian_dialect_normalizes_indonesian_spelling(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="Shalat dan UANG", dialect="malaysia"))

    assert response.normalized_text == "solat dan wang"
    assert response.optimized_response.startswith(f"{GREETING} Berdasarkan panduan syariah di rantau Melayu")


def test_blank_dialect_falls_back_to_detection(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="berapa wang zakat", dialect=""))

    assert response.detected_dialect == "malaysia"


def test_unknown_dialect_is_reported_and_uses_malaysian_framing(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="solat", dialect="brunei"))

    assert response.detected_dialect == "brunei"
    assert response.normalized_text == "solat"
    assert "(Malaysia)" in response.optimized_response


def test_jawi_input_is_detected_and_transliterated(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text=JAWI_SOLAT))

    assert response.detected_script == "jawi"
    assert response.normalized_text == JAWI_SOLAT
    assert response.transliterated_text is not None
    assert JAWI_SOLAT in response.transliterated_text


def test_repeated_term_spellings_are_reported_once(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="solat shalat salat"))

    assert len(response.islamic_terms_identified) == 1


def test_text_without_known_terms_yields_no_terms(processor: MalayProcessor) -> None:
    response = processor.process(MalayQueryRequest(text="Selamat pagi semua"))

    assert response.islamic_terms_identified == []


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_text_is_processed_without_error(processor: MalayProcessor, text: str) -> None:
    response = processor.process(MalayQueryRequest(text=text))

    assert response.original_text == text
    assert response.normalized_text == ""
    assert response.islamic_terms_identified == []
    assert response.transliterated_text is None


def test_package_exports_public_api() -> None:
    assert set(malay.__all__) == {"MalayQueryRequest", "MalayQueryResponse", "MalayProcessor"}
    assert malay.MalayProcessor is MalayProcessor
    assert malay.MalayQueryRequest is MalayQueryRequest
    assert malay.MalayQueryResponse is MalayQueryResponse
