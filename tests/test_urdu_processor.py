"""Offline tests for Urdu normalization, terminology, and mixed-script processing."""

from fastapi.testclient import TestClient
from fastapi import FastAPI

from urdu_processor import (
    analyze_script,
    extract_islamic_terms,
    normalize_urdu,
    process_urdu,
    router,
    tokenize_urdu,
    transliterate_urdu,
)


def test_normalizes_arabic_keyboard_variants_and_nastaliq_spacing() -> None:
    assert normalize_urdu("  زكاة\u200c  كي  حكم؟ ") == "زکاۃ کی حکم؟"


def test_diacritics_are_preserved_by_default_and_optionally_removed() -> None:
    text = "قُرْآن"
    assert "ُ" in normalize_urdu(text)
    assert normalize_urdu(text, preserve_diacritics=False) == "قرآن"


def test_multiword_islamic_terms_are_single_tokens() -> None:
    tokens = tokenize_urdu("فقہ حنفی میں زکوٰۃ کا حکم")
    assert tokens[0].text == "فقہ حنفی"
    assert tokens[0].kind == "islamic_term"
    assert tokens[0].term_id == "ur-fiqh-hanafi"


def test_recognizes_urdu_arabic_and_latin_variants_without_duplicates() -> None:
    terms = extract_islamic_terms("وضو اور wudu کے بعد نماز salah")
    ids = [term.id for term in terms]
    assert ids.count("ur-ibadat-wudu") == 1
    assert ids.count("ur-ibadat-salah") == 1


def test_mixed_script_profile() -> None:
    profile = analyze_script("زکوٰۃ nisab 2.5%")
    assert profile.mixed_script is True
    assert profile.dominant_script == "mixed"
    assert profile.urdu_arabic_characters > 0
    assert profile.latin_characters > 0


def test_curated_transliteration_is_preferred_for_terms() -> None:
    result = transliterate_urdu("زکوٰۃ اور نماز")
    assert "zakat" in result
    assert "namaz / salah" in result


def test_full_analysis_supplies_urdu_generation_and_citation_guidance() -> None:
    result = process_urdu("Quran میں صبر کے متعلق کیا حکم ہے؟")
    assert result.normalized_text
    assert result.script_profile.mixed_script is True
    assert any(term.id == "ur-quran-quran" for term in result.recognized_terms)
    assert "قرآنی آیات" in result.generation_guidance
    assert "حوالہ" in result.generation_guidance


def test_router_process_and_term_search() -> None:
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    processed = client.post("/urdu/process", json={"text": "نماز اور وضو"})
    assert processed.status_code == 200
    assert len(processed.json()["recognized_terms"]) == 2

    searched = client.get("/urdu/terms", params={"query": "tawhid"})
    assert searched.status_code == 200
    assert searched.json()[0]["id"] == "ur-belief-tawhid"
