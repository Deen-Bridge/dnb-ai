from calligraphy_recognition import (
    RecognitionRequest,
    classify_style,
    normalize_arabic,
    recognize_calligraphy,
    resolve_style,
)


def test_normalize_removes_diacritics_and_tatweel():
    assert normalize_arabic("بِسْمِ  اللّٰهِ") == "بسم الله"
    assert normalize_arabic("قـــل") == "قل"


def test_style_hint_aliases_resolve():
    assert resolve_style("Kufi") == "kufic"
    assert resolve_style("thulth") == "thuluth"
    assert resolve_style("rūqah") == "ruqah"
    assert resolve_style("unknown") is None


def test_hint_style_wins_over_fallback():
    style, confidence = classify_style(RecognitionRequest(transcript="قل هو الله أحد", style_hint="Diwani"))
    assert style == "diwani"
    assert confidence == 0.95


def test_full_recognition_reports_style_and_quran_reference():
    result = recognize_calligraphy(RecognitionRequest(transcript="قُلْ هُوَ اللَّهُ أَحَدٌ", style_hint="naskh"))
    assert result.transcription == "قل هو الله أحد"
    assert result.style == "naskh"
    assert result.quran_reference == "112:1"
    assert result.partial is False
    assert 0.0 <= result.character_confidence <= 1.0
    assert result.warnings == []


def test_damaged_manuscript_is_partial_with_warning():
    result = recognize_calligraphy(RecognitionRequest(transcript="قل هو", style_hint="kufic", damaged=True))
    assert result.partial is True
    assert result.character_confidence < 0.8
    assert any("Damaged" in warning for warning in result.warnings)


def test_short_transcript_uses_low_confidence_fallback_style():
    result = recognize_calligraphy(RecognitionRequest(transcript="قل"))
    assert result.style == "naskh"
    assert result.style_confidence == 0.6
    assert result.partial is True
    assert any("Short" in warning for warning in result.warnings)
