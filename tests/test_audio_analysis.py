"""Tests for the voice-note transcription / Islamic audio-analysis module.

Offline only: no network, no live ASR, no API keys.
"""

import struct

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio_analysis import (
    Segment,
    StaticTranscriber,
    TemplateResponder,
    analyze_transcript,
    assess_noise,
    detect_audio_format,
    detect_recitation,
    estimate_speakers,
    extract_questions,
    identify_language,
    recognize_terms,
    router,
    validate_audio_upload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_bytes(size_seconds: float = 1.0, channels: int = 1, sample_rate: int = 16_000) -> bytes:
    """Build a minimal (structurally valid enough) RIFF/WAVE blob."""
    data_len = int(size_seconds * sample_rate) * channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16)
    data = b"data" + struct.pack("<I", data_len) + b"\x00" * data_len
    return header + fmt + data


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_wav_by_magic_bytes():
    assert detect_audio_format(_wav_bytes()) == "wav"


def test_detect_ogg_mp3_and_m4a():
    assert detect_audio_format(b"OggS" + b"\x00" * 32) == "ogg"
    assert detect_audio_format(b"ID3\x04\x00\x00\x00" + b"\x00" * 32) == "mp3"
    assert detect_audio_format(b"\xff\xfb\x90\x00" + b"\x00" * 16) == "mp3"
    assert detect_audio_format(b"\x00\x00\x00\x18ftypM4A \x00" * 8) == "m4a"


def test_detect_unknown_returns_none():
    assert detect_audio_format(b"definitely not audio") is None


def test_detect_falls_back_to_extension():
    assert detect_audio_format(b"garbage-here", filename="clip.mp3") == "mp3"


def test_validate_upload_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        validate_audio_upload(b"")
    with pytest.raises(ValueError, match="not recognised"):
        validate_audio_upload(b"plain text payload")


def test_validate_upload_rejects_oversize():
    big = b"RIFF" + b"WAVE" + b"\x00" * (26 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        validate_audio_upload(big)


def test_validate_upload_accepts_valid_wav():
    assert validate_audio_upload(_wav_bytes(), "clip.wav") == "wav"


# ---------------------------------------------------------------------------
# Noise assessment
# ---------------------------------------------------------------------------


def test_assess_noise_silent_wav_recommends_denoising():
    assessment = assess_noise(_wav_bytes(), "wav")
    assert assessment.recommended_profile == "noise_suppression"
    assert assessment.denoise_recommended is True


def test_assess_noise_unknown_format_is_informational():
    assessment = assess_noise(b"OggS" + b"\x00" * 128, "ogg")
    assert assessment.estimator == "unavailable"
    assert assessment.denoise_recommended is False


# ---------------------------------------------------------------------------
# Language & dialect identification
# ---------------------------------------------------------------------------


def test_identify_language_english():
    result = identify_language("What is the ruling on fasting during Ramadan?")
    assert result.primary == "en"
    assert result.script == "latin"


def test_identify_language_arabic():
    result = identify_language("ما حكم صيام رمضان؟")
    assert result.primary == "ar"
    assert result.script == "arabic"


def test_identify_language_egyptian_dialect():
    result = identify_language("إزيك يا شيخ، أنت فين النهاردة؟")
    assert result.primary == "ar"
    assert result.dialect == "egyptian"


def test_identify_language_empty_is_unknown():
    result = identify_language("   ")
    assert result.primary == "unknown"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Recitation detection
# ---------------------------------------------------------------------------


def test_detect_recitation_basmala_fires():
    result = detect_recitation("بسم الله الرحمن الرحيم، قل هو الله أحد")
    assert result.is_recitation is True
    assert result.ratio > 0.0
    assert result.reasons


def test_detect_recitation_plain_text_does_not_fire():
    result = detect_recitation("The meeting starts at nine tomorrow morning.")
    assert result.is_recitation is False


# ---------------------------------------------------------------------------
# Question extraction
# ---------------------------------------------------------------------------


def test_extract_questions_english():
    questions = extract_questions("How does one perform wudu? Tell me the steps please.")
    assert len(questions) >= 1
    assert questions[0].language == "en"


def test_extract_questions_arabic():
    questions = extract_questions("ما هي شروط صحة الصلاة؟ أرجو التوضيح")
    assert len(questions) >= 1
    assert questions[0].language == "ar"


# ---------------------------------------------------------------------------
# Terminology recognition
# ---------------------------------------------------------------------------


def test_recognize_terms_english_transliterations():
    terms = recognize_terms("The core of tawheed is the shahada and salah.")
    labels = {match.term for match in terms}
    assert "توحيد" in labels
    assert "شهادة" in labels
    assert "صلاة" in labels


def test_recognize_terms_arabic():
    terms = recognize_terms("التوحيد والشهادة أمران عظيمان")
    assert any(match.count >= 1 for match in terms)


def test_recognize_terms_noisy_text_is_empty():
    assert recognize_terms("The quarterly report beat expectations by 12%.") == []


# ---------------------------------------------------------------------------
# Speaker estimation
# ---------------------------------------------------------------------------


def test_estimate_speakers_single_flow():
    estimate = estimate_speakers("hello", [])
    assert estimate.estimated_speakers == 1
    assert estimate.method == "single"


def test_estimate_speakers_language_switch_confirmed():
    segments = [
        Segment(start=0.0, end=3.0, text="salam", language="ar"),
        Segment(start=3.2, end=6.0, text="what is the ruling", language="en"),
    ]
    estimate = estimate_speakers("", segments)
    assert estimate.estimated_speakers >= 2


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------


def test_analyze_transcript_integrates_dimensions():
    transcript = (
        "Assalamu alaykum. What is the ruling on fasting during Ramadan? "
        "The core of tawheed is the shahada. May Allah reward you abundantly."
    )
    analysis = analyze_transcript(transcript)
    assert analysis.transcript == transcript
    assert analysis.language.primary == "en"
    assert len(analysis.questions) >= 1
    assert len(analysis.terminology) >= 1
    assert analysis.speakers.estimated_speakers >= 1
    assert analysis.emotion is not None
    assert not analysis.warnings


def test_analyze_transcript_empty_adds_warning():
    analysis = analyze_transcript("   ")
    assert any("empty" in warning for warning in analysis.warnings)


def test_analyze_transcript_language_hint_overrides():
    analysis = analyze_transcript("In the name of God the most merciful", language_hint="ar")
    assert analysis.language.primary == "ar"


# ---------------------------------------------------------------------------
# Transcriber backends (offline)
# ---------------------------------------------------------------------------


def test_static_transcriber_returns_canned_output():
    import asyncio

    out = asyncio.run(
        StaticTranscriber("peace be upon you", segments=[Segment(start=0.0, end=1.0, text="peace")]).transcribe(
            b"irrelevant", "wav"
        )
    )
    assert out.text == "peace be upon you"
    assert out.transcriber == "static"
    assert len(out.segments) == 1


def test_template_responder_generates_offline_answer():
    import asyncio

    from audio_analysis import GenerateRequest

    response = asyncio.run(
        TemplateResponder().generate(GenerateRequest(transcript="What about zakat?", question="Explain zakat"))
    )
    assert response.generator == "template"
    assert "zakat" in response.response


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_formats_endpoint_lists_supported_types():
    body = _client().get("/audio/formats").json()
    assert "mp3" in body["formats"]
    assert body["mime_types"]["wav"] == "audio/wav"


def test_terminology_endpoint_returns_glossary():
    body = _client().get("/audio/terminology").json()
    assert body["count"] >= 50
    assert any(entry["term"] == "توحيد" for entry in body["entries"])


def test_transcribe_endpoint_with_mocked_transcriber(monkeypatch):
    monkeypatch.setattr(
        "audio_analysis.get_transcriber", lambda: StaticTranscriber("Assalamu alaykum, how do I pray Fajr?")
    )
    resp = _client().post("/audio/transcribe", files={"file": ("note.wav", _wav_bytes(), "audio/wav")})
    assert resp.status_code == 200
    assert resp.json()["text"].startswith("Assalamu")


def test_transcribe_endpoint_rejects_bad_upload():
    resp = _client().post("/audio/transcribe", files={"file": ("note.xyz", b"not audio", "application/octet-stream")})
    assert resp.status_code == 415


def test_generate_endpoint_uses_template_responder(monkeypatch):
    monkeypatch.setattr("audio_analysis.get_responder", lambda: TemplateResponder())
    body = _client().post("/audio/generate", json={"transcript": "What is the ruling on zakat?"}).json()
    assert body["generator"] == "template"
    assert body["response"]


def test_generate_endpoint_validates_empty_transcript(monkeypatch):
    monkeypatch.setattr("audio_analysis.get_responder", lambda: TemplateResponder())
    resp = _client().post("/audio/generate", json={"transcript": ""})
    assert resp.status_code == 422
