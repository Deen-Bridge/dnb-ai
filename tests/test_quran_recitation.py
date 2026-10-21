"""Tests for Quranic recitation analysis — no network, no live ASR."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quran_recitation import (
    AnalyzeRequest,
    PassthroughTranscriber,
    Transcriber,
    analyze_recitation,
    find_reciter,
    normalize_arabic,
    router,
    tokenize,
)

IKHLAS = "قل هو الله أحد الله الصمد لم يلد ولم يولد ولم يكن له كفوا أحد"
FATIHAH = "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك نستعين اهدنا الصراط المستقيم صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين"


def _analyze(transcript: str | None = None, **kwargs):
    return analyze_recitation(AnalyzeRequest(transcript=transcript, **kwargs))


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_folds_diacritics_and_alef_variants():
    assert normalize_arabic("قُلْ هُوَ ٱللَّهُ أَحَدٌ") == normalize_arabic("قل هو الله احد")


def test_tokenize_splits_words():
    assert tokenize("والعصر، إن الإنسان") == ["والعصر", "ان", "الانسان"]


# ---------------------------------------------------------------------------
# Identification and verification
# ---------------------------------------------------------------------------


def test_identifies_full_surah_as_accurate():
    result = _analyze(IKHLAS)
    assert result.identified
    assert result.surah.number == 112
    assert result.surah.reference == "112:1-4"
    assert result.verification == "accurate"
    assert result.accuracy == 1.0
    assert result.partial is False
    assert result.issues == []
    assert len(result.words) == 15


def test_identifies_fatihah():
    result = _analyze(FATIHAH)
    assert result.surah.number == 1
    assert result.surah.ayah_end == 7
    assert result.verification == "accurate"


def test_partial_recitation_reports_ayah_range():
    result = _analyze("الله الصمد لم يلد ولم يولد")
    assert result.surah.number == 112
    assert (result.surah.ayah_start, result.surah.ayah_end) == (2, 3)
    assert result.partial is True
    assert [a.ayah for a in result.ayat] == [2, 3]
    assert all(a.complete for a in result.ayat)


def test_mixed_content_is_separated_from_recitation():
    result = _analyze("بسم الله الرحمن الرحيم " + IKHLAS + " صدق الله العظيم")
    assert result.surah.number == 112
    assert result.basmalah_detected is True
    assert result.leading_content == []
    assert result.trailing_content == ["صدق", "الله", "العظيم"]
    assert result.verification == "accurate"


def test_omitted_word_is_reported():
    result = _analyze(IKHLAS.replace(" الصمد", ""))
    assert result.verification == "minor_deviations"
    assert [(i.type, i.expected) for i in result.issues] == [("omission", "الصمد")]
    assert any(w.status == "missing" and w.expected == "الصمد" for w in result.words)


def test_substituted_word_is_reported():
    result = _analyze(IKHLAS.replace("الصمد", "الصمت"))
    issue = result.issues[0]
    assert (issue.type, issue.expected, issue.heard) == ("substitution", "الصمد", "الصمت")


def test_inserted_word_is_reported():
    result = _analyze(IKHLAS.replace("الصمد", "الصمد يا"))
    assert [(i.type, i.heard) for i in result.issues] == [("insertion", "يا")]


def test_unrelated_text_is_unidentified():
    result = _analyze("hello world this is not the quran")
    assert result.identified is False
    assert result.verification == "unidentified"
    assert result.surah is None


def test_empty_text_is_unidentified():
    assert _analyze("!!!").identified is False


# ---------------------------------------------------------------------------
# Audio quality
# ---------------------------------------------------------------------------


def test_noisy_audio_loosens_word_matching():
    noisy_text = IKHLAS.replace("الصمد", "الصمت")
    clean = _analyze(noisy_text, asr_confidence=0.95)
    noisy = _analyze(noisy_text, asr_confidence=0.3)
    assert next(w for w in clean.words if w.expected == "الصمد").status == "missing"
    assert next(w for w in noisy.words if w.expected == "الصمد").status == "approximate"


def test_close_asr_variant_is_approximate_not_exact():
    result = _analyze(IKHLAS.replace("الصمد", "الصمدد"))
    word = next(w for w in result.words if w.expected == "الصمد")
    assert word.status == "approximate"
    assert result.accuracy < 1.0


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_provided_word_timings_are_attached():
    tokens = IKHLAS.split()
    timings = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.5} for i, w in enumerate(tokens)]
    result = _analyze(word_timings=timings)
    assert result.timing_source == "provided"
    assert (result.words[0].start, result.words[0].end) == (0.0, 0.5)
    assert (result.ayat[0].start, result.ayat[0].end) == (0.0, 2.0)


def test_mismatched_timings_are_ignored():
    result = _analyze(IKHLAS, word_timings=[{"word": "قل", "start": 0, "end": 1}])
    assert result.timing_source == "none"
    assert result.words[0].start is None


def test_timings_estimated_from_duration():
    result = _analyze(IKHLAS, duration_seconds=10)
    assert result.timing_source == "estimated"
    assert result.words[0].start == 0.0
    assert abs(result.words[-1].end - 10) < 0.01
    assert "estimated" in result.note


# ---------------------------------------------------------------------------
# Riwayah, reciter, maqam
# ---------------------------------------------------------------------------


def test_long_mim_maliki_is_hafs_compatible():
    result = _analyze(FATIHAH)
    assert result.riwayah.name == "Hafs 'an 'Asim"
    assert result.riwayah.basis == "text_variant"


def test_short_maliki_is_flagged_as_non_hafs_and_not_an_error():
    result = _analyze(FATIHAH.replace("مالك", "ملك"))
    assert result.riwayah.basis == "text_variant"
    assert result.riwayah.name == "Warsh 'an Nafi'"
    assert result.verification == "accurate"
    assert result.issues == []


def test_riwayah_hint_conflict_is_noted():
    result = _analyze(FATIHAH, riwayah_hint="warsh")
    assert result.riwayah.name == "Hafs 'an 'Asim"
    assert "conflicts" in result.riwayah.note


def test_riwayah_hint_used_when_passage_has_no_variant():
    result = _analyze(IKHLAS, riwayah_hint="Warsh")
    assert (result.riwayah.name, result.riwayah.basis) == ("Warsh 'an Nafi'", "hint")


def test_riwayah_defaults_to_hafs():
    result = _analyze(IKHLAS)
    assert (result.riwayah.name, result.riwayah.basis) == ("Hafs 'an 'Asim", "default")


def test_reciter_and_maqam_from_hints():
    result = _analyze(IKHLAS, reciter_hint="Alafasy", maqam_hint="bayati")
    assert result.reciter.name == "Mishary Rashid Alafasy"
    assert result.reciter.basis == "hint"
    assert result.maqam == "Bayati"


def test_reciter_from_metadata_and_no_guessing():
    assert _analyze(IKHLAS, metadata_text="Husary_Ikhlas.mp3").reciter.basis == "metadata"
    unknown = _analyze(IKHLAS, maqam_hint="not-a-maqam")
    assert unknown.reciter is None
    assert unknown.maqam is None
    assert find_reciter(None, "") is None


# ---------------------------------------------------------------------------
# Transcriber interface and endpoints
# ---------------------------------------------------------------------------


def test_passthrough_transcriber():
    assert isinstance(PassthroughTranscriber(), Transcriber)
    assert PassthroughTranscriber().transcribe("قل هو الله أحد") == "قل هو الله أحد"


def test_analyze_endpoint():
    response = _client().post("/quran-recitation/analyze", json={"transcript": IKHLAS})
    assert response.status_code == 200
    body = response.json()
    assert body["surah"]["name"] == "Al-Ikhlas"
    assert body["verification"] == "accurate"


def test_analyze_endpoint_requires_input():
    client = _client()
    assert client.post("/quran-recitation/analyze", json={}).status_code == 422
    assert client.post("/quran-recitation/analyze", json={"transcript": "   "}).status_code == 422


def test_surahs_endpoint():
    body = _client().get("/quran-recitation/surahs").json()
    assert {s["number"] for s in body["surahs"]} >= {1, 112}
    assert "Hafs 'an 'Asim" in body["riwayat"]
    assert "Bayati" in body["maqamat"]
