"""Tests for the deterministic religious sentiment analyzer.

All offline — imports only the ``sentiment`` module, no model calls, no
GEMINI_API_KEY, no FastAPI app boot.
"""

import sentiment
from sentiment import (
    CRISIS_THRESHOLD,
    EMOTION_TAXONOMY,
    AnalyzeRequest,
    CareIndicators,
    SentimentAnalysis,
    adapt_tone,
    analyze,
    score_terms,
)

NEUTRAL_INFO = "What are the five pillars of Islam and how many rakats are in Fajr prayer?"
DISTRESS = "I feel so hopeless and depressed lately, I'm overwhelmed and I don't know what to do."
CRISIS = "I don't want to live anymore and I keep thinking about ending my life. Please help me."
SPIRITUAL = "I am losing my faith and I feel like Allah abandoned me."
DOUBT = "I have serious doubts about whether Islam is true, how do I know for sure?"
NEW_MUSLIM = "I just became muslim and took my shahada last week, how do I pray Fajr?"
COMFORT = "Please comfort me and make dua for me, I feel so alone and heartbroken."


# ---------------------------------------------------------------------------
# Distress detection
# ---------------------------------------------------------------------------


class TestDistressDetection:
    def test_distress_fires_on_distress_phrasing(self) -> None:
        analysis = analyze(DISTRESS)
        assert analysis.flags.emotional_distress is True
        assert analysis.dimensions.emotional > 0.0

    def test_distress_does_not_fire_on_neutral_info(self) -> None:
        analysis = analyze(NEUTRAL_INFO)
        assert analysis.flags.emotional_distress is False
        assert analysis.dimensions.emotional == 0.0
        assert analysis.flags.information_seeking is True

    def test_negation_suppresses_distress(self) -> None:
        analysis = analyze("I do not feel hopeless, I'm just curious what breaks wudu.")
        assert analysis.flags.emotional_distress is False


# ---------------------------------------------------------------------------
# Crisis threshold → pastoral care / referral
# ---------------------------------------------------------------------------


class TestCrisisReferral:
    def test_crisis_triggers_care_indicators(self) -> None:
        analysis = analyze(CRISIS)
        care = analysis.care
        assert isinstance(care, CareIndicators)
        assert care.crisis_detected is True
        assert care.referral_recommended is True
        assert care.severity == "high"
        assert care.triggers  # the offending phrases are surfaced
        assert care.message is not None

    def test_neutral_info_has_no_care_signal(self) -> None:
        care = analyze(NEUTRAL_INFO).care
        assert care.crisis_detected is False
        assert care.referral_recommended is False
        assert care.severity == "none"
        assert care.message is None

    def test_crisis_score_crosses_threshold(self) -> None:
        # The crisis dimension must exceed the documented threshold for a hit.
        raw, hits = score_terms("i want to die", sentiment.CRISIS_TERMS)
        assert hits
        assert sentiment._saturate(raw) >= CRISIS_THRESHOLD


# ---------------------------------------------------------------------------
# Doubt / spiritual crisis
# ---------------------------------------------------------------------------


class TestDoubtAndSpiritual:
    def test_doubt_detection(self) -> None:
        analysis = analyze(DOUBT)
        assert analysis.flags.faith_doubt is True
        assert analysis.dimensions.spiritual > 0.0

    def test_spiritual_crisis_detection(self) -> None:
        analysis = analyze(SPIRITUAL)
        assert analysis.flags.spiritual_crisis is True
        assert analysis.care.referral_recommended is True

    def test_neutral_has_no_doubt(self) -> None:
        analysis = analyze(NEUTRAL_INFO)
        assert analysis.flags.faith_doubt is False
        assert analysis.flags.spiritual_crisis is False


# ---------------------------------------------------------------------------
# Comfort vs information classification
# ---------------------------------------------------------------------------


class TestComfortVsInformation:
    def test_comfort_seeking_classified_as_comfort(self) -> None:
        analysis = analyze(COMFORT)
        assert analysis.flags.comfort_seeking is True
        assert analysis.primary_intent == "comfort"

    def test_information_seeking_classified_as_information(self) -> None:
        analysis = analyze(NEUTRAL_INFO)
        assert analysis.flags.information_seeking is True
        assert analysis.primary_intent == "information"


# ---------------------------------------------------------------------------
# New-Muslim detection
# ---------------------------------------------------------------------------


class TestNewMuslim:
    def test_new_muslim_detected(self) -> None:
        analysis = analyze(NEW_MUSLIM)
        assert analysis.flags.new_muslim is True

    def test_regular_question_not_new_muslim(self) -> None:
        assert analyze(NEUTRAL_INFO).flags.new_muslim is False


# ---------------------------------------------------------------------------
# Tone adaptation
# ---------------------------------------------------------------------------


class TestToneAdaptation:
    def test_distress_gets_supportive_tone(self) -> None:
        analysis = analyze(DISTRESS)
        tone = adapt_tone(analysis)
        assert tone.tone == "empathetic_supportive"
        assert tone.response_prefix is not None
        assert tone.preserve_information_note  # accuracy reminder always present

    def test_crisis_gets_referral_tone(self) -> None:
        assert analyze(CRISIS).tone.tone == "compassionate_urgent_referral"

    def test_neutral_gets_informative_tone(self) -> None:
        tone = analyze(NEUTRAL_INFO).tone
        assert tone.tone == "clear_informative"
        assert tone.response_prefix is None

    def test_new_muslim_gets_welcoming_tone(self) -> None:
        assert analyze(NEW_MUSLIM).tone.tone == "welcoming_foundational"

    def test_recommended_tone_matches_tone_block(self) -> None:
        analysis = analyze(SPIRITUAL)
        assert analysis.recommended_tone == analysis.tone.tone


# ---------------------------------------------------------------------------
# Taxonomy shape and request model
# ---------------------------------------------------------------------------


class TestTaxonomyAndModels:
    def test_taxonomy_shape(self) -> None:
        assert isinstance(EMOTION_TAXONOMY, dict)
        assert "crisis" in EMOTION_TAXONOMY
        assert "emotional_distress" in EMOTION_TAXONOMY
        for key, entry in EMOTION_TAXONOMY.items():
            assert "dimension" in entry, key
            assert "description" in entry, key
            assert entry["dimension"] in {"emotional", "spiritual", "informational"}

    def test_analysis_is_serializable(self) -> None:
        analysis = analyze(DISTRESS)
        assert isinstance(analysis, SentimentAnalysis)
        dumped = analysis.model_dump()
        assert set(dumped["dimensions"]) == {"emotional", "spiritual", "informational"}

    def test_request_model_validates_nonempty(self) -> None:
        req = AnalyzeRequest(text="Assalamu alaikum")
        assert req.text == "Assalamu alaikum"

    def test_determinism(self) -> None:
        assert analyze(DISTRESS).model_dump() == analyze(DISTRESS).model_dump()
