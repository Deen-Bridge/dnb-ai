"""Tests for hierarchical intent classification (#203).

Each acceptance criterion from the issue gets at least one test: detailed
Islamic categories, response-format identification, fiqh urgency, learning vs
action orientation, comparative/historical/meta detection, multi-intent,
answer depth, academic vs practical register, and accuracy tracking.
"""

from intent import (
    IntentAccuracyTracker,
    classify_intent,
    detect_domains,
    detect_response_format,
    pick_primary_domain,
)

# ---------------------------------------------------------------------------
# Knowledge-domain classification (hierarchical taxonomy)
# ---------------------------------------------------------------------------


class TestDomainClassification:
    def test_fiqh_ibadat(self):
        result = classify_intent("How do I perform wudu correctly?")
        assert "fiqh_ibadat" in result.domains

    def test_fiqh_muamalat(self):
        result = classify_intent("Is taking a mortgage with interest permissible?")
        assert "fiqh_muamalat" in result.domains

    def test_contemporary_issues(self):
        result = classify_intent("Is trading cryptocurrency permissible in Islam?")
        assert "contemporary_issues" in result.domains

    def test_fiqh_munakahat_mirath(self):
        result = classify_intent("How is inheritance divided between children?")
        assert "fiqh_munakahat_mirath" in result.domains

    def test_aqeedah(self):
        result = classify_intent("What does tawhid mean and why is shirk so grave?")
        assert "aqeedah" in result.domains
        assert result.primary_domain == "aqeedah"

    def test_ulum_al_quran(self):
        result = classify_intent("What is the tafsir of the verse on charity?")
        assert "ulum_al_quran" in result.domains

    def test_mustalah_al_hadith(self):
        result = classify_intent("What makes a hadith sahih — is it the isnad or the matn?")
        assert "mustalah_al_hadith" in result.domains

    def test_seerah(self):
        result = classify_intent("Tell me about the seerah of Prophet Muhammad")
        assert "seerah" in result.domains

    def test_tarikh_islami(self):
        result = classify_intent("Who was the caliph during the battle of Yarmuk?")
        assert "tarikh_islami" in result.domains

    def test_tasawwuf_adab_akhlaq(self):
        result = classify_intent("How do I develop patience and good akhlaq?")
        assert "tasawwuf_adab_akhlaq" in result.domains

    def test_multi_label(self):
        # A question can touch several domains at once; all should be reported.
        domains = detect_domains("what is the ruling on riba and how does tawhid relate to it")
        assert "fiqh_muamalat" in domains
        assert "aqeedah" in domains
        assert len(domains) >= 2

    def test_no_domain_cues_gives_empty_domains(self):
        result = classify_intent("What is the weather like today?")
        assert result.domains == []
        assert result.primary_domain is None

    def test_primary_domain_prefers_specific_over_general(self):
        # "islam" alone would match general_islam; the specific domain wins.
        assert pick_primary_domain(["general_islam", "fiqh_ibadat"]) == "fiqh_ibadat"
        assert pick_primary_domain(["general_islam"]) == "general_islam"


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------


class TestResponseFormat:
    def test_fatwa_ruling_request(self):
        assert detect_response_format("is it permissible to combine prayers?") == "fatwa"

    def test_tafsir_explanation_request(self):
        assert detect_response_format("explain the meaning of Ayat al-Kursi") == "tafsir_explanation"

    def test_comparison_request(self):
        assert detect_response_format("difference between zakat and sadaqah") == "comparison"

    def test_practical_guidance_request(self):
        assert detect_response_format("how do I pray maghrib step by step") == "practical_guidance"

    def test_factual_lookup_request(self):
        assert detect_response_format("who was Umar ibn al-Khattab?") == "factual_lookup"

    def test_comparison_outranks_fatwa(self):
        # Comparing rulings wants analysis, not a single verdict.
        fmt = detect_response_format("compare the rulings on combining intentions across schools")
        assert fmt == "comparison"

    def test_no_format_for_chitchat(self):
        assert detect_response_format("hello there") is None


# ---------------------------------------------------------------------------
# Fiqh urgency
# ---------------------------------------------------------------------------


class TestFiqhUrgency:
    def test_urgent_personal_ruling_is_high(self):
        result = classify_intent("URGENT: my prayer time ends soon, is my wudu broken?")
        assert result.fiqh_urgency == "high"

    def test_non_urgent_fiqh_is_none(self):
        result = classify_intent("What is the ruling on combining prayers while travelling?")
        assert result.fiqh_urgency == "none"

    def test_urgency_without_fiqh_context_stays_none(self):
        result = classify_intent("I urgently need a recipe for dinner tonight")
        assert result.fiqh_urgency == "none"


# ---------------------------------------------------------------------------
# Learning vs action orientation
# ---------------------------------------------------------------------------


class TestOrientation:
    def test_learning_orientation(self):
        result = classify_intent("I want to understand and learn about the pillars of iman")
        assert result.orientation == "learning"

    def test_action_orientation(self):
        result = classify_intent("Can I pray in a church building?")
        assert result.orientation == "action"

    def test_neither_orientation(self):
        result = classify_intent("Zakat rates for agricultural produce are well documented.")
        assert result.orientation is None


# ---------------------------------------------------------------------------
# Comparative / historical / meta flags
# ---------------------------------------------------------------------------


class TestFacetFlags:
    def test_comparative(self):
        result = classify_intent("Compare the Hanafi and Shafi'i positions on witr")
        assert result.is_comparative is True

    def test_not_comparative(self):
        result = classify_intent("What is the position on witr prayer?")
        assert result.is_comparative is False

    def test_historical(self):
        result = classify_intent("When was the Quran compiled into a single mushaf?")
        assert result.is_historical is True

    def test_meta_question_about_methodology(self):
        result = classify_intent("How do you determine which school of thought your answers follow?")
        assert result.is_meta is True

    def test_ordinary_question_is_not_meta(self):
        result = classify_intent("What is the ruling on shortening prayers?")
        assert result.is_meta is False


# ---------------------------------------------------------------------------
# Multi-intent detection
# ---------------------------------------------------------------------------


class TestMultiIntent:
    def test_two_distinct_asks(self):
        result = classify_intent("What is riba and how do I perform wudu properly?")
        assert result.is_multi_intent is True

    def test_single_ask_spanning_two_domains_is_not_multi(self):
        result = classify_intent("What is the ruling on riba in cryptocurrency transactions?")
        assert len(result.domains) >= 1
        assert result.is_multi_intent is False

    def test_single_domain_single_segment(self):
        result = classify_intent("How do I fast in Ramadan?")
        assert result.is_multi_intent is False


# ---------------------------------------------------------------------------
# Answer depth
# ---------------------------------------------------------------------------


class TestAnswerDepth:
    def test_brief_requested(self):
        result = classify_intent("Quick question — briefly, is coffee halal?")
        assert result.answer_depth == "brief"

    def test_detailed_requested(self):
        result = classify_intent("Explain the Islamic philosophy of law in depth with examples")
        assert result.answer_depth == "detailed"

    def test_standard_default(self):
        result = classify_intent("What is the ruling on gifting?")

        assert result.answer_depth == "standard"


# ---------------------------------------------------------------------------
# Academic vs practical register
# ---------------------------------------------------------------------------


class TestAcademicRegister:
    def test_academic(self):
        result = classify_intent("I am writing a research paper on usul al-fiqh; cite sources please")
        assert result.academic_register == "academic"

    def test_practical(self):
        result = classify_intent("Should I break my fast if I feel dizzy?")
        assert result.academic_register == "practical"

    def test_unmarked_register(self):
        result = classify_intent("The four Sunni schools of jurisprudence emerged gradually.")
        assert result.academic_register is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_empty_prompt(self):
        result = classify_intent("")
        assert result.primary_domain is None
        assert result.response_format is None

    def test_none_like_prompt(self):
        result = classify_intent("   ")
        assert result.domains == []

    def test_word_boundaries_prevent_substring_false_positives(self):
        # 'iman' must not fire inside 'imagine'; 'asr' must not fire inside
        # 'considerable'.
        result = classify_intent("Imagine a considerable sunset view")
        assert "fiqh_ibadat" not in result.domains
        assert "aqeedah" not in result.domains


# ---------------------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------------------


class TestIntentAccuracyTracker:
    def test_record_prediction_counts_by_domain(self):
        tracker = IntentAccuracyTracker()
        tracker.record_prediction(classify_intent("What is tawhid?"))
        tracker.record_prediction(classify_intent("Hello world"))
        snapshot = tracker.snapshot()
        assert snapshot["total_predictions"] == 2
        by_domain = snapshot["predictions_by_domain"]
        assert by_domain.get("aqeedah") == 1
        assert by_domain.get("unclassified") == 1

    def test_record_labelled_examples_and_accuracy(self):
        tracker = IntentAccuracyTracker()
        tracker.record("response_format", predicted="fatwa", actual="fatwa")
        tracker.record("response_format", predicted="fatwa", actual="comparison")
        accuracy = tracker.snapshot()["accuracy"]["response_format"]
        assert accuracy["total"] == 2.0
        assert accuracy["correct"] == 1.0
        assert accuracy["accuracy"] == 0.5

    def test_unknown_field_rejected(self):
        import pytest

        tracker = IntentAccuracyTracker()
        with pytest.raises(ValueError):
            tracker.record("not_a_field", "x", "y")

    def test_reset_clears_everything(self):
        tracker = IntentAccuracyTracker()
        tracker.record_prediction(classify_intent("What is tawhid?"))
        tracker.record("primary_domain", "aqeedah", "aqeedah")
        tracker.reset()
        snapshot = tracker.snapshot()
        assert snapshot["total_predictions"] == 0
        assert snapshot["accuracy"] == {}
