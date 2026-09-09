"""Tests for uncertainty quantification, Islamic epistemology taxonomy, and evidence strength scoring (#199).

All tests are offline — no real model calls or network dependencies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from citations import QuranCitation
from confidence import ConfidenceAssessment, assess, build_signals
from hadith import HadithReference
from main import app
from uncertainty import (
    EpistemicCertainty,
    EvidenceStrength,
    PositionType,
    calculate_bayesian_confidence,
    classify_epistemic_certainty,
    classify_position_type,
    detect_high_stakes_and_consultation,
    evaluate_evidence_strength,
    quantify_uncertainty,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Epistemic Classification Tests
# ---------------------------------------------------------------------------


class TestEpistemicClassification:
    def test_qati_core_obligation_classified_correctly(self):
        prompt = "Is fasting the month of Ramadan obligatory?"
        answer = "Yes, fasting Ramadan is one of the five pillars of Islam and an absolute obligation with ijma."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.QATI

    def test_qati_clear_prohibition(self):
        prompt = "Is consuming interest (riba) permissible?"
        answer = "No, the prohibition of riba is absolute and established by explicit Quranic text."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.QATI

    def test_disputed_matter_detected_by_keywords(self):
        prompt = "Does bleeding break wudu?"
        answer = (
            "The Hanafi school holds that flowing blood invalidates wudu, while the Shafi'i school holds it does not."
        )
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.DISPUTED

    def test_disputed_matter_detected_by_ikhtilaf_markers(self):
        prompt = "What is the ruling on raising hands before ruku?"
        answer = "Scholars differ on this point: the majority recommend it based on hadith, while the Hanafi school prefers not raising."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.DISPUTED

    def test_novel_contemporary_nawazil_detected(self):
        prompt = "What is the Islamic ruling on trading Bitcoin and cryptocurrencies?"
        answer = "Contemporary scholars have differing views on cryptocurrency based on whether it qualifies as mal and currency."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.NOVEL_CONTEMPORARY

    def test_dhanni_standard_branch_fiqh(self):
        prompt = "What is the proper method for performing tayammum?"
        answer = "Tayammum is performed by striking clean earth and wiping the face and hands."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        assert epistemic is EpistemicCertainty.DHANNI

    def test_non_religious_query_classified_as_general(self):
        prompt = "What is the capital of Turkey?"
        answer = "The capital of Turkey is Ankara."
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=False, is_religious=False)
        assert epistemic is EpistemicCertainty.GENERAL


# ---------------------------------------------------------------------------
# Position Type Classification Tests
# ---------------------------------------------------------------------------


class TestPositionTypeClassification:
    def test_ijma_for_qati_matters(self):
        pos = classify_position_type(
            "What are the five pillars?",
            "The five pillars are established by consensus",
            EpistemicCertainty.QATI,
            is_fiqh=True,
        )
        assert pos is PositionType.IJMA

    def test_scholarly_ikhtilaf_for_disputed_matters(self):
        pos = classify_position_type(
            "Does touching a woman break wudu?",
            "Scholars differ between schools",
            EpistemicCertainty.DISPUTED,
            is_fiqh=True,
        )
        assert pos is PositionType.SCHOLARLY_IKHTILAF

    def test_contemporary_ijtihad_for_nawazil(self):
        pos = classify_position_type(
            "Is artificial intelligence generated art halal?",
            "Modern juristic councils deliberated",
            EpistemicCertainty.NOVEL_CONTEMPORARY,
            is_fiqh=True,
        )
        assert pos is PositionType.CONTEMPORARY_IJTIHAD

    def test_established_madhhab_for_standard_fiqh(self):
        pos = classify_position_type(
            "How to perform wudu in the Hanafi school?",
            "The Hanafi school outlines four obligatory acts",
            EpistemicCertainty.DHANNI,
            is_fiqh=True,
        )
        assert pos is PositionType.ESTABLISHED_MADHHAB


# ---------------------------------------------------------------------------
# Evidence Strength Evaluation Tests
# ---------------------------------------------------------------------------


class TestEvidenceStrengthEvaluation:
    def test_very_strong_evidence_with_quran_and_sahih_hadith(self):
        quran_cite = QuranCitation(surah=2, ayah_start=183, surah_name="Al-Baqarah")
        hadith_cite = HadithReference(
            raw="Sahih al-Bukhari 8", collection="bukhari", hadith_number=8, grade="SAHIH", verified=True, flagged=False
        )
        strength = evaluate_evidence_strength(
            citations=[quran_cite], hadith_refs=[hadith_cite], is_religious=True, citation_score=1.0
        )
        assert strength is EvidenceStrength.VERY_STRONG

    def test_strong_evidence_with_sahih_hadith(self):
        hadith_cite = HadithReference(
            raw="Sahih Muslim 1", collection="muslim", hadith_number=1, grade="SAHIH", verified=True, flagged=False
        )
        strength = evaluate_evidence_strength(
            citations=[], hadith_refs=[hadith_cite], is_religious=True, citation_score=1.0
        )
        assert strength is EvidenceStrength.STRONG

    def test_moderate_evidence_with_hasan_hadith(self):
        hadith_cite = HadithReference(
            raw="Jami at-Tirmidhi 2415",
            collection="tirmidhi",
            hadith_number=2415,
            grade="HASAN",
            verified=True,
            flagged=False,
        )
        strength = evaluate_evidence_strength(
            citations=[], hadith_refs=[hadith_cite], is_religious=True, citation_score=1.0
        )
        assert strength is EvidenceStrength.MODERATE

    def test_weak_evidence_when_daif_hadith_cited(self):
        hadith_cite = HadithReference(
            raw="Sunan Ibn Majah 123",
            collection="ibnmajah",
            hadith_number=123,
            grade="DAIF",
            verified=True,
            flagged=True,
        )
        strength = evaluate_evidence_strength(
            citations=[], hadith_refs=[hadith_cite], is_religious=True, citation_score=0.5
        )
        assert strength is EvidenceStrength.WEAK_OR_LIMITED

    def test_weak_evidence_when_no_citations_for_religious_answer(self):
        strength = evaluate_evidence_strength(citations=[], hadith_refs=[], is_religious=True)
        assert strength is EvidenceStrength.WEAK_OR_LIMITED


# ---------------------------------------------------------------------------
# High-Stakes & Expert Consultation Tests
# ---------------------------------------------------------------------------


class TestHighStakesAndConsultation:
    @pytest.mark.parametrize(
        ("prompt", "answer"),
        [
            ("I uttered triple talaq to my wife in anger, is the divorce valid?", "Divorce rulings depend on phrasing"),
            (
                "How should the inheritance shares of my late father be divided?",
                "Inheritance requires full family tree",
            ),
            (
                "Is organ donation after brain death permissible for my relative?",
                "Bioethics councils hold varying views",
            ),
            ("Can a couple use IVF and surrogacy?", "Assisted reproduction has strict juristic constraints"),
        ],
    )
    def test_sensitive_fatwa_triggers_expert_consultation(self, prompt: str, answer: str):
        epistemic = classify_epistemic_certainty(prompt, answer, is_fiqh=True, is_religious=True)
        is_high_stakes, reason = detect_high_stakes_and_consultation(prompt, answer, epistemic)
        assert is_high_stakes is True
        assert reason is not None
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# Bayesian Confidence & Range Calculation Tests
# ---------------------------------------------------------------------------


class TestBayesianConfidenceCalculation:
    def test_qati_matter_has_high_confidence_and_tight_interval(self):
        score, uncertainty, interval = calculate_bayesian_confidence(
            epistemic=EpistemicCertainty.QATI,
            evidence_strength=EvidenceStrength.VERY_STRONG,
            citation_score=1.0,
            expressed_certainty_score=1.0,
            consistency_score=0.95,
            is_high_stakes=False,
            total_citations_count=3,
        )
        assert score >= 0.85
        assert uncertainty <= 0.15
        assert interval[0] <= score <= interval[1]
        assert (interval[1] - interval[0]) <= 0.20  # Tight uncertainty range

    def test_disputed_matter_has_lower_score_and_wider_interval(self):
        score, uncertainty, interval = calculate_bayesian_confidence(
            epistemic=EpistemicCertainty.DISPUTED,
            evidence_strength=EvidenceStrength.MODERATE,
            citation_score=0.7,
            expressed_certainty_score=0.8,
            consistency_score=0.6,
            is_high_stakes=False,
            total_citations_count=1,
        )
        assert score < 0.70
        assert uncertainty > 0.30
        assert interval[0] <= score <= interval[1]
        assert (interval[1] - interval[0]) >= 0.25  # Wider uncertainty range

    def test_high_stakes_penalty_reduces_score(self):
        normal_score, _, _ = calculate_bayesian_confidence(
            epistemic=EpistemicCertainty.DHANNI,
            evidence_strength=EvidenceStrength.STRONG,
            citation_score=1.0,
            expressed_certainty_score=1.0,
            consistency_score=None,
            is_high_stakes=False,
            total_citations_count=2,
        )
        stakes_score, _, _ = calculate_bayesian_confidence(
            epistemic=EpistemicCertainty.DHANNI,
            evidence_strength=EvidenceStrength.STRONG,
            citation_score=1.0,
            expressed_certainty_score=1.0,
            consistency_score=None,
            is_high_stakes=True,
            total_citations_count=2,
        )
        assert stakes_score < normal_score


# ---------------------------------------------------------------------------
# Complete Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestQuantifyUncertaintyPipeline:
    def test_qati_pipeline_result(self):
        q = quantify_uncertainty(
            prompt="Is Salah obligatory in Islam?",
            answer="The five daily prayers are a foundational obligation (fard ayn) with universal consensus (ijma).",
            is_fiqh=True,
            is_religious=True,
            citations=[QuranCitation(surah=2, ayah_start=43, surah_name="Al-Baqarah")],
            hadith_refs=[HadithReference(raw="Sahih al-Bukhari 8", grade="SAHIH", verified=True, flagged=False)],
            citation_score=1.0,
        )
        assert q.epistemic_certainty is EpistemicCertainty.QATI
        assert q.position_type is PositionType.IJMA
        assert q.evidence_strength is EvidenceStrength.VERY_STRONG
        assert q.is_high_uncertainty is False
        assert q.limited_sources_warning is False
        assert "Allahu" in q.epistemic_humility_note or "Allah" in q.epistemic_humility_note

    def test_disputed_matter_pipeline_result(self):
        q = quantify_uncertainty(
            prompt="Does bleeding invalidate wudu?",
            answer="There is ikhtilaf: the Hanafi school holds that flowing blood invalidates wudu, while Shafi'i scholars disagree.",
            is_fiqh=True,
            is_religious=True,
            citations=[],
            hadith_refs=[],
            citation_score=0.0,
        )
        assert q.epistemic_certainty is EpistemicCertainty.DISPUTED
        assert q.position_type is PositionType.SCHOLARLY_IKHTILAF
        assert q.is_high_uncertainty is True
        assert q.limited_sources_warning is True
        assert len(q.reduction_paths) > 0
        assert any("Madhhab" in path for path in q.reduction_paths)

    def test_sensitive_fatwa_demands_expert_consultation(self):
        q = quantify_uncertainty(
            prompt="My husband pronounced talaq three times in anger, is my marriage over?",
            answer="Rulings on divorce depend on the exact state of mind, intention, and phrasing.",
            is_fiqh=True,
            is_religious=True,
        )
        assert q.requires_expert_consultation is True
        assert q.consultation_reason is not None
        assert "divorce" in q.consultation_reason or "talaq" in q.consultation_reason


# ---------------------------------------------------------------------------
# Confidence Module Integration Tests
# ---------------------------------------------------------------------------


class TestConfidenceIntegration:
    def test_assess_attaches_uncertainty_quantification(self):
        signals = build_signals(
            answer="Fasting Ramadan is obligatory by universal consensus.",
            is_religious=True,
            is_high_stakes=True,
            citation_verification=1.0,
            prompt="Is fasting Ramadan obligatory?",
        )
        assessment: ConfidenceAssessment = assess(signals)
        assert assessment.uncertainty is not None
        assert assessment.uncertainty.epistemic_certainty is EpistemicCertainty.QATI
        assert assessment.uncertainty.confidence_interval[0] <= assessment.uncertainty.confidence_score

    def test_assess_handles_disputed_matter_signals(self):
        signals = build_signals(
            answer="Scholars differ on raising hands in prayer before ruku.",
            is_religious=True,
            is_high_stakes=True,
            citation_verification=0.5,
            prompt="Should I raise my hands before ruku?",
        )
        assessment = assess(signals)
        assert assessment.uncertainty is not None
        assert assessment.uncertainty.epistemic_certainty is EpistemicCertainty.DISPUTED
        assert assessment.uncertainty.is_high_uncertainty is True


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestUncertaintyEndpoint:
    def test_taxonomy_endpoint_returns_valid_structure(self):
        response = client.get("/uncertainty/taxonomy")
        assert response.status_code == 200
        data = response.json()
        assert "epistemic_certainty" in data
        assert "qati" in data["epistemic_certainty"]
        assert "dhanni" in data["epistemic_certainty"]
        assert "disputed" in data["epistemic_certainty"]
        assert "position_types" in data
        assert "evidence_strengths" in data
        assert "uncertainty_factors" in data
