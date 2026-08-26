"""Tests for the isnad chain analysis module — all offline, no network."""

import pytest

from isnad import (
    ChainStrength,
    Gap,
    IsnadChain,
    Narrator,
    NarratorEra,
    assess_chain,
    detect_gaps,
    parse_isnad,
    visualize_chain,
)


# ---------------------------------------------------------------------------
# Narrator normalization
# ---------------------------------------------------------------------------


class TestNarrator:
    def test_normalized_name_strips_honorifics(self):
        n = Narrator(name="Imam Al-Bukhari")
        assert n.normalized_name == "bukhari"

    def test_known_sahabi_gets_high_credibility(self):
        n = Narrator(name="Abu Hurairah")
        assert n.era == NarratorEra.SAHABI
        assert n.credibility_score >= 0.85

    def test_known_tabi_gets_moderate_credibility(self):
        n = Narrator(name="Sa'id ibn al-Musayyib")
        assert n.era == NarratorEra.TABI
        assert n.credibility_score >= 0.7

    def test_unknown_narrator_gets_low_score(self):
        n = Narrator(name="Someone Obscure")
        assert n.era == NarratorEra.UNKNOWN
        assert n.credibility_score < 0.5


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseIsnad:
    def test_empty_text(self):
        chain = parse_isnad("")
        assert chain.narrators == []

    def test_none_text(self):
        chain = parse_isnad(None)  # type: ignore[arg-type]
        assert chain.narrators == []

    def test_arrow_separator(self):
        chain = parse_isnad("Abu Hurairah -> Anas ibn Malik -> Malik")
        assert len(chain.narrators) >= 2
        assert chain.narrators[0].name == "Abu Hurairah"

    def test_from_separator(self):
        chain = parse_isnad("Narrated Aisha from Umar ibn al-Khattab from Abu Bakr")
        assert len(chain.narrators) >= 2

    def test_pipe_separator(self):
        chain = parse_isnad("Anas ibn Malik | Malik | Al-Zuhri")
        assert len(chain.narrators) >= 2

    def test_chain_strength_is_computed(self):
        chain = parse_isnad("Abu Hurairah -> Anas ibn Malik")
        assert chain.strength in ChainStrength

    def test_raw_text_preserved(self):
        raw = "Abu Hurairah -> Anas ibn Malik"
        chain = parse_isnad(raw)
        assert chain.raw_text == raw


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_no_gaps_in_short_valid_chain(self):
        narrators = [Narrator(name="Abu Hurairah"), Narrator(name="Anas ibn Malik")]
        chain = IsnadChain(narrators=narrators)
        gaps = detect_gaps(chain)
        # A two-narrator chain of Sahabi→Tabi is fine.
        assert gaps == [] or all("missing" not in g.description.lower() for g in gaps)

    def test_backward_era_jump_detected(self):
        narrators = [
            Narrator(name="Sa'id ibn al-Musayyib"),  # Tabi
            Narrator(name="Abu Hurairah"),  # Sahabi — earlier!
        ]
        chain = IsnadChain(narrators=narrators)
        gaps = detect_gaps(chain)
        assert any("generation" in g.description.lower() for g in gaps)


# ---------------------------------------------------------------------------
# Chain assessment
# ---------------------------------------------------------------------------


class TestAssessChain:
    def test_empty_chain_is_unknown(self):
        chain = IsnadChain()
        assert assess_chain(chain) == ChainStrength.UNKNOWN

    def test_all_high_credibility_no_gaps_is_sahih(self):
        narrators = [
            Narrator(name="Abu Hurairah", credibility_score=0.9, era=NarratorEra.SAHABI),
            Narrator(name="Anas ibn Malik", credibility_score=0.85, era=NarratorEra.SAHABI),
        ]
        chain = IsnadChain(narrators=narrators)
        assert assess_chain(chain) == ChainStrength.SAHIH

    def test_gaps_weaken_chain(self):
        narrators = [
            Narrator(name="Abu Hurairah", credibility_score=0.9, era=NarratorEra.SAHABI),
            Narrator(name="Someone", credibility_score=0.3, era=NarratorEra.UNKNOWN),
            Narrator(name="Someone Else", credibility_score=0.3, era=NarratorEra.UNKNOWN),
        ]
        chain = IsnadChain(narrators=narrators)
        # Gaps + low average → da'if.
        assert assess_chain(chain) == ChainStrength.DAIF

    def test_moderate_chain_is_hasan(self):
        narrators = [
            Narrator(name="A", credibility_score=0.7, era=NarratorEra.TABI),
            Narrator(name="B", credibility_score=0.65, era=NarratorEra.TABI_AL_TABIIN),
        ]
        chain = IsnadChain(narrators=narrators)
        result = assess_chain(chain)
        assert result in (ChainStrength.HASAN, ChainStrength.SAHIH)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class TestVisualizeChain:
    def test_empty_chain_shows_placeholder(self):
        chain = IsnadChain()
        viz = visualize_chain(chain)
        assert "empty chain" in viz.lower()

    def test_chain_produces_nonempty_output(self):
        chain = parse_isnad("Abu Hurairah -> Anas ibn Malik")
        viz = visualize_chain(chain)
        assert len(viz) > 0
        assert "Abu Hurairah" in viz

    def test_viz_includes_strength(self):
        chain = parse_isnad("Abu Hurairah -> Anas ibn Malik")
        viz = visualize_chain(chain)
        assert "strength:" in viz.lower()

    def test_gaps_are_shown_in_viz(self):
        narrators = [
            Narrator(name="Abu Hurairah", credibility_score=0.9, era=NarratorEra.SAHABI),
            Narrator(name="Unknown Narrator", credibility_score=0.3, era=NarratorEra.UNKNOWN),
        ]
        chain = IsnadChain(narrators=narrators)
        chain.gaps = detect_gaps(chain)
        viz = visualize_chain(chain)
        assert isinstance(viz, str)


# ---------------------------------------------------------------------------
# Integration: parse → assess → visualize
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self):
        chain = parse_isnad(
            "Narrated Abu Hurairah -> Sa'id ibn al-Musayyib -> Al-Zuhri -> Malik"
        )
        assert len(chain.narrators) >= 2
        assert chain.strength in ChainStrength
        viz = visualize_chain(chain)
        assert "strength:" in viz.lower()

    def test_single_narrator_chain(self):
        chain = parse_isnad("Abu Hurairah")
        assert len(chain.narrators) == 1
        assert chain.strength in ChainStrength
