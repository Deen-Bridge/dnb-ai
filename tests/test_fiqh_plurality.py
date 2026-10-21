"""Tests for multi-madhhab fiqh difference representation (#527)."""

from __future__ import annotations

from fiqh_plurality import MadhhabPosition, format_fiqh_plurality_response


def test_recognized_differences_preserve_school_attribution_and_avoid_universal_claim() -> None:
    positions = [
        MadhhabPosition("Hanafi", "Witr is wajib with three rak'ahs.", "Al-Hidayah 1/66"),
        MadhhabPosition("Shafi'i", "Witr is confirmed sunnah (sunnah mu'akkadah).", "Al-Majmu' 4/13"),
    ]
    res = format_fiqh_plurality_response("Witr prayer ruling", positions, high_impact_personal_case=False)
    assert res.presents_single_universal_ruling is False
    assert len(res.positions) == 2
    assert "not a binding personal fatwa" in res.educational_boundary_note


def test_insufficient_evidence_does_not_fabricate_consensus() -> None:
    res = format_fiqh_plurality_response("Unindexed contemporary issue", [], high_impact_personal_case=True)
    assert res.positions == []
    assert "Insufficient sourced evidence" in res.educational_boundary_note
