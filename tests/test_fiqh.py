"""Tests for the fiqh module — no live API calls."""

import json
import os

import pytest

from fiqh import (
    FIQH_IKHTILAF_CONTEXT,
    MADHHAB_LEAD_INSTRUCTION,
    FiqhInfo,
    VALID_MADHHABS,
    classify_fiqh,
    keyword_match,
    normalize_madhhab,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

FIQH_CASES_PATH = os.path.join(FIXTURE_DIR, "fiqh_cases.jsonl")


def load_cases():
    cases = []
    with open(FIQH_CASES_PATH) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Madhhab normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("hanafi", "hanafi"),
    ("Hanafi", "hanafi"),
    ("hanafee", "hanafi"),
    ("maliki", "maliki"),
    ("Maliki", "maliki"),
    ("shafii", "shafii"),
    ("shafi'i", "shafii"),
    ("Shafi'i", "shafii"),
    ("shafie", "shafii"),
    ("hanbali", "hanbali"),
    ("Hanbali", "hanbali"),
])
def test_normalize_valid(raw, expected):
    assert normalize_madhhab(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_normalize_empty_degrades_to_none(raw):
    assert normalize_madhhab(raw) is None


def test_normalize_unknown_degrades_to_none():
    result = normalize_madhhab("zahiri")
    assert result is None


# ---------------------------------------------------------------------------
# Keyword pre-filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Does touching a woman break wudu?", True),
    ("What is the ruling on marriage?", True),
    ("Is this halal?", True),
    ("How do I pray?", True),
    ("Can I fast while travelling?", True),
    ("What is the story of Prophet Musa?", False),
    ("Hello, how are you?", False),
    ("Explain the concept of tawheed.", False),
    ("What is the capital of Egypt?", False),
])
def test_keyword_match(text, expected):
    assert keyword_match(text) == expected


# ---------------------------------------------------------------------------
# classify_fiqh with no classify_fn (uses keyword pre-filter only)
# ---------------------------------------------------------------------------


def test_classify_fiqh_no_classifier():
    assert classify_fiqh("What is the ruling on divorce?", classify_fn=None) is True
    assert classify_fiqh("How was your day?", classify_fn=None) is False


def test_classify_fiqh_with_classifier():
    def always_true(prompt):
        return True

    def always_false(prompt):
        return False

    # keyword match + classifier says true
    assert classify_fiqh("What is the ruling on divorce?", always_true) is True
    # keyword match + classifier says false
    assert classify_fiqh("What is the ruling on divorce?", always_false) is False
    # no keyword match — classifier not called
    assert classify_fiqh("Hello world", always_true) is False


# ---------------------------------------------------------------------------
# FiqhInfo model
# ---------------------------------------------------------------------------


def test_fiqh_info_default():
    info = FiqhInfo(is_fiqh_question=False)
    assert info.is_fiqh_question is False
    assert info.madhhab_requested is None


def test_fiqh_info_with_madhhab():
    info = FiqhInfo(is_fiqh_question=True, madhhab_requested="shafii")
    assert info.is_fiqh_question is True
    assert info.madhhab_requested == "shafii"


# ---------------------------------------------------------------------------
# Context block structure
# ---------------------------------------------------------------------------


def test_fiqh_ikhtilaf_context_has_madhhabs():
    for school in ("Hanafi", "Maliki", "Shafi'i", "Hanbali"):
        assert school in FIQH_IKHTILAF_CONTEXT


def test_madhhab_lead_instruction_format():
    result = MADHHAB_LEAD_INSTRUCTION.format(madhhab="shafii")
    assert "shafii" in result


# ---------------------------------------------------------------------------
# Eval case structure
# ---------------------------------------------------------------------------


def test_fiqh_cases_jsonl_loads():
    cases = load_cases()
    assert len(cases) >= 8
    for case in cases:
        assert "prompt" in case
        assert "is_fiqh" in case
        assert "expect_schools" in case


def test_non_fiqh_cases_have_empty_schools():
    for case in load_cases():
        if not case["is_fiqh"]:
            assert case["expect_schools"] == []


# ---------------------------------------------------------------------------
# Keyword pre-filter: eval cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", [c for c in load_cases() if c["is_fiqh"]], ids=lambda c: c["prompt"][:40])
def test_fiqh_cases_keyword_match_true(case):
    assert keyword_match(case["prompt"]), f"Expected keyword match for fiqh case: {case['prompt'][:60]}"
