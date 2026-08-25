"""Tests for the Tajweed error detection and feedback system.

All offline — imports only the ``tajweed_detector`` module, no model calls,
no GEMINI_API_KEY, no FastAPI app boot.
"""

import pytest

from tajweed_detector import (
    TajweedAnalysis,
    TajweedError,
    FeedbackItem,
    RuleBreakdown,
    TajweedAnalysisModel,
    TajweedErrorModel,
    FeedbackItemModel,
    RuleListItem,
    RuleBreakdownModel,
    FeedbackRequest,
    PhoneticPosition,
    AnalyzeRequest,
    analyze_recitation,
    generate_feedback,
    get_all_rules,
    _build_context,
    _compute_score,
    _build_breakdown,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic phonetic sequences
# ---------------------------------------------------------------------------


def _pos(letter: str, vowel: str | None = None, **kw: object) -> dict:
    d: dict = {"letter": letter, "diacritics": list(kw.pop("diacritics", [])), **kw}
    if vowel is not None:
        d["vowel"] = vowel
    return d


# A correct ghunnah sequence: مِنْهُمْ — both ن and م have sukun + ghunnah held
SEQ_GHUNNAH_CORRECT = [
    _pos("م", vowel="kasra"),
    _pos("ن", vowel="sukun", ghunnah_held=True),
    _pos("ه", vowel="damma"),
    _pos("م", vowel="sukun", ghunnah_held=True),
]

# A wrong ghunnah sequence: مِنْهُمْ — neither held
SEQ_GHUNNAH_WRONG = [
    _pos("م", vowel="kasra"),
    _pos("ن", vowel="sukun", ghunnah_held=False),
    _pos("ه", vowel="damma"),
    _pos("م", vowel="sukun", ghunnah_held=False),
]

# Wrong qalqalah: ق with sukun, no echo
SEQ_QALQALAH_WRONG = [
    _pos("ق", vowel="sukun", qalqalah_echo=False),
]

# Correct qalqalah: ق with sukun, echo produced
SEQ_QALQALAH_CORRECT = [
    _pos("ق", vowel="sukun", qalqalah_echo=True),
]

# Wrong iqlab: نْ before ب, not converted
SEQ_IQLAB_WRONG = [
    _pos("ن", vowel="sukun", iqlab_applied=False),
    _pos("ب", vowel="fatha"),
]

# Correct iqlab
SEQ_IQLAB_CORRECT = [
    _pos("ن", vowel="sukun", iqlab_applied=True),
    _pos("ب", vowel="fatha"),
]

# Wrong ikhfa: نْ before ت, not concealed
SEQ_IKHFA_WRONG = [
    _pos("ن", vowel="sukun", concealed=False),
    _pos("ت", vowel="fatha"),
]

# Wrong idghaam: نْ before ر, not merged
SEQ_IDGHAAM_WRONG = [
    _pos("ن", vowel="sukun", merged=False),
    _pos("ر", vowel="fatha"),
]

# Wrong madd: alif held 1 harakat, expected 2
SEQ_MADD_WRONG = [
    _pos("ا", vowel="fatha", expected_madd_duration=2, actual_madd_duration=1, madd_type="madd_al_tabi_i"),
]

# Wrong madd: expected 4, held 3
SEQ_MADD_MUTASIL_WRONG = [
    _pos("و", diacritics=["sukun"], expected_madd_duration=4, actual_madd_duration=3, madd_type="madd_al_mutasil"),
]

# Wrong waqf
SEQ_WAQF_WRONG = [
    _pos("ن", vowel="fatha", pause=True, waqf_correct=False, waqf_type="mandatory", waqf_detail="Did not stop"),
]


# ---------------------------------------------------------------------------
# analyse_recitation — error detection
# ---------------------------------------------------------------------------


class TestErrorDetection:
    def test_ghunnah_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_WRONG)
        rule_ids = [e.rule_id for e in analysis.errors]
        assert "ghunnah" in rule_ids

    def test_ghunnah_correct_no_error(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_CORRECT)
        assert all(e.rule_id != "ghunnah" for e in analysis.errors)

    def test_qalqalah_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_QALQALAH_WRONG)
        assert any(e.rule_id == "qalqalah" for e in analysis.errors)

    def test_qalqalah_correct_no_error(self) -> None:
        analysis = analyze_recitation(SEQ_QALQALAH_CORRECT)
        assert all(e.rule_id != "qalqalah" for e in analysis.errors)

    def test_iqlab_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_IQLAB_WRONG)
        assert any(e.rule_id == "iqlab" for e in analysis.errors)

    def test_iqlab_correct_no_error(self) -> None:
        analysis = analyze_recitation(SEQ_IQLAB_CORRECT)
        assert all(e.rule_id != "iqlab" for e in analysis.errors)

    def test_ikhfa_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_IKHFA_WRONG)
        assert any(e.rule_id == "ikhfa" for e in analysis.errors)

    def test_idghaam_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_IDGHAAM_WRONG)
        assert any(e.rule_id == "idghaam" for e in analysis.errors)

    def test_madd_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_MADD_WRONG)
        madd_errors = [e for e in analysis.errors if "madd" in e.rule_id]
        assert madd_errors

    def test_madd_mutasil_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_MADD_MUTASIL_WRONG)
        madd_errors = [e for e in analysis.errors if "madd" in e.rule_id]
        assert madd_errors

    def test_waqf_violation_detected(self) -> None:
        analysis = analyze_recitation(SEQ_WAQF_WRONG)
        assert any(e.rule_id == "waqf_ibtida" for e in analysis.errors)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


class TestScore:
    def test_empty_sequence_perfect_score(self) -> None:
        analysis = analyze_recitation([])
        assert analysis.score == 1.0

    def test_score_between_0_and_1(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_WRONG)
        assert 0.0 <= analysis.score <= 1.0

    def test_perfect_recitation_score_one(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_CORRECT + SEQ_QALQALAH_CORRECT + SEQ_IQLAB_CORRECT)
        assert analysis.score == 1.0

    def test_score_decreases_with_errors(self) -> None:
        clean = analyze_recitation(SEQ_GHUNNAH_CORRECT)
        dirty = analyze_recitation(SEQ_GHUNNAH_WRONG)
        assert dirty.score < clean.score

    def test_multiple_errors_lower_score(self) -> None:
        combined = SEQ_GHUNNAH_WRONG + SEQ_QALQALAH_WRONG + SEQ_IQLAB_WRONG
        analysis = analyze_recitation(combined)
        single_gh = analyze_recitation(SEQ_GHUNNAH_WRONG)
        assert analysis.score < single_gh.score


# ---------------------------------------------------------------------------
# Breakdown
# ---------------------------------------------------------------------------


class TestBreakdown:
    def test_breakdown_populated_on_errors(self) -> None:
        combined = SEQ_GHUNNAH_WRONG + SEQ_QALQALAH_WRONG
        analysis = analyze_recitation(combined)
        rule_ids = [b.rule_id for b in analysis.breakdown]
        assert "ghunnah" in rule_ids
        assert "qalqalah" in rule_ids

    def test_breakdown_empty_on_clean(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_CORRECT + SEQ_QALQALAH_CORRECT + SEQ_IQLAB_CORRECT)
        assert analysis.breakdown == []

    def test_breakdown_weights_nonnegative(self) -> None:
        combined = SEQ_GHUNNAH_WRONG + SEQ_QALQALAH_WRONG + SEQ_IQLAB_WRONG
        analysis = analyze_recitation(combined)
        for b in analysis.breakdown:
            assert b.total_weight >= 0.0
            assert b.error_count >= 1


# ---------------------------------------------------------------------------
# Level filtering
# ---------------------------------------------------------------------------


class TestLevelFiltering:
    def test_beginner_excludes_advanced_rules(self) -> None:
        # waqf_ibtida is advanced-only
        analysis_advanced = analyze_recitation(SEQ_WAQF_WRONG, level="advanced")
        analysis_beginner = analyze_recitation(SEQ_WAQF_WRONG, level="beginner")
        assert any(e.rule_id == "waqf_ibtida" for e in analysis_advanced.errors)
        assert not any(e.rule_id == "waqf_ibtida" for e in analysis_beginner.errors)

    def test_intermediate_includes_intermediate_rules(self) -> None:
        # idghaam is intermediate
        analysis = analyze_recitation(SEQ_IDGHAAM_WRONG, level="intermediate")
        assert any(e.rule_id == "idghaam" for e in analysis.errors)

    def test_intermediate_excludes_advanced_rules(self) -> None:
        analysis = analyze_recitation(SEQ_WAQF_WRONG, level="intermediate")
        assert not any(e.rule_id == "waqf_ibtida" for e in analysis.errors)


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------


class TestFeedbackGeneration:
    def test_feedback_for_errors(self) -> None:
        errors = analyze_recitation(SEQ_GHUNNAH_WRONG + SEQ_QALQALAH_WRONG).errors
        items = generate_feedback(errors)
        assert len(items) >= 1
        assert all(isinstance(i, FeedbackItem) for i in items)

    def test_feedback_groups_by_rule(self) -> None:
        # Two ghunnah errors → single feedback item with occurrences=2
        errors = analyze_recitation(SEQ_GHUNNAH_WRONG).errors
        items = generate_feedback(errors, level="advanced")
        ghunnah_items = [i for i in items if i.rule_id == "ghunnah"]
        assert len(ghunnah_items) == 1
        assert ghunnah_items[0].occurrences == 2  # 2 positions with ghunnah violations

    def test_feedback_exercises_are_lists(self) -> None:
        errors = analyze_recitation(SEQ_QALQALAH_WRONG).errors
        items = generate_feedback(errors)
        for item in items:
            assert isinstance(item.exercises, list)
            assert len(item.exercises) >= 1

    def test_feedback_empty_on_no_errors(self) -> None:
        errors = analyze_recitation(SEQ_GHUNNAH_CORRECT).errors
        items = generate_feedback(errors)
        assert items == []

    def test_feedback_level_filters(self) -> None:
        errors = analyze_recitation(SEQ_WAQF_WRONG).errors
        advanced_items = generate_feedback(errors, level="advanced")
        beginner_items = generate_feedback(errors, level="beginner")
        assert any(i.rule_id == "waqf_ibtida" for i in advanced_items)
        assert not any(i.rule_id == "waqf_ibtida" for i in beginner_items)


# ---------------------------------------------------------------------------
# get_all_rules
# ---------------------------------------------------------------------------


class TestGetAllRules:
    def test_returns_nonempty_list(self) -> None:
        rules = get_all_rules()
        assert len(rules) >= 6

    def test_each_rule_has_required_fields(self) -> None:
        for rule in get_all_rules():
            assert "rule_id" in rule
            assert "name" in rule
            assert "description" in rule
            assert "severity_weight" in rule
            assert "level" in rule
            assert "exercises" in rule

    def test_rules_include_core_tajweed_topics(self) -> None:
        ids = {r["rule_id"] for r in get_all_rules()}
        assert "ghunnah" in ids
        assert "qalqalah" in ids
        assert "iqlab" in ids
        assert "ikhfa" in ids
        assert "idghaam" in ids


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_builds_context_from_dict(self) -> None:
        seq = [_pos("ن", vowel="sukun"), _pos("ب", vowel="fatha")]
        ctx = _build_context(seq, 0)
        assert ctx.letter == "ن"
        assert ctx.vowel == "sukun"
        assert ctx.next_letter == "ب"
        assert ctx.prev_letter is None

    def test_word_boundary_flags(self) -> None:
        seq = [
            _pos("م", vowel="kasra", word_start=True),
            _pos("ن", vowel="sukun"),
            _pos("ه", vowel="damma", word_end=True),
        ]
        ctx_start = _build_context(seq, 0)
        ctx_end = _build_context(seq, 2)
        assert ctx_start.is_word_start is True
        assert ctx_end.is_word_end is True

    def test_pause_flag(self) -> None:
        seq = [_pos("ن", vowel="fatha", pause=True)]
        ctx = _build_context(seq, 0)
        assert ctx.is_pause is True


# ---------------------------------------------------------------------------
# Pydantic models & API schema
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_phonetic_position_defaults(self) -> None:
        p = PhoneticPosition(letter="ن")
        assert p.letter == "ن"
        assert p.diacritics == []
        assert p.vowel is None
        assert p.pause is False

    def test_analyze_request_validates(self) -> None:
        req = AnalyzeRequest(
            phonetic_sequence=[PhoneticPosition(letter="ن", vowel="sukun")],
            level="beginner",
        )
        assert len(req.phonetic_sequence) == 1
        assert req.level == "beginner"

    def test_analyze_request_rejects_empty(self) -> None:
        with pytest.raises(Exception):
            AnalyzeRequest(phonetic_sequence=[], level="advanced")

    def test_analysis_model_from_analysis(self) -> None:
        analysis = analyze_recitation(SEQ_GHUNNAH_WRONG)
        model = TajweedAnalysisModel(
            errors=[TajweedErrorModel(**vars(e)) for e in analysis.errors],
            score=analysis.score,
            total_positions=analysis.total_positions,
            breakdown=[RuleBreakdownModel(**vars(b)) for b in analysis.breakdown],
            level=analysis.level,
        )
        dumped = model.model_dump()
        assert "score" in dumped
        assert "errors" in dumped
        assert "breakdown" in dumped
        assert "level" in dumped

    def test_feedback_request_model(self) -> None:
        analysis = analyze_recitation(SEQ_QALQALAH_WRONG)
        errors = [TajweedErrorModel(**vars(e)) for e in analysis.errors]
        req = FeedbackRequest(errors=errors, level="intermediate")
        assert req.level == "intermediate"

    def test_rule_list_item_model(self) -> None:
        rules = get_all_rules()
        for r in rules:
            item = RuleListItem(**r)
            assert item.rule_id

    def test_tajweed_error_fields(self) -> None:
        err = TajweedError(
            rule_id="test",
            rule_name="Test Rule",
            position=0,
            severity=0.1,
            detail="detail",
            explanation="explanation",
            exercise="exercise",
        )
        assert err.rule_id == "test"
        assert err.position == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        r1 = analyze_recitation(SEQ_GHUNNAH_WRONG)
        r2 = analyze_recitation(SEQ_GHUNNAH_WRONG)
        assert r1.score == r2.score
        assert len(r1.errors) == len(r2.errors)
        assert [e.rule_id for e in r1.errors] == [e.rule_id for e in r2.errors]
