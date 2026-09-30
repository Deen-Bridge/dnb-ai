"""Tests for the multi-step Islamic reasoning-chain engine (#152).

All offline — no model calls, no GEMINI_API_KEY. Imports only the module under
test, never main.py.
"""

import reasoning_chains
from reasoning_chains import (
    Connector,
    EvidenceRef,
    ReasoningStep,
    SourceType,
    StepSubmission,
    build_chain,
    decompose,
    find_contradictions,
    find_weak_points,
    list_templates,
    validate_chain,
)


def _step(step_id: str, conclusion: str, confidence: float, evidence: list[EvidenceRef] | None = None) -> ReasoningStep:
    evidence = evidence or []
    return ReasoningStep(
        id=step_id,
        order=1,
        facet="",
        source_type=evidence[0].source_type if evidence else SourceType.GENERAL,
        intermediate_conclusion=conclusion,
        evidence=evidence,
        confidence=confidence,
        supported=bool(evidence),
    )


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_compound_question_yields_multiple_steps() -> None:
    steps = decompose("Is coffee permissible and what does the hadith say about it?")
    assert len(steps) >= 2
    # Steps are ordered and carry stable, unique ids.
    assert [s.order for s in steps] == list(range(1, len(steps) + 1))
    assert len({s.id for s in steps}) == len(steps)


def test_every_step_has_evidence_and_confidence() -> None:
    steps = decompose("What is the ruling on fasting while travelling?")
    for step in steps:
        assert 0.0 <= step.confidence <= 1.0
        # A recognised facet is supported by at least one evidence reference.
        assert step.evidence, f"{step.id} should carry evidence"
        assert step.supported is True


def test_connectors_link_steps() -> None:
    steps = decompose("Is music permissible and is singing forbidden?")
    # Every step but the last points forward with a logical connector.
    assert all(s.connector is not Connector.NONE for s in steps[:-1])
    assert steps[-1].connector is Connector.NONE


# ---------------------------------------------------------------------------
# Consistency / contradiction
# ---------------------------------------------------------------------------


def test_consistency_flags_contradiction() -> None:
    steps = [
        _step("step-1", "The ruling is that music is permissible.", 0.7),
        _step("step-2", "The ruling is that music is forbidden.", 0.7),
    ]
    issues = find_contradictions(steps)
    assert issues, "opposite conclusions on a shared topic must be flagged"
    assert issues[0].kind == "contradiction"
    assert set(issues[0].step_ids) == {"step-1", "step-2"}

    report = validate_chain(steps)
    assert report.consistent is False


def test_consistent_chain_reports_no_contradiction() -> None:
    steps = decompose("What is the ruling on fasting while travelling?")
    report = validate_chain(steps)
    assert report.consistent is True


# ---------------------------------------------------------------------------
# Weak points
# ---------------------------------------------------------------------------


def test_weak_point_detection_returns_lowest_confidence_step() -> None:
    quran = [EvidenceRef(source_type=SourceType.QURAN, reference="Qur'an (quran.com)")]
    steps = [
        _step("step-1", "The Qur'anic evidence is weighed.", 0.82, quran),
        _step("step-2", "No specific evidence is identified.", 0.30),
        _step("step-3", "The narrations are examined.", 0.68, quran),
    ]
    weak = find_weak_points(steps)
    assert weak, "there must be at least one weak point"
    assert weak[0].step_id == "step-2"
    assert weak[0].confidence == 0.30


# ---------------------------------------------------------------------------
# Madhhab branching
# ---------------------------------------------------------------------------


def test_madhhab_question_yields_parallel_branches() -> None:
    chain = build_chain("What is the ruling on combining prayers?", madhhab="hanafi")
    assert chain.branches, "a madhhab-tagged question must fan out into branches"
    assert len(chain.branches) == len(reasoning_chains.MADHHABS)
    # Each branch is a full, independently-addressable parallel path.
    for branch in chain.branches:
        assert branch.steps
        assert all(step.id.startswith(f"{branch.madhhab}-") for step in branch.steps)
    # Branch step ids do not collide across paths.
    all_ids = [step.id for branch in chain.branches for step in branch.steps]
    assert len(all_ids) == len(set(all_ids))


def test_plain_question_has_no_branches() -> None:
    chain = build_chain("What time is Fajr?")
    assert chain.branches == []


# ---------------------------------------------------------------------------
# Rendering and templates
# ---------------------------------------------------------------------------


def test_chain_renders_markdown_with_step_ids() -> None:
    chain = build_chain("Is coffee permissible and what does the hadith say?")
    outline = reasoning_chains.render_markdown(chain)
    assert chain.question in outline
    for step in chain.steps:
        assert f"[{step.id}]" in outline


def test_templates_endpoint_data_shape() -> None:
    templates = list_templates()
    assert len(templates) >= 3
    patterns = {t.pattern for t in templates}
    assert {"ruling", "tafsir"} <= patterns
    for template in templates:
        assert template.steps, "each template needs ordered step prompts"
        assert template.source_types


def test_validate_accepts_submitted_steps() -> None:
    submissions = [
        StepSubmission(intermediate_conclusion="This action is permissible.", confidence=0.7),
        StepSubmission(intermediate_conclusion="This action is forbidden.", confidence=0.4),
    ]
    steps = [_step(f"step-{i}", s.intermediate_conclusion, s.confidence) for i, s in enumerate(submissions, start=1)]
    report = validate_chain(steps)
    assert report.consistent is False
    assert report.weak_points[0].step_id == "step-2"
