"""Tests for the personalized learning-path endpoint.

All tests run offline with the model mocked via an injected ``FakeGenerator``
(or ``USE_FAKE_GENERATOR``) — no GEMINI_API_KEY and no network are used.
The catalog is always an in-memory fixture, and every recommended course_id is
asserted against that fixture.
"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from learning import (
    COMPLETION_THRESHOLD,
    MAX_CATALOG_SIZE,
    MAX_NOTES_LENGTH,
    CourseCatalogItem,
    FakeGenerator,
    LearnerLevel,
    LearnerProfile,
    LearningGoal,
    LearningPath,
    LearningPathRequest,
    LearningStep,
    ProgressRecord,
    _apply_guardrails,
    _build_prompt,
    _generate_learning_path,
    _parse_steps,
    get_generator,
    pydantic_to_gemini_schema,
)
from main import app

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_catalog() -> list[CourseCatalogItem]:
    return [
        CourseCatalogItem(
            course_id="arabic-101",
            title="Arabic Alphabet",
            category="arabic",
            level=LearnerLevel.beginner,
            prerequisites=[],
            description="Learn the Arabic letters and their sounds.",
        ),
        CourseCatalogItem(
            course_id="quran-101",
            title="Quran Reading Basics",
            category="quran",
            level=LearnerLevel.beginner,
            prerequisites=["arabic-101"],
            description="Read the Quran from the mushaf with correct letters.",
        ),
        CourseCatalogItem(
            course_id="tajweed-201",
            title="Introduction to Tajweed",
            category="quran",
            level=LearnerLevel.intermediate,
            prerequisites=["quran-101"],
            description="Rules of recitation: makharij and basic ahkam.",
        ),
    ]


def make_progress() -> list[ProgressRecord]:
    return [
        ProgressRecord(
            course_id="arabic-101",
            title="Arabic Alphabet",
            category="arabic",
            level=LearnerLevel.beginner,
            completion_pct=100,
        )
    ]


def make_request(**overrides) -> LearningPathRequest:
    data = {
        "profile": LearnerProfile(
            level=LearnerLevel.beginner,
            goals=[LearningGoal.quran_reading, LearningGoal.tajweed],
            time_per_week_hours=5,
            notes="Reverted last year.",
        ),
        "progress": make_progress(),
        "catalog": make_catalog(),
    }
    data.update(overrides)
    return LearningPathRequest(**data)


def make_step(course_id: str, order: int, title: str = "Course") -> dict:
    return {
        "course_id": course_id,
        "title": title,
        "order": order,
        "reason": f"Recommended because it advances the learner ({course_id}).",
        "prerequisites_satisfied": True,
        "estimated_weeks": 6,
    }


def valid_response(course_ids: list[str]) -> str:
    steps = [make_step(cid, i + 1) for i, cid in enumerate(course_ids)]
    return json.dumps(
        {
            "steps": steps,
            "summary": "A path toward fluent Quran reading and tajweed.",
            "scholarly_note": "This is study guidance, not a fatwa; learn under a qualified teacher.",
        }
    )


def request_payload(**overrides) -> dict:
    payload = {
        "profile": {
            "level": "beginner",
            "goals": ["quran_reading", "tajweed"],
            "time_per_week_hours": 5,
            "notes": "Reverted last year.",
        },
        "progress": [
            {
                "course_id": "arabic-101",
                "title": "Arabic Alphabet",
                "category": "arabic",
                "level": "beginner",
                "completion_pct": 100,
            }
        ],
        "catalog": [
            {
                "course_id": "arabic-101",
                "title": "Arabic Alphabet",
                "category": "arabic",
                "level": "beginner",
                "prerequisites": [],
                "description": "Learn the Arabic letters and their sounds.",
            },
            {
                "course_id": "quran-101",
                "title": "Quran Reading Basics",
                "category": "quran",
                "level": "beginner",
                "prerequisites": ["arabic-101"],
                "description": "Read the Quran from the mushaf.",
            },
            {
                "course_id": "tajweed-201",
                "title": "Introduction to Tajweed",
                "category": "quran",
                "level": "intermediate",
                "prerequisites": ["quran-101"],
                "description": "Rules of recitation.",
            },
        ],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client_with_generator():
    """Yield ``(client, install)`` where ``install(gen)`` overrides the route
    generator with a specific ``FakeGenerator`` and clears the override on exit.
    """
    client = TestClient(app)

    def install(gen: FakeGenerator) -> None:
        app.dependency_overrides[get_generator] = lambda: gen

    try:
        yield client, install
    finally:
        app.dependency_overrides.pop(get_generator, None)


# ---------------------------------------------------------------------------
# Request-model validation
# ---------------------------------------------------------------------------


class TestRequestModels:
    def test_valid_request(self):
        req = make_request()
        assert len(req.catalog) == 3
        assert req.profile.goals[0] == LearningGoal.quran_reading

    def test_empty_catalog_rejected(self):
        with pytest.raises(ValidationError):
            make_request(catalog=[])

    def test_oversized_catalog_rejected(self):
        item = make_catalog()[0]
        big = [item.model_copy(update={"course_id": f"c-{i}"}) for i in range(MAX_CATALOG_SIZE + 1)]
        with pytest.raises(ValidationError):
            make_request(catalog=big)

    def test_duplicate_catalog_ids_rejected(self):
        cat = make_catalog()
        cat[1] = cat[1].model_copy(update={"course_id": "arabic-101"})
        with pytest.raises(ValidationError, match="unique"):
            make_request(catalog=cat)

    def test_goals_required(self):
        with pytest.raises(ValidationError):
            LearnerProfile(level=LearnerLevel.beginner, goals=[])

    def test_notes_length_capped(self):
        with pytest.raises(ValidationError):
            LearnerProfile(
                level=LearnerLevel.beginner,
                goals=[LearningGoal.arabic],
                notes="x" * (MAX_NOTES_LENGTH + 1),
            )

    def test_completion_pct_bounds(self):
        with pytest.raises(ValidationError):
            ProgressRecord(
                course_id="c1",
                title="T",
                category="cat",
                level=LearnerLevel.beginner,
                completion_pct=101,
            )

    def test_quiz_scores_range(self):
        with pytest.raises(ValidationError, match="between 0 and 100"):
            ProgressRecord(
                course_id="c1",
                title="T",
                category="cat",
                level=LearnerLevel.beginner,
                completion_pct=50,
                quiz_scores=[50, 200],
            )


# ---------------------------------------------------------------------------
# Response-model validators
# ---------------------------------------------------------------------------


class TestLearningPathValidators:
    def _steps(self, ids_orders: list[tuple[str, int]]) -> list[LearningStep]:
        return [LearningStep(**make_step(cid, order)) for cid, order in ids_orders]

    def test_valid_path(self):
        path = LearningPath(
            steps=self._steps([("quran-101", 1), ("tajweed-201", 2)]),
            summary="Summary.",
            scholarly_note="Not a fatwa.",
        )
        assert len(path.steps) == 2

    def test_duplicate_course_id_rejected(self):
        with pytest.raises(ValidationError, match="duplicate course_id"):
            LearningPath(
                steps=self._steps([("quran-101", 1), ("quran-101", 2)]),
                summary="S",
                scholarly_note="N",
            )

    def test_noncontiguous_order_rejected(self):
        with pytest.raises(ValidationError, match="contiguous"):
            LearningPath(
                steps=self._steps([("quran-101", 1), ("tajweed-201", 3)]),
                summary="S",
                scholarly_note="N",
            )

    def test_blank_reason_rejected(self):
        with pytest.raises(ValidationError, match="reason"):
            LearningStep(
                course_id="quran-101",
                title="T",
                order=1,
                reason="   ",
                prerequisites_satisfied=True,
                estimated_weeks=4,
            )

    def test_empty_steps_rejected(self):
        with pytest.raises(ValidationError):
            LearningPath(steps=[], summary="S", scholarly_note="N")

    def test_catalog_context_rejects_unknown_id(self):
        payload = {
            "steps": [make_step("ghost-999", 1)],
            "summary": "S",
            "scholarly_note": "N",
        }
        with pytest.raises(ValidationError, match="not in catalog"):
            LearningPath.model_validate(payload, context={"catalog_ids": {"quran-101"}})

    def test_catalog_context_accepts_known_id(self):
        payload = {
            "steps": [make_step("quran-101", 1)],
            "summary": "S",
            "scholarly_note": "N",
        }
        path = LearningPath.model_validate(payload, context={"catalog_ids": {"quran-101"}})
        assert path.steps[0].course_id == "quran-101"


# ---------------------------------------------------------------------------
# Schema translation (reused from study.py)
# ---------------------------------------------------------------------------


class TestSchema:
    def test_no_defs_and_required_fields_present(self):
        schema = pydantic_to_gemini_schema(LearningPath)
        assert "$defs" not in schema
        assert "steps" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "scholarly_note" in schema["properties"]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_catalog_and_goals(self):
        prompt = _build_prompt(make_request())
        assert "quran-101" in prompt
        assert "tajweed-201" in prompt
        assert "quran_reading" in prompt
        assert "recommend only" in prompt.lower() or "only from" in prompt.lower()

    def test_prompt_notes_new_learner(self):
        prompt = _build_prompt(make_request(progress=[]))
        assert "new learner" in prompt.lower()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseSteps:
    def test_valid(self):
        steps, summary, note = _parse_steps(valid_response(["quran-101", "tajweed-201"]))
        assert len(steps) == 2
        assert summary
        assert note

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_steps("{not json")

    def test_missing_steps(self):
        with pytest.raises(ValueError, match="steps"):
            _parse_steps(json.dumps({"summary": "s", "scholarly_note": "n"}))

    def test_missing_summary(self):
        payload = {"steps": [make_step("quran-101", 1)], "scholarly_note": "n"}
        with pytest.raises(ValueError, match="summary"):
            _parse_steps(json.dumps(payload))


# ---------------------------------------------------------------------------
# Deterministic grounding guardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_drops_non_catalog_id(self):
        req = make_request()
        steps = [LearningStep(**make_step("ghost-999", 1)), LearningStep(**make_step("quran-101", 2))]
        survivors = _apply_guardrails(steps, req)
        ids = {s.course_id for s in survivors}
        assert "ghost-999" not in ids
        assert "quran-101" in ids

    def test_drops_completed_course(self):
        req = make_request()  # arabic-101 completed 100%
        steps = [LearningStep(**make_step("arabic-101", 1)), LearningStep(**make_step("quran-101", 2))]
        survivors = _apply_guardrails(steps, req)
        ids = {s.course_id for s in survivors}
        assert "arabic-101" not in ids
        assert "quran-101" in ids

    def test_drops_prerequisite_violation(self):
        # New learner (no progress): tajweed-201 requires quran-101 which is absent.
        req = make_request(progress=[])
        steps = [LearningStep(**make_step("tajweed-201", 1))]
        survivors = _apply_guardrails(steps, req)
        assert survivors == []

    def test_prereq_satisfied_by_earlier_step(self):
        req = make_request(progress=[])
        steps = [
            LearningStep(**make_step("arabic-101", 1)),
            LearningStep(**make_step("quran-101", 2)),
            LearningStep(**make_step("tajweed-201", 3)),
        ]
        survivors = _apply_guardrails(steps, req)
        assert [s.course_id for s in survivors] == ["arabic-101", "quran-101", "tajweed-201"]
        assert [s.order for s in survivors] == [1, 2, 3]

    def test_renumbers_after_drop(self):
        req = make_request()
        steps = [
            LearningStep(**make_step("ghost-999", 1)),
            LearningStep(**make_step("quran-101", 2)),
            LearningStep(**make_step("tajweed-201", 3)),
        ]
        survivors = _apply_guardrails(steps, req)
        assert [s.order for s in survivors] == [1, 2]

    def test_dedupes(self):
        req = make_request()
        steps = [LearningStep(**make_step("quran-101", 1)), LearningStep(**make_step("quran-101", 2))]
        survivors = _apply_guardrails(steps, req)
        assert len(survivors) == 1


# ---------------------------------------------------------------------------
# Generation loop with retry / 502
# ---------------------------------------------------------------------------


class TestGenerateLoop:
    def test_happy_path(self):
        gen = FakeGenerator(responses=[valid_response(["quran-101", "tajweed-201"])])
        path = _generate_learning_path(gen, make_request())
        assert [s.course_id for s in path.steps] == ["quran-101", "tajweed-201"]

    def test_adversarial_output_repaired_via_retry(self):
        # First response smuggles a non-catalog id (only step) -> empty -> retry.
        bad = valid_response(["ghost-999"])
        good = valid_response(["quran-101", "tajweed-201"])
        gen = FakeGenerator(responses=[bad, good])
        path = _generate_learning_path(gen, make_request(), max_retries=2)
        ids = {s.course_id for s in path.steps}
        assert "ghost-999" not in ids
        assert ids == {"quran-101", "tajweed-201"}

    def test_adversarial_output_repaired_in_place(self):
        # A single response mixing a ghost id with a real one is repaired, not retried away.
        mixed = json.dumps(
            {
                "steps": [make_step("ghost-999", 1), make_step("quran-101", 2)],
                "summary": "s",
                "scholarly_note": "n",
            }
        )
        gen = FakeGenerator(responses=[mixed])
        path = _generate_learning_path(gen, make_request())
        assert {s.course_id for s in path.steps} == {"quran-101"}
        assert gen.call_count == 1

    def test_persistent_ungrounded_output_raises_502(self):
        from fastapi import HTTPException

        bad = valid_response(["ghost-999"])
        gen = FakeGenerator(responses=[bad, bad, bad])
        with pytest.raises(HTTPException) as exc:
            _generate_learning_path(gen, make_request(), max_retries=2)
        assert exc.value.status_code == 502
        assert exc.value.detail["retries_used"] == 3

    def test_invalid_json_exhausted_raises_502(self):
        from fastapi import HTTPException

        gen = FakeGenerator(responses=["{bad", "{bad", "{bad"])
        with pytest.raises(HTTPException) as exc:
            _generate_learning_path(gen, make_request(), max_retries=2)
        assert exc.value.status_code == 502
        assert "last_validation_error" in exc.value.detail


# ---------------------------------------------------------------------------
# Route (TestClient) — the "registered + reachable" acceptance criteria
# ---------------------------------------------------------------------------


class TestRoute:
    def test_route_is_registered(self):
        paths = {route.path for route in app.routes}
        assert "/learning-path" in paths

    def test_post_returns_200_and_valid_path(self, client_with_generator):
        client, install = client_with_generator
        install(FakeGenerator(responses=[valid_response(["quran-101", "tajweed-201"])]))
        resp = client.post("/learning-path", json=request_payload())
        assert resp.status_code == 200
        body = resp.json()
        # Schema-valid LearningPath.
        LearningPath.model_validate(body)
        assert [s["course_id"] for s in body["steps"]] == ["quran-101", "tajweed-201"]

    def test_every_recommended_id_is_in_catalog(self, client_with_generator):
        client, install = client_with_generator
        install(FakeGenerator(responses=[valid_response(["quran-101", "tajweed-201"])]))
        payload = request_payload()
        resp = client.post("/learning-path", json=payload)
        assert resp.status_code == 200
        catalog_ids = {c["course_id"] for c in payload["catalog"]}
        step_ids = {s["course_id"] for s in resp.json()["steps"]}
        assert step_ids <= catalog_ids

    def test_adversarial_id_not_passed_through(self, client_with_generator):
        client, install = client_with_generator
        mixed = json.dumps(
            {
                "steps": [make_step("ghost-999", 1), make_step("quran-101", 2)],
                "summary": "s",
                "scholarly_note": "n",
            }
        )
        install(FakeGenerator(responses=[mixed]))
        resp = client.post("/learning-path", json=request_payload())
        assert resp.status_code == 200
        step_ids = {s["course_id"] for s in resp.json()["steps"]}
        assert "ghost-999" not in step_ids

    def test_persistent_ungrounded_returns_502(self, client_with_generator):
        client, install = client_with_generator
        install(FakeGenerator(responses=[valid_response(["ghost-999"])] * 3))
        resp = client.post("/learning-path", json=request_payload())
        assert resp.status_code == 502

    def test_empty_catalog_returns_422_before_generation(self, client_with_generator):
        client, install = client_with_generator
        gen = FakeGenerator(responses=[valid_response(["quran-101"])])
        install(gen)
        resp = client.post("/learning-path", json=request_payload(catalog=[]))
        assert resp.status_code == 422
        assert gen.call_count == 0

    def test_oversized_notes_returns_422_before_generation(self, client_with_generator):
        client, install = client_with_generator
        gen = FakeGenerator(responses=[valid_response(["quran-101"])])
        install(gen)
        payload = request_payload()
        payload["profile"]["notes"] = "x" * (MAX_NOTES_LENGTH + 1)
        resp = client.post("/learning-path", json=payload)
        assert resp.status_code == 422
        assert gen.call_count == 0

    def test_use_fake_generator_env_seam(self, monkeypatch):
        """USE_FAKE_GENERATOR selects the offline generator without a real key."""
        monkeypatch.setenv("USE_FAKE_GENERATOR", "1")
        gen = get_generator()
        assert isinstance(gen, FakeGenerator)


# ---------------------------------------------------------------------------
# Sanity: completion threshold is a percentage
# ---------------------------------------------------------------------------


def test_completion_threshold_is_percentage():
    assert 0 < COMPLETION_THRESHOLD <= 100
