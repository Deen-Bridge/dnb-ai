"""Production contract tests for the memorization planner (#843, PROD-AUDIT AI-067).

Covers grounding and provenance, Arabic/English localization and fallback,
user-scoped isolation and reset, storage failures, and input limits. All
identities are synthetic; no network, model or secret is needed.
"""

from __future__ import annotations

import hashlib
import logging
import re
import string
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import main
import memorization_planner as planner
from errors import APIException
from memorization_i18n import (
    MESSAGE_CATALOG_VERSION,
    SUPPORTED_LOCALES,
    catalog_keys,
    catalog_template,
    message,
    resolve_locale,
)
from memorization_planner import (
    PLAN_ID_PATTERN,
    PLANNER_RULES_VERSION,
    AdjustPlanRequest,
    CheckMilestoneRequest,
    GeneratePlanRequest,
    MemorizationPlannerEngine,
)
from memorization_store import (
    InMemoryPlannerStore,
    PlannerRecordCorrupt,
    PlannerStore,
    PlannerStoreUnavailable,
    RedisPlannerStore,
    set_planner_store,
)

API_KEY = "synthetic-test-key"
SERVICE_LOGGERS = ("memorization_planner", "memorization_store")
HEADERS = {"X-API-Key": API_KEY}
ROOT = Path(__file__).resolve().parent.parent
ARABIC = re.compile(r"[؀-ۿ]")
MOJIBAKE = re.compile(r"[ØÙ]|ðŸ|ï·")
START = "2026-11-23"

JUZ_30_PLAN: dict[str, Any] = {
    "goal": {"goal_type": "juz", "target_juz": [30]},
    "capacity": {"daily_available_minutes": 30, "days_per_week": 6, "pace_level": "moderate"},
    "start_date": START,
}
SMALL_PLAN: dict[str, Any] = {
    "goal": {"goal_type": "specific_surahs", "target_surahs": [112]},
    "capacity": {"daily_available_minutes": 30, "days_per_week": 6, "pace_level": "moderate"},
    "start_date": START,
}


def _service_log(caplog: pytest.LogCaptureFixture) -> str:
    """Log text from the planner and store only. HTTP access logs include the
    request path (and so the user_id) by design, like /memory/{user_id}."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name in SERVICE_LOGGERS)


@pytest.fixture
def store() -> Iterator[InMemoryPlannerStore]:
    fresh = InMemoryPlannerStore()
    set_planner_store(fresh)
    yield fresh
    set_planner_store(None)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, store: InMemoryPlannerStore) -> TestClient:
    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SERVICE_API_KEY", API_KEY)
    return TestClient(main.app)


def _plan(**overrides: Any) -> GeneratePlanRequest:
    return GeneratePlanRequest(**{**JUZ_30_PLAN, "user_id": "synthetic-user", **overrides})


def _create(client: TestClient, user_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = client.post(f"/memorization/users/{user_id}/plans", json=body or JUZ_30_PLAN, headers=HEADERS)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Grounding in versioned reference data
# ---------------------------------------------------------------------------


class TestGrounding:
    def test_embedded_table_matches_versioned_surah_index(self):
        assert planner.surah_reference_mismatches() == []

    def test_reference_hash_is_the_index_file_hash(self):
        expected = hashlib.sha256((ROOT / "data" / "quran" / "surah_index.json").read_bytes()).hexdigest()
        assert expected == planner.SURAH_INDEX_SHA256

    def test_arabic_surah_names_are_real_arabic(self):
        names = [planner.get_surah_info(n).arabic_name for n in range(1, 115)]
        assert all(ARABIC.search(name) for name in names)
        assert not any(MOJIBAKE.search(name) for name in names)
        assert planner.get_surah_info(1).arabic_name == "الفاتحة"
        assert planner.get_surah_info(114).arabic_name == "الناس"

    def test_planner_source_has_no_mis_encoded_text(self):
        for module in ("memorization_planner.py", "memorization_i18n.py"):
            assert not MOJIBAKE.search((ROOT / module).read_text(encoding="utf-8")), module

    def test_missing_index_is_reported_not_crashing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(planner, "SURAH_INDEX_FILE", ROOT / "data" / "quran" / "does-not-exist.json")
        assert planner._load_surah_reference() == (None, {})
        monkeypatch.setattr(planner, "_SURAH_REFERENCE", {})
        assert planner.surah_reference_mismatches() == ["surah_index.json unavailable"]

    def test_plan_provenance_and_limitations(self):
        plan = MemorizationPlannerEngine.generate_plan(_plan())
        prov = plan.provenance
        assert prov.planner_rules_version == PLANNER_RULES_VERSION
        assert prov.message_catalog_version == MESSAGE_CATALOG_VERSION
        assert prov.surah_reference.path == "data/quran/surah_index.json"
        assert prov.surah_reference.sha256 == planner.SURAH_INDEX_SHA256
        assert (prov.surah_reference.surah_count, prov.surah_reference.total_ayahs) == (114, 6236)
        assert prov.citations == [
            "Sahih al-Bukhari 5027",
            "Sahih Muslim 798",
            "Jami` at-Tirmidhi 2910",
            "Sunan Abi Dawud 1464",
        ]
        assert any("not a religious ruling" in item for item in plan.limitations)
        assert any("estimates" in item for item in plan.limitations)

    def test_steady_adjustment_has_no_citation(self):
        res = MemorizationPlannerEngine.adjust_plan(AdjustPlanRequest(plan_id="p", completed_ayahs_count=5))
        assert res.provenance.citations == []
        assert res.limitations


# ---------------------------------------------------------------------------
# Locale resolution and catalog
# ---------------------------------------------------------------------------


class TestLocale:
    @pytest.mark.parametrize(
        ("tag", "resolved", "requested", "reason"),
        [
            (None, "en", None, "missing"),
            ("", "en", None, "missing"),
            ("   ", "en", None, "missing"),
            ("en", "en", "en", None),
            ("EN-us", "en", "en-us", None),
            ("ar", "ar", "ar", None),
            ("ar-EG", "ar", "ar-EG", None),
            ("AR_sa", "ar", "ar-sa", None),
            (" ar ", "ar", "ar", None),
            ("fr", "en", "fr", "unsupported"),
            ("zh-Hant-TW", "en", "zh-Hant-TW", "unsupported"),
            ("123", "en", None, "malformed"),
            ("en--US", "en", None, "malformed"),
            ("e", "en", None, "malformed"),
            ("ar;DROP", "en", None, "malformed"),
            ("a" * 40, "en", None, "malformed"),
        ],
    )
    def test_resolution(self, tag, resolved, requested, reason):
        res = resolve_locale(tag)
        assert (res.locale, res.requested, res.fallback_reason) == (resolved, requested, reason)
        assert res.fallback is (reason is not None)
        assert res.direction == ("rtl" if resolved == "ar" else "ltr")

    def test_catalogs_have_identical_keys_and_placeholders(self):
        assert catalog_keys("en") == catalog_keys("ar")
        formatter = string.Formatter()
        for key in catalog_keys("en"):
            fields = {
                locale: {name for _, name, _, _ in formatter.parse(catalog_template(locale, key)) if name}
                for locale in SUPPORTED_LOCALES
            }
            assert fields["en"] == fields["ar"], key

    def test_arabic_catalog_is_arabic_and_quotes_no_hadith_text(self):
        for key in catalog_keys("ar"):
            text = catalog_template("ar", key)
            assert ARABIC.search(text), key
            # No Latin text: English hadith quotations stay out of the Arabic catalog.
            assert not re.search(r"[A-Za-z]", re.sub(r"\{[a-z_]+\}", "", text)), key

    def test_unknown_locale_uses_english(self):
        assert message("xx", "tip.review") == message("en", "tip.review")

    def test_arabic_plan_changes_text_only(self):
        en = MemorizationPlannerEngine.generate_plan(_plan(locale="en"))
        ar = MemorizationPlannerEngine.generate_plan(_plan(locale="ar-EG"))
        assert ar.locale.model_dump() == {
            "requested": "ar-EG",
            "resolved": "ar",
            "direction": "rtl",
            "fallback": False,
            "fallback_reason": None,
        }
        # Same schedule, dates, numbers and citations; only user-facing text differs.
        for field in ("total_target_ayahs", "daily_capacity_ayahs", "estimated_completion_date", "daily_time_split"):
            assert getattr(en, field) == getattr(ar, field)
        assert [m.citation for m in en.milestones] == [m.citation for m in ar.milestones]
        assert [m.estimated_date for m in en.milestones] == [m.estimated_date for m in ar.milestones]
        assert ARABIC.search(ar.goal_summary) and not ARABIC.search(en.goal_summary)
        first_day = ar.weekly_schedule[0]
        assert first_day.sabaq_new.surah_name == "An-Nas"  # canonical transliteration unchanged
        assert first_day.sabaq_new.surah_arabic_name == "الناس"
        assert "سورة الناس" in first_day.focal_tip
        assert all(ARABIC.search(item) for item in ar.limitations)

    def test_same_input_is_deterministic(self):
        first = MemorizationPlannerEngine.generate_plan(_plan(locale="ar")).model_dump()
        second = MemorizationPlannerEngine.generate_plan(_plan(locale="ar")).model_dump()
        for volatile in ("plan_id", "created_at"):
            first.pop(volatile)
            second.pop(volatile)
        assert first == second

    @pytest.mark.parametrize(("tag", "reason"), [(None, "missing"), ("fr-FR", "unsupported"), ("!!", "malformed")])
    def test_fallback_is_english_and_reported(self, tag, reason):
        plan = MemorizationPlannerEngine.generate_plan(_plan(locale=tag))
        assert plan.locale.resolved == "en"
        assert plan.locale.fallback_reason == reason
        assert plan.goal_summary.startswith("Personalized schedule")

    def test_arabic_adjustment_keeps_machine_codes_canonical(self):
        res = MemorizationPlannerEngine.adjust_plan(
            AdjustPlanRequest(plan_id="p", completed_ayahs_count=10, days_missed=4, locale="ar")
        )
        assert res.retention_health == "Needs Consolidation"
        assert res.retention_health_label == "يحتاج إلى تثبيت"
        assert res.status == "adjusted"
        assert all(ARABIC.search(step) for step in res.immediate_next_steps)
        assert res.provenance.citations == ["Sahih Muslim 798"]

    def test_arabic_celebration_uses_arabic_surah_name_and_canonical_citation(self):
        en = MemorizationPlannerEngine.check_milestone(
            CheckMilestoneRequest(plan_id="p", current_memorized_ayahs=4, recently_finished_surah=112)
        )
        ar = MemorizationPlannerEngine.check_milestone(
            CheckMilestoneRequest(plan_id="p", current_memorized_ayahs=4, recently_finished_surah=112, locale="ar")
        )
        assert "الإخلاص" in ar.milestone_title
        assert ar.authentic_citation == en.authentic_citation
        assert ar.badge.startswith("🏆")

    def test_arabic_reminder(self):
        progress = [planner.AyahProgress(surah=1, ayah=2, next_review=date.today() - timedelta(days=2))]
        res = planner.MurajaahEngine.reminder(progress, locale="ar")
        assert res.locale is not None and res.locale.resolved == "ar"
        assert res.message is not None and ARABIC.search(res.message) and "1:2" in res.message


# ---------------------------------------------------------------------------
# Input limits and malformed requests
# ---------------------------------------------------------------------------


class TestInputContract:
    def test_plan_id_is_opaque_and_unique(self):
        ids = {MemorizationPlannerEngine.generate_plan(_plan(user_id="synthetic-user-7")).plan_id for _ in range(20)}
        assert len(ids) == 20
        assert all(re.fullmatch(PLAN_ID_PATTERN, plan_id) for plan_id in ids)
        assert not any("synthetic-user-7" in plan_id for plan_id in ids)

    @pytest.mark.parametrize("user_id", ["has space", "../etc", "a/b", "x" * 129, ""])
    def test_rejects_malformed_user_ids(self, user_id):
        with pytest.raises(ValueError):
            _plan(user_id=user_id)

    def test_rejects_oversized_locale(self):
        with pytest.raises(ValueError):
            _plan(locale="a" * 65)

    def test_rejects_too_many_sessions(self):
        sessions = [{"date_str": "2026-11-01", "completed": True}] * 61
        with pytest.raises(ValueError):
            AdjustPlanRequest(plan_id="p", completed_ayahs_count=1, recent_sessions=sessions)

    def test_rejects_impossible_progress(self):
        with pytest.raises(ValueError):
            CheckMilestoneRequest(plan_id="p", current_memorized_ayahs=6237)

    @pytest.mark.parametrize("locale", ["en", "ar"])
    def test_completed_goal_is_a_clear_400(self, locale):
        juz_30 = [s.number for s in planner.get_juz_surahs(30)]
        req = _plan(locale=locale, goal={"goal_type": "juz", "target_juz": [30], "already_memorized_surahs": juz_30})
        with pytest.raises(APIException) as exc:
            MemorizationPlannerEngine.generate_plan(req)
        assert exc.value.status_code == 400

    def test_inverted_custom_range_is_a_clear_400(self):
        req = _plan(goal={"goal_type": "custom_range", "custom_start": [10, 1], "custom_end": [2, 1]})
        with pytest.raises(APIException) as exc:
            MemorizationPlannerEngine.generate_plan(req)
        assert exc.value.status_code == 400

    def test_invalid_start_date_is_a_clear_400(self):
        with pytest.raises(APIException) as exc:
            MemorizationPlannerEngine.generate_plan(_plan(start_date="23/11/2026"))
        assert exc.value.status_code == 400

    def test_revision_record_rejects_ayah_beyond_surah(self):
        with pytest.raises(ValueError):
            planner.RevisionRecordRequest(surah=1, ayah=8, quality=4)

    def test_boundary_full_quran_one_day_per_week(self):
        plan = MemorizationPlannerEngine.generate_plan(
            _plan(goal={"goal_type": "full_quran"}, capacity={"days_per_week": 1, "daily_available_minutes": 10})
        )
        assert plan.total_target_ayahs == 6236
        assert sum(1 for day in plan.weekly_schedule if day.is_rest_day) == 6


# ---------------------------------------------------------------------------
# User-bound API: authentication, isolation, reset
# ---------------------------------------------------------------------------


class TestUserBoundApi:
    USER_ROUTES = [
        ("post", "/memorization/users/synthetic-a/plans", JUZ_30_PLAN),
        ("get", "/memorization/users/synthetic-a/plans", None),
        ("post", "/memorization/users/synthetic-a/plans/plan_x/adjust", {"completed_ayahs_count": 1}),
        ("post", "/memorization/users/synthetic-a/plans/plan_x/milestone-check", {"current_memorized_ayahs": 1}),
        ("post", "/memorization/users/synthetic-a/revision/record", {"surah": 1, "ayah": 1, "quality": 4}),
        ("post", "/memorization/users/synthetic-a/revision/session", {"max_minutes": 5}),
        ("get", "/memorization/users/synthetic-a/revision/reminder", None),
        ("get", "/memorization/users/synthetic-a/revision/stats", None),
        ("delete", "/memorization/users/synthetic-a", None),
    ]

    @pytest.mark.parametrize(("method", "path", "body"), USER_ROUTES)
    @pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
    def test_user_routes_require_the_service_key(self, client, store, method, path, body, headers):
        kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 401
        assert len(store) == 0

    def test_stateless_routes_stay_public(self, client):
        assert client.get("/memorization/surahs").status_code == 200
        assert client.post("/memorization/plan", json={**JUZ_30_PLAN, "user_id": "anon"}).status_code == 200

    def test_create_and_list_plans(self, client):
        created = _create(client, "synthetic-a", {**JUZ_30_PLAN, "locale": "ar"})
        assert created["user_id"] == "synthetic-a"
        assert re.fullmatch(PLAN_ID_PATTERN, created["plan_id"])
        listed = client.get("/memorization/users/synthetic-a/plans", headers=HEADERS).json()
        assert [p["plan_id"] for p in listed["plans"]] == [created["plan_id"]]
        assert listed["plans"][0]["locale"] == "ar"
        assert listed["plans"][0]["planner_rules_version"] == PLANNER_RULES_VERSION

    def test_unknown_user_has_no_plans(self, client):
        assert client.get("/memorization/users/nobody/plans", headers=HEADERS).json() == {
            "user_id": "nobody",
            "plans": [],
        }

    def test_plan_count_is_bounded_newest_first(self, client):
        ids = [_create(client, "synthetic-a")["plan_id"] for _ in range(planner.MAX_PLANS_PER_USER + 1)]
        listed = client.get("/memorization/users/synthetic-a/plans", headers=HEADERS).json()["plans"]
        assert [p["plan_id"] for p in listed] == list(reversed(ids))[: planner.MAX_PLANS_PER_USER]

    def test_other_users_plan_is_indistinguishable_from_missing(self, client):
        plan_id = _create(client, "synthetic-a")["plan_id"]
        foreign = client.post(
            f"/memorization/users/synthetic-b/plans/{plan_id}/adjust",
            json={"completed_ayahs_count": 5},
            headers=HEADERS,
        )
        missing = client.post(
            "/memorization/users/synthetic-b/plans/plan_000000000000000000000000/adjust",
            json={"completed_ayahs_count": 5},
            headers=HEADERS,
        )
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json()
        assert foreign.json()["code"] == "memorization_plan_not_found"
        assert "synthetic-a" not in foreign.text
        milestone = client.post(
            f"/memorization/users/synthetic-b/plans/{plan_id}/milestone-check",
            json={"current_memorized_ayahs": 5},
            headers=HEADERS,
        )
        assert milestone.status_code == 404

    def test_bound_adjustment_uses_the_stored_plan_and_is_repeatable(self, client):
        created = _create(client, "synthetic-a")
        url = f"/memorization/users/synthetic-a/plans/{created['plan_id']}/adjust"
        body = {"completed_ayahs_count": 100, "locale": "en"}
        first = client.post(url, json=body, headers=HEADERS).json()
        second = client.post(url, json=body, headers=HEADERS).json()
        assert first == second
        assert first["plan_bound"] is True
        assert first["plan_id"] == created["plan_id"]
        assert first["recommended_pace_ayahs"] == created["daily_capacity_ayahs"]
        listed = client.get("/memorization/users/synthetic-a/plans", headers=HEADERS).json()["plans"][0]
        assert listed["completed_ayahs"] == 100

    def test_bound_adjustment_clamps_progress_beyond_goal(self, client):
        created = _create(client, "synthetic-a", SMALL_PLAN)
        res = client.post(
            f"/memorization/users/synthetic-a/plans/{created['plan_id']}/adjust",
            json={"completed_ayahs_count": 50},
            headers=HEADERS,
        ).json()
        assert res["new_estimated_completion_date"] == date.today().isoformat()
        listed = client.get("/memorization/users/synthetic-a/plans", headers=HEADERS).json()["plans"][0]
        assert listed["completed_ayahs"] == 4

    def test_bound_milestone_uses_the_plan_size(self, client):
        plan_id = _create(client, "synthetic-a", SMALL_PLAN)["plan_id"]
        bound = client.post(
            f"/memorization/users/synthetic-a/plans/{plan_id}/milestone-check",
            json={"current_memorized_ayahs": 4},
            headers=HEADERS,
        ).json()
        unbound = client.post(
            "/memorization/milestone-check", json={"plan_id": plan_id, "current_memorized_ayahs": 4}
        ).json()
        assert (bound["plan_bound"], bound["goal_total_ayahs"], bound["achieved"]) == (True, 4, True)
        assert (unbound["plan_bound"], unbound["goal_total_ayahs"], unbound["achieved"]) == (False, 560, False)

    def test_revision_progress_is_per_user(self, client):
        record = client.post(
            "/memorization/users/synthetic-a/revision/record",
            json={"surah": 1, "ayah": 1, "quality": 1},
            headers=HEADERS,
        )
        assert record.status_code == 200
        assert record.json()["lapses"] == 1
        stats_a = client.get("/memorization/users/synthetic-a/revision/stats", headers=HEADERS).json()
        stats_b = client.get("/memorization/users/synthetic-b/revision/stats", headers=HEADERS).json()
        assert stats_a["total_ayahs"] == 1 and stats_a["total_lapses"] == 1
        assert stats_b["total_ayahs"] == 0
        session_b = client.post(
            "/memorization/users/synthetic-b/revision/session", json={"max_minutes": 5}, headers=HEADERS
        ).json()
        assert session_b["ayahs"] == []

    def test_recording_the_same_ayah_updates_one_entry(self, client):
        for quality in (4, 5):
            client.post(
                "/memorization/users/synthetic-a/revision/record",
                json={"surah": 112, "ayah": 1, "quality": quality},
                headers=HEADERS,
            )
        stats = client.get("/memorization/users/synthetic-a/revision/stats", headers=HEADERS).json()
        assert stats["total_ayahs"] == 1

    def test_reminder_is_localized_for_the_user(self, client, store):
        client.post(
            "/memorization/users/synthetic-a/revision/record",
            json={"surah": 1, "ayah": 1, "quality": 0},
            headers=HEADERS,
        )
        # Make the ayah due today without waiting a day.
        record = store._records["memorization:user:synthetic-a"]
        store._records["memorization:user:synthetic-a"] = record.replace(
            (date.today() + timedelta(days=1)).isoformat(), date.today().isoformat()
        )
        res = client.get(
            "/memorization/users/synthetic-a/revision/reminder", params={"locale": "ar"}, headers=HEADERS
        ).json()
        assert res["due_today"] == 1
        assert ARABIC.search(res["message"])
        assert res["locale"]["direction"] == "rtl"

    def test_reset_removes_only_that_users_data(self, client):
        _create(client, "synthetic-a")
        _create(client, "synthetic-b")
        client.post(
            "/memorization/users/synthetic-a/revision/record",
            json={"surah": 1, "ayah": 1, "quality": 4},
            headers=HEADERS,
        )
        reset = client.delete("/memorization/users/synthetic-a", headers=HEADERS)
        assert reset.status_code == 200
        assert reset.json() == {
            "user_id": "synthetic-a",
            "plans_deleted": 1,
            "progress_entries_deleted": 1,
            "message": "Planner data deleted.",
        }
        assert client.get("/memorization/users/synthetic-a/plans", headers=HEADERS).json()["plans"] == []
        assert client.get("/memorization/users/synthetic-a/revision/stats", headers=HEADERS).json()["total_ayahs"] == 0
        assert len(client.get("/memorization/users/synthetic-b/plans", headers=HEADERS).json()["plans"]) == 1

    def test_reset_is_idempotent(self, client):
        first = client.delete("/memorization/users/synthetic-a", headers=HEADERS).json()
        assert first["message"] == "No planner data was stored for this user."
        assert (first["plans_deleted"], first["progress_entries_deleted"]) == (0, 0)

    def test_malformed_path_user_id_is_rejected(self, client):
        resp = client.get("/memorization/users/has%20space/plans", headers=HEADERS)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Storage failures
# ---------------------------------------------------------------------------


class _UnavailableStore(PlannerStore):
    backend = "broken"

    async def load(self, user_id: str) -> dict[str, Any] | None:
        raise PlannerStoreUnavailable("planner storage read failed")

    async def save(self, user_id: str, record: dict[str, Any]) -> None:
        raise PlannerStoreUnavailable("planner storage write failed")

    async def delete(self, user_id: str) -> bool:
        raise PlannerStoreUnavailable("planner storage delete failed")


class _CorruptStore(InMemoryPlannerStore):
    async def load(self, user_id: str) -> dict[str, Any] | None:
        return {"plans": "not-a-list", "progress": 7}


class TestStorageFailures:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", "/memorization/users/secret-user-1/plans", JUZ_30_PLAN),
            ("get", "/memorization/users/secret-user-1/plans", None),
            ("get", "/memorization/users/secret-user-1/revision/stats", None),
            ("delete", "/memorization/users/secret-user-1", None),
        ],
    )
    def test_unavailable_store_is_a_safe_503(self, client, caplog, method, path, body):
        set_planner_store(_UnavailableStore())
        kwargs: dict[str, Any] = {"headers": HEADERS}
        if body is not None:
            kwargs["json"] = body
        with caplog.at_level(logging.INFO):
            resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 503
        assert resp.json()["code"] == "memorization_storage_unavailable"
        assert "secret-user-1" not in resp.text
        assert "secret-user-1" not in _service_log(caplog)

    def test_corrupt_record_fails_closed_and_reset_recovers(self, client):
        set_planner_store(_CorruptStore())
        resp = client.get("/memorization/users/synthetic-a/plans", headers=HEADERS)
        assert resp.status_code == 500
        assert resp.json()["code"] == "memorization_record_unreadable"
        reset = client.delete("/memorization/users/synthetic-a", headers=HEADERS).json()
        assert reset["plans_deleted"] is None and reset["progress_entries_deleted"] is None

    def test_logs_use_a_hash_not_the_user_id(self, client, caplog):
        with caplog.at_level(logging.INFO):
            _create(client, "secret-user-2")
            client.delete("/memorization/users/secret-user-2", headers=HEADERS)
        service_log = _service_log(caplog)
        assert "Memorization plan stored" in service_log
        assert "planner data reset" in service_log
        assert "secret-user-2" not in service_log


# ---------------------------------------------------------------------------
# Store backends
# ---------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    async def get(self, key: str) -> str | None:
        if self.fail:
            raise ConnectionError("down")
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        if self.fail:
            raise ConnectionError("down")
        self.data[key], self.ttls[key] = value, ttl

    async def delete(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("down")
        return 1 if self.data.pop(key, None) is not None else 0


class TestStores:
    @pytest.mark.asyncio
    async def test_redis_round_trip_with_ttl(self):
        fake = _FakeRedis()
        redis_store = RedisPlannerStore(client=fake, ttl_seconds=123)
        await redis_store.save("synthetic-a", {"plans": [], "progress": [], "note": "الناس"})
        assert fake.ttls["memorization:user:synthetic-a"] == 123
        assert await redis_store.load("synthetic-a") == {"plans": [], "progress": [], "note": "الناس"}
        assert await redis_store.load("synthetic-b") is None
        assert await redis_store.delete("synthetic-a") is True
        assert await redis_store.delete("synthetic-a") is False

    @pytest.mark.asyncio
    async def test_redis_errors_become_unavailable(self):
        redis_store = RedisPlannerStore(client=_FakeRedis(fail=True))
        for call in (redis_store.load("u"), redis_store.save("u", {}), redis_store.delete("u")):
            with pytest.raises(PlannerStoreUnavailable):
                await call

    @pytest.mark.asyncio
    async def test_redis_corrupt_json_is_reported(self):
        fake = _FakeRedis()
        fake.data["memorization:user:u"] = "{not json"
        with pytest.raises(PlannerRecordCorrupt):
            await RedisPlannerStore(client=fake).load("u")

    @pytest.mark.asyncio
    async def test_in_memory_store_is_bounded_lru(self):
        bounded = InMemoryPlannerStore(max_users=2)
        await bounded.save("a", {"n": 1})
        await bounded.save("b", {"n": 2})
        await bounded.load("a")  # touch "a" so "b" is least recently used
        await bounded.save("c", {"n": 3})
        assert len(bounded) == 2
        assert await bounded.load("b") is None
        assert await bounded.load("a") == {"n": 1}

    @pytest.mark.asyncio
    async def test_in_memory_store_does_not_share_objects(self):
        isolated = InMemoryPlannerStore()
        record: dict[str, Any] = {"plans": []}
        await isolated.save("a", record)
        record["plans"].append("mutated")
        assert await isolated.load("a") == {"plans": []}
