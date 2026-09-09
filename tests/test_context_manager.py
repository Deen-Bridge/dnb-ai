"""Tests for the session-based context manager — offline, in-memory.

Covers session lifecycle, turn tracking, follow-up detection, madhhab
persistence, topic continuity, knowledge-level adaptation, and API schemas.
"""

import time

import pytest

from context_manager import (
    ContextManager,
    ConversationTurn,
    FollowUpIntent,
    ResolvedContext,
    SessionContext,
    UserProfile,
    _sessions,
    _user_profiles,
    detect_follow_up,
    extract_topic,
    get_context_manager,
    resolve_complexity,
)


@pytest.fixture(autouse=True)
def _clean_stores():
    """Reset module-level stores between tests."""
    _user_profiles.clear()
    _sessions.clear()
    yield
    _user_profiles.clear()
    _sessions.clear()


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


class TestExtractTopic:
    def test_faqh_keywords(self):
        assert extract_topic("What breaks wudu?") == "fiqh"

    def test_hadith_keywords(self):
        assert extract_topic("Tell me about a hadith from Sahih Bukhari") == "hadith"

    def test_quran_keywords(self):
        assert extract_topic("What does Surah Al-Baqarah say?") == "quran"

    def test_finance_keywords(self):
        # "zakat" appears in both fiqh and finance; fiqh wins on hit count
        # Use a finance-only phrase to verify the finance category.
        assert extract_topic("How much sadaqah should I give on stocks?") == "finance"

    def test_no_match_returns_none(self):
        assert extract_topic("Hello there") is None

    def test_multiple_categories_picks_most_hits(self):
        text = "What is the hadith about prayer and salah?"
        topic = extract_topic(text)
        # Both fiqh and hadith may match; whichever has more hits wins.
        assert topic in ("fiqh", "hadith")

    def test_case_insensitive(self):
        assert extract_topic("WHAT IS WUDU?") == "fiqh"


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------


class TestFollowUpDetection:
    def _make_turn(self, role: str, content: str, topic: str | None = None) -> ConversationTurn:
        return ConversationTurn(role=role, content=content, topic=topic)

    def test_pronoun_reference_is_follow_up(self):
        turns = [self._make_turn("user", "What is the ruling on zakat?")]
        result = detect_follow_up("Is it obligatory?", turns, ["fiqh"])
        assert result.is_follow_up is True
        assert "it" in result.implicit_references

    def test_topic_continuation_is_follow_up(self):
        turns = [self._make_turn("user", "What is wudu?", topic="fiqh")]
        # "and if" matches the extension pattern, but topic is also continued
        result = detect_follow_up("What about ghusl?", turns, ["fiqh"])
        assert result.is_follow_up is True
        assert result.referenced_topic == "fiqh"

    def test_elaboration_request(self):
        turns = []
        result = detect_follow_up("Why is that the case?", turns, [])
        assert result.is_follow_up is True
        assert result.intent == "elaboration"

    def test_extension_request(self):
        turns = []
        result = detect_follow_up("What about fasting in Ramadan?", turns, [])
        assert result.is_follow_up is True
        assert result.intent == "extension"

    def test_not_follow_up_for_independent_question(self):
        turns = [self._make_turn("user", "What is wudu?", topic="fiqh")]
        result = detect_follow_up("Tell me about the Prophet Muhammad's seerah", turns, ["fiqh"])
        # Independent question — different topic, no pronouns, no follow-up phrases
        assert result.is_follow_up is False

    def test_deictic_phrase_detection(self):
        turns = [self._make_turn("user", "What is the ruling onriba?")]
        result = detect_follow_up("Can you explain that verse?", turns, ["fiqh"])
        assert result.is_follow_up is True
        assert any("that" in ref for ref in result.implicit_references)

    def test_earlier_reference(self):
        turns = [self._make_turn("user", "What is halal meat?")]
        result = detect_follow_up("Earlier you mentioned something about slaughter", turns, ["fiqh"])
        assert result.is_follow_up is True


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_create_session(self):
        mgr = ContextManager()
        ctx = mgr.create_session(user_id="u1")
        assert ctx.session_id
        assert ctx.user_id == "u1"
        assert ctx.turns == []

    def test_get_session(self):
        mgr = ContextManager()
        ctx = mgr.create_session()
        fetched = mgr.get_session(ctx.session_id)
        assert fetched is not None
        assert fetched.session_id == ctx.session_id

    def test_get_missing_session_returns_none(self):
        mgr = ContextManager()
        assert mgr.get_session("nonexistent") is None

    def test_session_expiry(self, monkeypatch):
        from unittest.mock import MagicMock

        mgr = ContextManager()
        ctx = mgr.create_session()
        # Simulate old activity
        ctx.last_activity = time.time() - 999999
        _sessions[ctx.session_id] = ctx
        fake_settings = MagicMock()
        fake_settings.context_session_ttl_hours = 0
        fake_settings.context_max_turns = 50
        monkeypatch.setattr("config.get_settings", lambda: fake_settings)
        assert mgr.get_session(ctx.session_id) is None
        assert ctx.session_id not in _sessions

    def test_add_turn(self):
        mgr = ContextManager()
        ctx = mgr.create_session()
        turn = mgr.add_turn(ctx.session_id, "user", "What is zakat?")
        assert turn is not None
        assert turn.role == "user"
        assert turn.content == "What is zakat?"
        assert turn.topic == "fiqh"

    def test_add_turn_to_missing_session(self):
        mgr = ContextManager()
        assert mgr.add_turn("nope", "user", "hello") is None

    def test_topic_tracking(self):
        mgr = ContextManager()
        ctx = mgr.create_session()
        mgr.add_turn(ctx.session_id, "user", "What is zakat?")
        mgr.add_turn(ctx.session_id, "model", "Zakat is...")
        mgr.add_turn(ctx.session_id, "user", "What about hajj?")
        context = mgr.get_context(ctx.session_id)
        assert context is not None
        assert context.topic_continuity == "fiqh"
        # Both zakat and hajj are fiqh keywords, so topic_history is ["fiqh"]
        assert "fiqh" in context.topic_history

    def test_turn_trimming(self, monkeypatch):
        mgr = ContextManager()
        ctx = mgr.create_session()
        # Monkeypatch get_settings to return a config with small max_turns
        from unittest.mock import MagicMock

        fake_settings = MagicMock()
        fake_settings.context_max_turns = 3
        fake_settings.context_session_ttl_hours = 24
        monkeypatch.setattr("config.get_settings", lambda: fake_settings)
        for i in range(5):
            mgr.add_turn(ctx.session_id, "user", f"Turn {i} about wudu")
        assert len(ctx.turns) == 3
        assert ctx.turns[0].content == "Turn 2 about wudu"


# ---------------------------------------------------------------------------
# User profile persistence
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_update_profile_creates(self):
        mgr = ContextManager()
        profile = mgr.update_profile("u1", madhhab="hanafi", knowledge_level="beginner")
        assert profile.user_id == "u1"
        assert profile.madhhab == "hanafi"
        assert profile.knowledge_level == "beginner"

    def test_update_profile_merges(self):
        mgr = ContextManager()
        mgr.update_profile("u1", madhhab="hanafi")
        mgr.update_profile("u1", location="Dubai", knowledge_level="advanced")
        profile = mgr.get_profile("u1")
        assert profile is not None
        assert profile.madhhab == "hanafi"
        assert profile.location == "Dubai"
        assert profile.knowledge_level == "advanced"

    def test_madhhab_normalization(self):
        mgr = ContextManager()
        profile = mgr.update_profile("u1", madhhab="Shafi'i")
        assert profile.madhhab == "shafii"

    def test_invalid_madhhab_rejected(self):
        mgr = ContextManager()
        profile = mgr.update_profile("u1", madhhab="totally_fake")
        # Should not be set
        assert profile.madhhab is None

    def test_invalid_knowledge_level_rejected(self):
        mgr = ContextManager()
        profile = mgr.update_profile("u1", knowledge_level="expert")
        assert profile.knowledge_level is None

    def test_valid_knowledge_levels(self):
        mgr = ContextManager()
        for level in ("beginner", "intermediate", "advanced"):
            profile = mgr.update_profile("u1", knowledge_level=level)
            assert profile.knowledge_level == level

    def test_get_profile_missing(self):
        mgr = ContextManager()
        assert mgr.get_profile("nobody") is None


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------


class TestContextResolution:
    def test_resolve_context_basic(self):
        mgr = ContextManager()
        ctx = mgr.create_session(user_id="u1")
        mgr.update_profile("u1", madhhab="hanafi", knowledge_level="beginner")
        mgr.add_turn(ctx.session_id, "user", "What is wudu?")
        mgr.add_turn(ctx.session_id, "model", "Wudu is...")
        result = mgr.resolve_context(ctx.session_id, "Is it obligatory?")
        assert result is not None
        assert result.profile is not None
        assert result.profile.madhhab == "hanafi"
        assert result.topic == "fiqh"
        assert result.follow_up_intent.is_follow_up is True
        assert result.complexity_setting.level == "simple"
        assert result.turn_count == 2

    def test_resolve_context_missing_session(self):
        mgr = ContextManager()
        assert mgr.resolve_context("nope", "hello") is None

    def test_complexity_default(self):
        mgr = ContextManager()
        ctx = mgr.create_session()
        result = mgr.resolve_context(ctx.session_id, "Hello")
        assert result is not None
        assert result.complexity_setting.level == "comprehensive"

    def test_geographic_location_persisted(self):
        mgr = ContextManager()
        mgr.update_profile("u1", location="Riyadh, Saudi Arabia")
        profile = mgr.get_profile("u1")
        assert profile is not None
        assert profile.location == "Riyadh, Saudi Arabia"

    def test_cultural_context_persisted(self):
        mgr = ContextManager()
        mgr.update_profile("u1", cultural_context="South Asian")
        profile = mgr.get_profile("u1")
        assert profile is not None
        assert profile.cultural_context == "South Asian"


# ---------------------------------------------------------------------------
# Complexity settings
# ---------------------------------------------------------------------------


class TestComplexitySettings:
    def test_beginner_maps_to_simple(self):
        profile = UserProfile(user_id="u1", knowledge_level="beginner")
        setting = resolve_complexity(profile)
        assert setting.level == "simple"

    def test_intermediate_maps_to_comprehensive(self):
        profile = UserProfile(user_id="u1", knowledge_level="intermediate")
        setting = resolve_complexity(profile)
        assert setting.level == "comprehensive"

    def test_advanced_maps_to_scholarly(self):
        profile = UserProfile(user_id="u1", knowledge_level="advanced")
        setting = resolve_complexity(profile)
        assert setting.level == "scholarly"

    def test_no_profile_defaults_to_comprehensive(self):
        setting = resolve_complexity(None)
        assert setting.level == "comprehensive"

    def test_unknown_level_defaults_to_comprehensive(self):
        profile = UserProfile(user_id="u1", knowledge_level="expert")
        setting = resolve_complexity(profile)
        assert setting.level == "comprehensive"


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_user_profile_schema(self):
        p = UserProfile(user_id="u1")
        d = p.model_dump()
        assert d["user_id"] == "u1"
        assert d["madhhab"] is None
        assert d["preferences"] == {}

    def test_conversation_turn_schema(self):
        t = ConversationTurn(role="user", content="Hello")
        d = t.model_dump()
        assert d["role"] == "user"
        assert d["topic"] is None

    def test_session_context_schema(self):
        s = SessionContext(session_id="abc")
        d = s.model_dump()
        assert d["session_id"] == "abc"
        assert d["turns"] == []

    def test_resolved_context_schema(self):
        r = ResolvedContext(session_id="x")
        d = r.model_dump()
        assert d["session_id"] == "x"
        assert d["follow_up_intent"]["is_follow_up"] is False

    def test_follow_up_intent_defaults(self):
        f = FollowUpIntent()
        assert f.is_follow_up is False
        assert f.intent is None
        assert f.implicit_references == []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_context_manager_returns_same_instance(self):
        a = get_context_manager()
        b = get_context_manager()
        assert a is b


# ---------------------------------------------------------------------------
# Session persistence across turns (integration-style)
# ---------------------------------------------------------------------------


class TestSessionIntegration:
    def test_multi_turn_session_with_madhhab(self):
        mgr = ContextManager()
        ctx = mgr.create_session(user_id="u2")
        mgr.update_profile("u2", madhhab="maliki", knowledge_level="intermediate")

        mgr.add_turn(ctx.session_id, "user", "What is the ruling on music?")
        mgr.add_turn(ctx.session_id, "model", "The Maliki school generally holds...")
        mgr.add_turn(ctx.session_id, "user", "What about singing nasheeds?")

        snapshot = mgr.get_context(ctx.session_id)
        assert snapshot is not None
        assert snapshot.profile is not None
        assert snapshot.profile.madhhab == "maliki"
        assert len(snapshot.recent_turns) == 3

        resolved = mgr.resolve_context(ctx.session_id, "Is it permissible to play drums?")
        assert resolved is not None
        assert resolved.profile.madhhab == "maliki"
        assert resolved.topic == "fiqh"
        assert resolved.follow_up_intent.is_follow_up is True
