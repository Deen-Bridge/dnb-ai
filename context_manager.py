"""Session-based context management — track user preferences, conversation
state, and follow-up intent across turns.

Why this exists
---------------
A user who follows the Hanafi madhhab and has been asking about zakat should not
have to re-state either fact on every turn.  This module stitches together a
per-user profile (madhhab, location, language proficiency, knowledge level,
cultural context) with per-session conversation history (turns, topic continuity,
implicit references) so downstream callers get a ready-to-use ``ResolvedContext``
without managing that state themselves.

Design constraints
------------------
- Pure Python, offline, in-memory dict storage (documented as swappable for DB).
- Zero new dependencies: pydantic is already available.
- Every public function is side-effect-free at import time.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config defaults — read from Settings at call time via get_context_manager()
# but exposed as module-level constants for callers that import them directly.
# ---------------------------------------------------------------------------

CONTEXT_MAX_TURNS: int = 50
CONTEXT_SESSION_TTL_HOURS: int = 24

# ---------------------------------------------------------------------------
# Knowledge-level → complexity mapping
# ---------------------------------------------------------------------------

COMPLEXITY_MAP: dict[str, str] = {
    "beginner": "simple",
    "intermediate": "comprehensive",
    "advanced": "scholarly",
}

# ---------------------------------------------------------------------------
# Topic classification heuristics
# ---------------------------------------------------------------------------

# Keyword → category, reused from fiqh.py / tafsir / hadith conventions.
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "fiqh": [
        "wudu",
        "wudhu",
        "ghusl",
        "tayammum",
        "salah",
        "salat",
        "prayer",
        "pray",
        "fasting",
        "zakat",
        "zakah",
        "hajj",
        "umrah",
        "halal",
        "haram",
        "makruh",
        "riba",
        "usury",
        "marriage",
        "nikah",
        "divorce",
        "talaq",
        "inheritance",
        "fatwa",
        "ruling",
        "permissible",
        "prohibited",
        "madhhab",
        "fiqh",
        "taharah",
        "najis",
    ],
    "hadith": [
        "hadith",
        "hadeeth",
        "narration",
        "sahih",
        "bukhari",
        "muslim",
        "tirmidhi",
        "abu dawud",
        "nasa'i",
        "ibn majah",
        "musplics",
        "sanad",
        "isnad",
        "matn",
    ],
    "quran": [
        "quran",
        "qur'an",
        "koran",
        "ayah",
        "surah",
        "verse",
        "recitation",
        "tilawah",
        "tajweed",
        "tafsir",
        "exegesis",
        "ibn kathir",
    ],
    "aqeedah": [
        "aqeedah",
        "aqidah",
        "creed",
        "belief",
        "iman",
        "tawhid",
        "shirk",
        "kufr",
        "munafiq",
    ],
    "seerah": [
        "seerah",
        "sirah",
        "prophet",
        "muhammad",
        "companions",
        "sahaba",
        "hijra",
        "mecca",
        "medina",
    ],
    "tasawwuf": [
        "tasawwuf",
        "sufism",
        "spiritual",
        "dhikr",
        "adab",
        "ikhlas",
        "tawakkul",
        "sabr",
        "shukr",
    ],
    "finance": [
        "zakat",
        "zakah",
        "sadaqah",
        "waqf",
        "business",
        "trade",
        "investment",
        "stock",
        "banking",
        "interest",
        "loan",
        "debt",
    ],
}

# Implicit reference patterns: pronouns / deictic phrases that signal follow-up.
_IMPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(it|this|that|these|those)\b", re.IGNORECASE),
    re.compile(r"\b(this ruling|that verse|the same (issue|matter|topic|question))\b", re.IGNORECASE),
    re.compile(r"\b(what you (said|mentioned|explained)|as you (said|mentioned))\b", re.IGNORECASE),
    re.compile(r"\b(earlier|before|previously|last time)\b", re.IGNORECASE),
    re.compile(r"\b(going back to|regarding what|about what)\b", re.IGNORECASE),
]

# Follow-up intent signal patterns.
_FOLLOWUP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(why|how come|explain|elaborate|clarify|expand)\b", re.IGNORECASE), "elaboration"),
    (re.compile(r"\b(what about|and (if|when|how)|also|additionally|furthermore)\b", re.IGNORECASE), "extension"),
    (
        re.compile(r"\b(so (you're|you are)|does that mean|in that case|so essentially)\b", re.IGNORECASE),
        "confirmation",
    ),
    (re.compile(r"\b(can you (go deeper|be more specific|give more detail))\b", re.IGNORECASE), "deepening"),
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Persistent user preferences, stored in-memory (swappable for DB)."""

    user_id: str
    madhhab: str | None = None
    location: str | None = None
    language_proficiency: str | None = None
    knowledge_level: str | None = None
    cultural_context: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class ConversationTurn(BaseModel):
    """A single turn in a session's conversation history."""

    role: str
    content: str
    topic: str | None = None
    timestamp: float = Field(default_factory=time.time)


class SessionContext(BaseModel):
    """Tracks per-session conversation state."""

    session_id: str
    user_id: str | None = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    current_topic: str | None = None
    topic_history: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)


class ContextSnapshot(BaseModel):
    """Read-only assembled view of a session's context for downstream use."""

    profile: UserProfile | None = None
    recent_turns: list[ConversationTurn] = Field(default_factory=list)
    topic_continuity: str | None = None
    topic_history: list[str] = Field(default_factory=list)
    relevance_notes: str = ""


class FollowUpIntent(BaseModel):
    """Detected follow-up characteristics of a new query."""

    is_follow_up: bool = False
    intent: str | None = None  # "elaboration", "extension", "confirmation", "deepening"
    implicit_references: list[str] = Field(default_factory=list)
    referenced_topic: str | None = None


class ComplexitySetting(BaseModel):
    """Response complexity derived from knowledge level and preferences."""

    level: str = "comprehensive"  # "simple", "comprehensive", "scholarly"
    knowledge_level: str | None = None


class ResolvedContext(BaseModel):
    """The final output of context resolution for a new query."""

    profile: UserProfile | None = None
    topic: str | None = None
    follow_up_intent: FollowUpIntent = Field(default_factory=FollowUpIntent)
    complexity_setting: ComplexitySetting = Field(default_factory=ComplexitySetting)
    session_id: str = ""
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------


def extract_topic(text: str) -> str | None:
    """Classify ``text`` into a topic category using keyword heuristics.

    Returns the category name with the most keyword hits, or ``None`` when no
    category matches.  Deterministic and side-effect-free.
    """
    lowered = text.lower()
    best_category: str | None = None
    best_count = 0
    for category, keywords in _TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lowered)
        if count > best_count:
            best_count = count
            best_category = category
    return best_category


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------


def detect_follow_up(
    new_query: str,
    recent_turns: list[ConversationTurn],
    topic_history: list[str],
) -> FollowUpIntent:
    """Compare ``new_query`` against recent turns for follow-up signals.

    Detection runs on three heuristics:
    1. **Implicit references** — pronouns / deictic phrases ("it", "this ruling").
    2. **Topic continuation** — the query's topic matches a recent topic.
    3. **Elaboration requests** — phrases like "why", "explain", "go deeper".
    """
    implicit_refs: list[str] = []
    for pattern in _IMPLICIT_PATTERNS:
        for match in pattern.finditer(new_query):
            term = match.group(0)
            if term not in implicit_refs:
                implicit_refs.append(term)

    new_topic = extract_topic(new_query)
    topic_continuation = new_topic is not None and topic_history and new_topic in topic_history[-3:]

    intent: str | None = None
    for pattern, label in _FOLLOWUP_PATTERNS:
        if pattern.search(new_query):
            intent = label
            break

    is_follow_up = bool(implicit_refs) or topic_continuation or intent is not None
    referenced_topic = new_topic if topic_continuation else (topic_history[-1] if topic_history else None)

    return FollowUpIntent(
        is_follow_up=is_follow_up,
        intent=intent or ("continuation" if topic_continuation else None),
        implicit_references=implicit_refs,
        referenced_topic=referenced_topic,
    )


# ---------------------------------------------------------------------------
# Complexity settings
# ---------------------------------------------------------------------------


def resolve_complexity(profile: UserProfile | None) -> ComplexitySetting:
    """Map a user profile's knowledge level to a response complexity setting."""
    if profile and profile.knowledge_level:
        level = COMPLEXITY_MAP.get(profile.knowledge_level, "comprehensive")
        return ComplexitySetting(level=level, knowledge_level=profile.knowledge_level)
    return ComplexitySetting(level="comprehensive", knowledge_level=None)


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# Documented as swappable for a real DB backend.
_user_profiles: dict[str, UserProfile] = {}
_sessions: dict[str, SessionContext] = {}


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Manages user profiles, session lifecycle, and context resolution.

    All state lives in module-level dicts.  Swap ``_user_profiles`` and
    ``_sessions`` for database handles to go persistent.
    """

    def create_session(self, user_id: str | None = None) -> SessionContext:
        """Create a new session and return its context."""
        session_id = str(uuid.uuid4())
        ctx = SessionContext(session_id=session_id, user_id=user_id)
        _sessions[session_id] = ctx
        return ctx

    def get_session(self, session_id: str) -> SessionContext | None:
        """Retrieve a session by id, or ``None`` if expired / missing."""
        ctx = _sessions.get(session_id)
        if ctx is None:
            return None
        from config import get_settings

        ttl_hours = getattr(get_settings(), "context_session_ttl_hours", CONTEXT_SESSION_TTL_HOURS)
        ttl_seconds = ttl_hours * 3600
        if time.time() - ctx.last_activity > ttl_seconds:
            del _sessions[session_id]
            return None
        return ctx

    def add_turn(self, session_id: str, role: str, content: str) -> ConversationTurn | None:
        """Append a turn to the session's conversation history."""
        ctx = self.get_session(session_id)
        if ctx is None:
            return None
        topic = extract_topic(content)
        turn = ConversationTurn(role=role, content=content, topic=topic)
        ctx.turns.append(turn)
        ctx.last_activity = time.time()

        if topic:
            ctx.current_topic = topic
            if not ctx.topic_history or ctx.topic_history[-1] != topic:
                ctx.topic_history.append(topic)

        # Trim to max turns.
        from config import get_settings

        max_turns = getattr(get_settings(), "context_max_turns", CONTEXT_MAX_TURNS)
        if len(ctx.turns) > max_turns:
            ctx.turns = ctx.turns[-max_turns:]
        if len(ctx.topic_history) > max_turns:
            ctx.topic_history = ctx.topic_history[-max_turns:]

        return turn

    def get_context(self, session_id: str) -> ContextSnapshot | None:
        """Assemble a read-only snapshot of a session's context."""
        ctx = self.get_session(session_id)
        if ctx is None:
            return None

        profile = _user_profiles.get(ctx.user_id) if ctx.user_id else None
        recent = ctx.turns[-10:] if ctx.turns else []

        # Build relevance notes from topic continuity.
        relevance_parts: list[str] = []
        if ctx.current_topic:
            relevance_parts.append(f"Current topic: {ctx.current_topic}")
        if len(ctx.topic_history) > 1:
            relevance_parts.append(f"Topic trajectory: {' → '.join(ctx.topic_history[-5:])}")

        return ContextSnapshot(
            profile=profile,
            recent_turns=recent,
            topic_continuity=ctx.current_topic,
            topic_history=list(ctx.topic_history),
            relevance_notes="; ".join(relevance_parts),
        )

    def update_profile(
        self,
        user_id: str,
        *,
        madhhab: str | None = None,
        location: str | None = None,
        language_proficiency: str | None = None,
        knowledge_level: str | None = None,
        cultural_context: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> UserProfile:
        """Create or update a user profile.  Returns the current profile."""
        profile = _user_profiles.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            _user_profiles[user_id] = profile

        if madhhab is not None:
            from fiqh import normalize_madhhab

            normalized = normalize_madhhab(madhhab)
            if normalized is not None:
                profile.madhhab = normalized
        if location is not None:
            profile.location = location
        if language_proficiency is not None:
            profile.language_proficiency = language_proficiency
        if knowledge_level is not None:
            if knowledge_level in ("beginner", "intermediate", "advanced"):
                profile.knowledge_level = knowledge_level
        if cultural_context is not None:
            profile.cultural_context = cultural_context
        if preferences is not None:
            profile.preferences.update(preferences)

        profile.updated_at = time.time()
        return profile

    def get_profile(self, user_id: str) -> UserProfile | None:
        """Retrieve a user profile, or ``None`` if none exists."""
        return _user_profiles.get(user_id)

    def resolve_context(self, session_id: str, new_query: str) -> ResolvedContext | None:
        """Resolve full context for a new query against a session.

        Combines profile, topic classification, follow-up detection, and
        complexity settings into a single ``ResolvedContext`` ready for the
        chat layer.
        """
        ctx = self.get_session(session_id)
        if ctx is None:
            return None

        profile = _user_profiles.get(ctx.user_id) if ctx.user_id else None
        topic = extract_topic(new_query)
        follow_up = detect_follow_up(new_query, ctx.turns, ctx.topic_history)
        complexity = resolve_complexity(profile)

        return ResolvedContext(
            profile=profile,
            topic=topic or ctx.current_topic,
            follow_up_intent=follow_up,
            complexity_setting=complexity,
            session_id=session_id,
            turn_count=len(ctx.turns),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: ContextManager | None = None


def get_context_manager() -> ContextManager:
    """Return the process-wide context manager, creating on first use."""
    global _manager
    if _manager is None:
        _manager = ContextManager()
    return _manager


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402

from config import get_settings  # noqa: E402
from errors import APIException  # noqa: E402

router = APIRouter(prefix="/context", tags=["context"])


class _ContextConfig:
    """Read feature flag and limits from settings / env at import time."""

    enabled: bool = True


def _check_enabled() -> None:
    settings = get_settings()
    if not getattr(settings, "enable_context_manager", True):
        raise APIException(
            status_code=503,
            detail="Context manager is disabled.",
            hint="Set ENABLE_CONTEXT_MANAGER=true to enable this feature.",
        )


class CreateSessionRequest(BaseModel):
    user_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str | None = None


class AddTurnRequest(BaseModel):
    session_id: str
    role: str = Field(..., pattern="^(user|model)$")
    content: str = Field(..., min_length=1, max_length=10000)


class AddTurnResponse(BaseModel):
    turn: ConversationTurn
    topic: str | None = None


class UpdateProfileRequest(BaseModel):
    user_id: str
    madhhab: str | None = None
    location: str | None = None
    language_proficiency: str | None = None
    knowledge_level: str | None = None
    cultural_context: str | None = None
    preferences: dict[str, Any] | None = None


@router.post("/session", response_model=CreateSessionResponse)
async def create_session_endpoint(body: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new context session, optionally linked to a user."""
    _check_enabled()
    mgr = get_context_manager()
    ctx = mgr.create_session(user_id=body.user_id)
    return CreateSessionResponse(session_id=ctx.session_id, user_id=ctx.user_id)


@router.post("/turn", response_model=AddTurnResponse)
async def add_turn_endpoint(body: AddTurnRequest) -> AddTurnResponse:
    """Append a conversation turn to an existing session."""
    _check_enabled()
    mgr = get_context_manager()
    turn = mgr.add_turn(body.session_id, body.role, body.content)
    if turn is None:
        raise APIException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found or expired.",
            hint="Create a new session via POST /context/session first.",
        )
    return AddTurnResponse(turn=turn, topic=turn.topic)


@router.get("/{session_id}", response_model=ContextSnapshot)
async def get_context_endpoint(session_id: str) -> ContextSnapshot:
    """Retrieve the assembled context snapshot for a session."""
    _check_enabled()
    mgr = get_context_manager()
    snapshot = mgr.get_context(session_id)
    if snapshot is None:
        raise APIException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired.",
            hint="Create a new session via POST /context/session first.",
        )
    return snapshot


@router.put("/profile/{user_id}", response_model=UserProfile)
async def update_profile_endpoint(user_id: str, body: UpdateProfileRequest) -> UserProfile:
    """Create or update a user's preference profile."""
    _check_enabled()
    mgr = get_context_manager()
    profile = mgr.update_profile(
        user_id,
        madhhab=body.madhhab,
        location=body.location,
        language_proficiency=body.language_proficiency,
        knowledge_level=body.knowledge_level,
        cultural_context=body.cultural_context,
        preferences=body.preferences,
    )
    return profile
