# ruff: noqa: E402
import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

# Must run before any module that reads os.getenv at import time (store,
# config, …) so local development reads .env the same way production reads
# real environment variables.
load_dotenv()

import google.generativeai as genai
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from google.api_core.exceptions import (
    DeadlineExceeded,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
)
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import telemetry
from arabic_dialect import router as arabic_dialect_router
from arabic_ocr import router as arabic_ocr_router
from audio_hadith import router as audio_hadith_router
from calligraphy import router as calligraphy_router
from citations import (
    CITATION_BLOCK_CONTEXT,
    Citation,
    CitationExtraction,
    CitationStreamFilter,
    extract_citations,
)
from confidence import (
    ConfidenceAssessment,
    ConfidenceBand,
    apply_policy,
    assess,
    build_signals,
    thresholds as confidence_thresholds,
)
from config import get_settings
from errors import APIException
from feedback import (
    COMMENT_MAX_CHARS,
    FEEDBACK_TAXONOMY,
    FeedbackRecord,
    env_int,
    rate_limiter,
    store as feedback_store,
)
from fiqh import (
    FIQH_IKHTILAF_CONTEXT,
    MADHHAB_LEAD_INSTRUCTION,
    FiqhInfo,
    classify_fiqh,
    normalize_madhhab,
)
from hadith import (
    HADITH_ADAB_CONTEXT,
    HadithReference,
    annotate as annotate_hadith,
    build_caution_note,
)
from hadith_context import router as hadith_context_router
from history import router as history_router
from memory import ChatSummary, UserProfile, create_memory_store, render_user_context
from memory.extraction import (
    MEMORY_EXTRACTION_ENABLED,
    apply_updates,
    extract_updates,
    merge_summaries,
    summarize_conversation_turns,
)
from misinformation_api import router as misinformation_router
from model_router import router as model_routing_router
from page_analysis import router as page_analysis_router
from query_optimizer import router as query_optimizer_router
from reasoning_chains import router as reasoning_router
from reformulation import router as reformulation_router
from review import enqueue_for_review, router as review_router
from review_store import get_review_store
from safety import InputGate, OutputCheck, SafetyPipeline, load_policy
from semantic_cache import (
    CHAT_CONTEXT_MAX_LENGTH,
    CHAT_PROMPT_MAX_LENGTH,
    CHAT_RATE_LIMIT_MAX,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
    SEMANTIC_CACHE_ENABLED,
    embed_text,
    get_cache,
    get_chat_exact_cache,
    get_token_quota_tracker,
    normalize_text,
)
from sentiment import router as sentiment_router
from stellar import (
    PurchaseContext,
    PurchaseInfo,
    PurchaseTransaction,
    ZakatContext,
    ZakatInfo,
    build_chat_purchase_context,
    build_chat_zakat_context,
    redact_secret_keys,
    router as stellar_router,
)
from store import create_session_store, dicts_to_contents, history_to_dicts
from study import router as study_router
from swahili import (
    analyze_swahili,
    router as swahili_router,
    swahili_response_enhancer,
)
from tafsir import (
    TafsirContext,
    TafsirInfo,
    build_chat_tafsir_context,
    router as tafsir_router,
    summarize_tafsir_context,
    tafsir_system_context,
)
from vocabulary import router as vocabulary_router

logger = logging.getLogger(__name__)

settings = get_settings()

GEMINI_API_KEY = settings.gemini_api_key

genai.configure(api_key=GEMINI_API_KEY)

EVIDENCE_VERIFICATION_ENABLED = os.getenv("EVIDENCE_VERIFICATION_ENABLED", "true").lower() not in {"0", "false", "off"}

app = FastAPI(title="DeenBridge AI API")

# --- Service API-key authentication ---
# A shared secret between the DeenBridge backend/frontend and this service.
# In production the key MUST be set; set AUTH_DISABLED=true to opt out locally.
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").lower() in {"1", "true", "yes"}

if not AUTH_DISABLED and not SERVICE_API_KEY:
    if os.getenv("ENVIRONMENT", "").lower() == "production":
        raise RuntimeError(
            "SERVICE_API_KEY must be set in production. Set AUTH_DISABLED=true to run without authentication locally."
        )
    logger.warning("SERVICE_API_KEY is not set; authenticated endpoints will reject all requests.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """Dependency that enforces X-API-Key on protected routes.

    Returns the validated key on success, or raises 401.
    """
    if AUTH_DISABLED:
        return ""
    if not api_key or not secrets.compare_digest(api_key, SERVICE_API_KEY):
        raise APIException(
            status_code=401,
            detail="Missing or invalid X-API-Key",
            hint=(
                "Provide a valid service API key in the 'X-API-Key' header (e.g., 'X-API-Key: <your-key>'). "
                "For local testing without authentication, set environment variable AUTH_DISABLED=true."
            ),
        )
    return api_key


# --- Per-client rate limiting ---
# Default: 20 requests/minute per API key (falls back to client IP).
# Render sits behind a proxy; use X-Forwarded-For for IP-based limiting.
# NOTE: in-memory storage is fine for single-instance Render.  A shared Redis
# backend is needed if multi-instance support is added (see session-persistence
# issue).


def _rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    retry_after = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "hint": f"Too many requests sent. Please wait {retry_after} seconds before retrying.",
        },
        headers={"Retry-After": str(retry_after)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    headers = getattr(exc, "headers", None) or {}
    hint = getattr(exc, "hint", None)

    if isinstance(exc.detail, dict):
        content = dict(exc.detail)
        if hint and "hint" not in content:
            content["hint"] = hint
    else:
        content = {"detail": exc.detail}
        if hint:
            content["hint"] = hint

    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(content), headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    hints = []
    for err in errors:
        loc = " -> ".join(str(part) for part in err.get("loc", []) if part != "body")
        msg = err.get("msg", "Invalid value")
        if loc:
            hints.append(f"Field '{loc}': {msg}")
        else:
            hints.append(msg)
    hint_str = "; ".join(hints) if hints else "Please check request parameters and schema."
    content = {
        "detail": errors,
        "hint": f"Validation failed ({hint_str}). Please provide valid input according to the API schema.",
    }
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(content),
    )


# Stellar integration: read-only zakat/balance features on the network
# the rest of the Deen Bridge platform settles on
app.include_router(stellar_router)
app.include_router(reasoning_router)
app.include_router(study_router)
# Religious sentiment analysis: reads the emotional/spiritual tone of a question
app.include_router(sentiment_router)
# Tafsir: grounded, attributed ayah explanations from named classical works
app.include_router(tafsir_router)
# Page analysis: layout understanding of scanned Islamic book pages
app.include_router(page_analysis_router)
# Calligraphy: deterministic style estimation for Arabic calligraphic hands
app.include_router(calligraphy_router)
# Scholar review: the human end of the abstention loop
app.include_router(review_router)
# Question reformulation: deterministic quality assessment + rewrite suggestions
app.include_router(reformulation_router)
# Contextual hadith interpretation: sharh, asbab al-wurud, and synthesis
app.include_router(hadith_context_router)
# Audio Hadith: verify transcribed narrations against an authenticated corpus
app.include_router(audio_hadith_router)
# Database query optimization: static anti-pattern analysis + runtime profiling
app.include_router(query_optimizer_router)
# Historical context: asbab al-nuzul, hadith circumstances, and fiqh development
app.include_router(history_router)
# Model routing: pick the optimal model per query by complexity, latency and cost
app.include_router(model_routing_router)
# Arabic OCR: manuscript digitization with calligraphy detection and diacritic preservation
app.include_router(arabic_ocr_router)
# Quranic vocabulary analysis: root extraction, frequency stats, search, and verse examples
app.include_router(vocabulary_router)
# Swahili language processing: Islamic terminology, loanword morphology, and East African context
app.include_router(swahili_router)
# Arabic dialect support: Egyptian/Gulf/Levantine identification, MSA
# normalization and dialectal terminology lexicon (#136)
app.include_router(arabic_dialect_router)
# Religious misinformation flagging: detection, correction, and blocking of misinformation
app.include_router(misinformation_router)

from adhkar import corpus as adhkar_corpus


class AdhkarRecommendRequest(BaseModel):
    category: str | None = None
    query: str | None = None


@app.post("/adhkar/recommend")
async def recommend_adhkar(body: AdhkarRecommendRequest) -> dict[str, Any]:
    matches = adhkar_corpus.search(category=body.category, query=body.query)
    message = (
        f"Found {len(matches)} authenticated supplication(s) matching your request."
        if matches
        else "No authenticated supplication found matching your request."
    )
    return {
        "matches": matches,
        "message": message,
    }


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class CitationVerificationResult(BaseModel):
    source: str  # "quran" | "hadith"
    surah: int | None = None
    ayah: int | None = None
    collection: str | None = None
    number: str | None = None
    status: str  # "verified" | "mismatch" | "unverified" | "not_quoted"
    reason: str | None = None


class EvidenceClaim(BaseModel):
    claim: str
    evidence: list[str] = Field(default_factory=list)
    source_type: str | None = None
    verification_status: str = "unverified"
    confidence: float | None = None
    reason: str | None = None
    alternative_evidence: list[str] = Field(default_factory=list)


class EvidenceVerificationResult(BaseModel):
    claims: list[EvidenceClaim] = Field(default_factory=list)
    overall_score: float = 0.0
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceVerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=30000)


class ChatRequest(BaseModel):
    prompt: str = Field(..., max_length=CHAT_PROMPT_MAX_LENGTH)
    chat_id: str | None = None
    context: str | None = Field(None, max_length=CHAT_CONTEXT_MAX_LENGTH)  # Additional context for specific queries
    madhhab: str | None = None  # User's madhhab: hanafi, maliki, shafii, hanbali
    language: str | None = None  # BCP-47 response language (ar, en, ur, etc.); auto-detect when omitted
    user_id: str | None = Field(default=None, max_length=128)  # Opaque user identifier for personalization
    remember: bool = True  # When False, existing memory is read but no new data persisted
    # Optional authenticated purchase context for Stellar payment questions.
    # Prefer a short structured summary from the signed-in frontend; otherwise
    # pass the user's JWT so this service can fetch history from dnb-backend.
    transactions: list[PurchaseTransaction] | None = None
    auth_token: str | None = None


class Message(BaseModel):
    role: str
    content: str
    message_id: str | None = None  # present on model turns, for feedback


class Moderation(BaseModel):
    category_id: str | None = None
    action: str


class ChatResponse(BaseModel):
    response: str | None = None
    text: str | None = None
    chat_id: str
    message_id: str | None = None  # stable id of the answer just returned
    history: list[Message] = []
    moderation: Moderation | None = None
    fiqh: FiqhInfo | None = None
    hadith_references: list[HadithReference] | None = None
    tafsir: TafsirInfo | None = None
    confidence: ConfidenceAssessment | None = None
    zakat: ZakatInfo | None = None
    purchases: PurchaseInfo | None = None
    language: str | None = None
    # Structured references parsed out of the answer (#15). Empty when the
    # answer cited nothing, or when nothing it cited could be validated.
    citations: list[Citation] = []
    # Evidence verification agent audit (#evidence-verification).
    evidence_verification: EvidenceVerificationResult | None = None


class FeedbackRequest(BaseModel):
    chat_id: str = Field(..., max_length=200)
    message_id: str = Field(..., max_length=200)
    rating: str = Field(..., description="'up' or 'down'")
    # At most one tag per taxonomy category; a caller cannot pad the list.
    categories: list[str] | None = Field(None, max_length=len(FEEDBACK_TAXONOMY))
    comment: str | None = Field(None, max_length=COMMENT_MAX_CHARS)
    # Supplied by the client when the snapshot is gone (restart). Bounded so an
    # anonymous caller cannot fill the store with multi-megabyte bodies — the
    # rate limiter caps request count, not request size.
    prompt: str | None = Field(None, max_length=8000)
    answer: str | None = Field(None, max_length=16000)

    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        return v

    @field_validator("categories")
    @classmethod
    def categories_must_be_valid(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        invalid = set(v) - FEEDBACK_TAXONOMY
        if invalid:
            raise ValueError(f"Unknown categories: {sorted(invalid)}. Valid choices: {sorted(FEEDBACK_TAXONOMY)}")
        return v


def classify_for_safety(prompt: str, candidate_ids: list[str]) -> dict[str, Any]:
    """Gemini classifier seam; offline tests replace this with a fixture."""
    classifier_instruction = (
        "Classify the user text for the Deen Bridge safety policy. Return only "
        "strict JSON with exactly category_id, confidence, and action. "
        "category_id must be one of the candidates or 'none'; action must be "
        "allow, allow_with_guidance, or refuse. Candidates: "
        f"{candidate_ids}"
    )
    model = genai.GenerativeModel(
        telemetry.GEMINI_MODEL,
        system_instruction=classifier_instruction,
    )
    _t0 = time.perf_counter()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
        request_options={"timeout": 30},
    )
    telemetry.record_model_call(
        response,
        telemetry.GEMINI_MODEL,
        (time.perf_counter() - _t0) * 1000.0,
        stage="classification",
    )
    return json.loads(response.text)


safety_policy = load_policy()
safety_pipeline = SafetyPipeline(InputGate(safety_policy, classify_for_safety), OutputCheck(safety_policy))

# Semantic response cache
semantic_cache = get_cache()
token_quota_tracker = get_token_quota_tracker()

# Durable queue for low-confidence religious answers awaiting a scholar
review_store = get_review_store()

# Per-user memory store (Redis-backed or in-memory)
memory_store = create_memory_store()

# Chat-history store: Firestore when a Firebase service account is configured,
# otherwise Redis, otherwise in-memory. Used by both the non-streaming and
# streaming chat endpoints, and by the history/list/delete endpoints below.
session_store = create_session_store()

MAX_CHAT_HISTORY_TURNS = 20

# Tafsir retrieval seam: returns None for prompts that are not
# verse-explanation questions. Offline tests replace this with a stub.
DEFAULT_TAFSIR_LANGUAGE = "en"


async def tafsir_retriever(prompt: str, language: str) -> TafsirContext | None:
    """Retrieve tafsir for a chat turn; never fail the turn over retrieval."""
    try:
        return await build_chat_tafsir_context(prompt, language)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("Tafsir retrieval failed; answering without it: %s", exc)
        return None


async def zakat_retriever(prompt: str, context: str | None) -> ZakatContext | None:
    """Compute zakat for a chat turn; never fail the turn over the lookup."""
    try:
        return await build_chat_zakat_context(prompt, context)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("Zakat lookup failed; answering without it: %s", exc)
        return None


async def purchase_retriever(
    prompt: str,
    transactions: list[PurchaseTransaction] | None,
    auth_token: str | None,
) -> PurchaseContext | None:
    """Load purchase metadata for a chat turn; never fail the turn over it."""
    try:
        return await build_chat_purchase_context(prompt, transactions=transactions, auth_token=auth_token)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("Purchase lookup failed; answering without it: %s", exc)
        return None


def get_safety_settings() -> list[dict[str, str]]:
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]


# In-memory session store for demo purposes
sessions: dict[str, Any] = {}
active_chats: dict[str, Any] = {}

# --- Feedback support ------------------------------------------------------
# The generation config captured into a feedback record so a flagged answer is
# reproducible evidence, kept beside the model name telemetry already tracks.
GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Stable ids for model answers, so the frontend can address one turn and the
# feedback endpoint can locate what was said. Kept parallel to active_chats
# rather than restructuring it: chat_id -> [message_id per model turn, in order].
chat_message_ids: dict[str, list[str]] = {}

# Faithful snapshot of what the user was actually shown for each answered turn,
# keyed by (chat_id, message_id). Feedback reads from here so the stored answer
# is the displayed text (post safety/hadith/abstention shaping), not the raw
# model output. Bounded LRU — on a free-tier restart it is empty, which is why
# the feedback endpoint also accepts a client-supplied prompt/answer.
FEEDBACK_SNAPSHOT_MAX = env_int("FEEDBACK_SNAPSHOT_MAX", 5000)
answer_snapshots: "OrderedDict[tuple, dict[str, str]]" = OrderedDict()

# Only honor X-Forwarded-For when the deployment actually sits behind a proxy we
# control; otherwise any client can rotate the header to mint a fresh rate-limit
# bucket on every request and defeat the only control on the write endpoint.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() in {
    "1",
    "true",
    "yes",
}


def _record_answer(chat_id: str, prompt: str, answer: str) -> str:
    """Assign a message id to a fresh answer, snapshot it, and return the id."""
    message_id = str(uuid.uuid4())
    chat_message_ids.setdefault(chat_id, []).append(message_id)
    answer_snapshots[(chat_id, message_id)] = {"prompt": prompt, "answer": answer}
    while len(answer_snapshots) > FEEDBACK_SNAPSHOT_MAX:
        answer_snapshots.popitem(last=False)
    return message_id


def _tag_history_with_message_ids(chat_id: str, history: list["Message"]) -> None:
    """Attach each model turn's stable id to the history returned to the client."""
    ids = chat_message_ids.get(chat_id, [])
    model_turn = 0
    for message in history:
        if message.role == "model":
            if model_turn < len(ids):
                message.message_id = ids[model_turn]
            model_turn += 1


# --- Admin auth (stopgap until real auth/rate-limiting infrastructure) ------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def require_admin(token: str | None = Depends(_admin_header)) -> None:
    """Gate admin endpoints on ADMIN_TOKEN; closed by default when unset."""
    if not ADMIN_TOKEN:
        raise APIException(
            status_code=503,
            detail="ADMIN_TOKEN is not configured on this server.",
            hint="Set the ADMIN_TOKEN environment variable in server configuration to enable admin management routes.",
        )
    # Compare as bytes: secrets.compare_digest raises on non-ASCII str, which
    # would turn a crafted header into a 500 instead of a clean 403.
    if not token or not secrets.compare_digest(token.encode("utf-8"), ADMIN_TOKEN.encode("utf-8")):
        raise APIException(
            status_code=403,
            detail="Invalid or missing admin token.",
            hint="Include the configured admin secret in the 'X-Admin-Token' request header (e.g., 'X-Admin-Token: <admin_token>').",
        )


ISLAMIC_CONTEXT = (
    "You are an AI assistant for Deen Bridge, a platform for authentic Islamic education. "
    "Provide respectful, accurate, and context-aware responses grounded in authentic Islamic knowledge.\n\n"
    "POLICY ON CITATIONS:\n"
    "- Cite sources when possible (Quran surah:ayah and authentic Hadith collections).\n"
    "- Ensure exact accuracy of surah/ayah numbers and quoted text.\n"
    "- If you cannot cite a verifiable source for a claim, state the point as general scholarly consensus or "
    "general knowledge—do NOT fabricate references.\n"
)

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "en": "English",
    "ur": "Urdu",
    "ms": "Malay",
    "fr": "French",
    "tr": "Turkish",
    "id": "Indonesian",
    "bn": "Bengali",
    "fa": "Persian",
    "ha": "Hausa",
    "sw": "Swahili",
    "tl": "Tagalog",
}

LANGUAGE_INSTRUCTIONS = (
    "\n\nLANGUAGE POLICY:\n"
    "- When a response_language code is provided, respond entirely in that language.\n"
    "- When no response_language is provided (auto mode), respond in the same language as the user's question.\n"
    "- ALWAYS quote Quran in the original Arabic script (e.g. بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ) "
    "followed by a translation in the response language, with the surah:ayah reference.\n"
    "- Use standard transliteration for core Islamic terms (e.g. salat, zakat, hajj, shahada) "
    "when writing in Latin-script languages.\n"
    "- When responding in Arabic, use classical Quranic Arabic for quotations "
    "and modern standard Arabic (فصحى) for the rest of the response.\n"
    "- When responding in Swahili (Kiswahili), use standard respectful Swahili (Kiswahili Sanifu) "
    "with proper Islamic honorifics (k.m. 'Mwenyezi Mungu (Subhanahu wa Ta'ala)', 'Mtume Muhammad (Swalla Allahu Alayhi wa Sallam / ﷺ)', "
    "'Maswahaba (Radhi Allahu Anhum)') and standard Swahili Islamic terminology (Swala, Udhu, Saumu, Zaka, Hija, Halali, Haramu, Kadhi).\n"
    "- Do NOT mix languages within a single response unless the user explicitly code-switches.\n"
)


def normalize_language(lang: str | None) -> str | None:
    """Validate a BCP-47 language code against SUPPORTED_LANGUAGES.

    Returns the lowercase code if valid, or None to signal auto-detection.
    An unrecognized code is not an error — it falls back to auto-detection
    so an unexpected locale degrades gracefully instead of failing with 422.
    """
    if not lang:
        return None
    code = lang.strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code
    base = code.split("-")[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    logger.warning("Unrecognized language code %r; falling back to auto-detection", lang)
    return None


def get_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(
        model_name=settings.model_name,
        system_instruction=ISLAMIC_CONTEXT,
    )


GEMINI_TIMEOUT = settings.gemini_timeout


def extract_text_safely(response: Any) -> str | None:
    """Safely extract text from Gemini response, handling safety blocks gracefully."""
    if not response:
        return None

    # Check candidates for finish reason / safety blocks
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            reason_name = getattr(finish_reason, "name", str(finish_reason)).upper()
            if reason_name in ("SAFETY", "BLOCKED", "PROMPT_FEEDBACK", "RECITATION", "SPII"):
                return None

    # Check prompt feedback
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        if block_reason:
            return None

    # Access text property safely (raises ValueError if response has no text/candidate)
    try:
        text = response.text
        if not text:
            return None
        return text
    except (ValueError, AttributeError):
        return None


async def send_message_with_retry(
    chat_session: Any,
    message: str,
    generation_config: dict[str, Any] | None = None,
    timeout: int = GEMINI_TIMEOUT,
    max_retries: int = 2,
) -> Any:
    """Send message asynchronously with retries for transient upstream errors.

    Preserves chat history integrity by cleaning up un-responded user messages
    if an upstream call fails.
    """
    attempt = 0
    while True:
        history_len_before = (
            len(chat_session.history) if hasattr(chat_session, "history") and chat_session.history is not None else 0
        )
        try:
            kwargs: dict[str, Any] = {"request_options": {"timeout": timeout}}
            if generation_config:
                kwargs["generation_config"] = generation_config
            response = await chat_session.send_message_async(
                message,
                **kwargs,
            )
            return response
        except (TimeoutError, ServiceUnavailable, DeadlineExceeded) as exc:
            if hasattr(chat_session, "history") and chat_session.history is not None:
                if len(chat_session.history) > history_len_before:
                    chat_session.history = chat_session.history[:history_len_before]

            attempt += 1
            if attempt > max_retries:
                logger.warning(
                    "Gemini send_message_async failed after %d retries: %s",
                    max_retries,
                    exc,
                )
                raise exc

            backoff = 0.5 * (2 ** (attempt - 1))
            logger.info(
                "Transient Gemini error (%s). Retrying in %.1fs (attempt %d/%d)...",
                exc,
                backoff,
                attempt,
                max_retries,
            )
            await asyncio.sleep(backoff)
        except Exception as exc:
            if hasattr(chat_session, "history") and chat_session.history is not None:
                if len(chat_session.history) > history_len_before:
                    chat_session.history = chat_session.history[:history_len_before]
            raise exc


async def run_strict_corrective_loop(
    chat_session: Any,
    user_message: str,
    original_text: str,
    mismatches: list[dict[str, Any]],
) -> str:
    """Run exactly one corrective regeneration when a citation mismatch occurs in strict mode."""
    corrections_text = []
    for m in mismatches:
        if m.get("source") == "quran" and "correct_text" in m:
            corrections_text.append(
                f"- Surah {m['surah']}:{m['ayah']} text in corpus is: '{m['correct_text']}'. Your quote did not match."
            )
        elif m.get("reason"):
            corrections_text.append(f"- {m['reason']}")

    correction_prompt = (
        "Your previous response had citation errors:\n"
        + "\n".join(corrections_text)
        + "\n\nPlease regenerate your response correcting the quotes/references, or remove any unverified references entirely."
    )

    corrective_response = await send_message_with_retry(chat_session, correction_prompt)
    safe_text = extract_text_safely(corrective_response)
    return safe_text or original_text


def _citation_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return {}


def _evidence_label(d: dict[str, Any]) -> str:
    if d.get("source") == "quran" and d.get("surah") is not None:
        return f"Quran {d['surah']}:{d.get('ayah', '')}"
    if d.get("collection"):
        return f"{d['collection']} {d.get('number', '')}".strip()
    return str(d.get("source") or "cited source")


def verify_evidence_claims(
    response_text: str,
    citation_extraction: CitationExtraction,
    hadith_refs: list[HadithReference],
) -> EvidenceVerificationResult:
    """Deterministic claim-evidence audit using already-verified citations."""
    result = EvidenceVerificationResult()
    claims = [
        s.strip()
        for s in response_text.replace("\n", " ").split(".")
        if len(s.strip()) >= 25
    ]
    evidence: list[tuple[str, str, bool]] = []

    for c in citation_extraction.citations:
        d = _citation_to_dict(c)
        label = _evidence_label(d)
        status = str(d.get("status") or d.get("verification") or "unverified")
        evidence.append((label, status, status in {"verified", "authentic", "sahih", "hasan", "mutawatir"}))

    for h in hadith_refs:
        d = _citation_to_dict(h)
        label = _evidence_label(d)
        grade = str(d.get("grade") or d.get("authenticity") or d.get("status") or "unverified").lower()
        evidence.append((label, grade, grade not in {"daif", "weak", "munkar", "fabricated", "mawdu", "unverified"}))

    for claim in claims:
        matched = [e for e in evidence if e[0].lower() in claim.lower()]
        if not matched and len(claims) == 1 and len(evidence) == 1:
            matched = evidence[:]

        if not matched:
            result.claims.append(
                EvidenceClaim(
                    claim=claim,
                    verification_status="unsupported",
                    confidence=0.1,
                    reason="No verifiable citation is present for this claim.",
                    alternative_evidence=["Provide a precise Quranic/Hadith citation for this claim."],
                )
            )
            result.audit_trail.append({"claim": claim, "status": "unsupported"})
            continue

        supported = any(e[2] for e in matched)
        status = "supported" if supported else "weak"
        result.claims.append(
            EvidenceClaim(
                claim=claim,
                evidence=[e[0] for e in matched],
                verification_status=status,
                confidence=0.9 if supported else 0.4,
                reason="Evidence linked and verified." if supported else "Evidence cited but could not be verified.",
            )
        )
        result.audit_trail.append({"claim": claim, "evidence": [e[0] for e in matched], "status": status})

    result.overall_score = (
        sum(1 for c in result.claims if c.verification_status == "supported") / len(result.claims)
        if result.claims
        else 0.0
    )
    return result


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{CHAT_RATE_LIMIT_MAX}/{CHAT_RATE_LIMIT_WINDOW_SECONDS} seconds")
async def chat(body: ChatRequest, request: Request, fastapi_response: Response) -> ChatResponse:
    trace = telemetry.Trace()
    _ctx_token = telemetry.current_trace.set(trace)
    _handler_start = time.perf_counter()
    _succeeded = False

    def _finalize() -> None:
        """Stamp content-free telemetry onto the response and record the request."""
        handler_ms = (time.perf_counter() - _handler_start) * 1000.0
        totals = trace.request_totals()
        fastapi_response.headers["X-Trace-Id"] = trace.trace_id
        fastapi_response.headers["X-LLM-Total-Tokens"] = str(totals["total_tokens"])
        fastapi_response.headers["X-LLM-Cost-USD"] = f"{totals['cost_usd']:.8f}"
        fastapi_response.headers["X-Handler-Latency-Ms"] = f"{handler_ms:.2f}"
        telemetry.registry.record_request(handler_ms, error=False)

    try:
        chat_id = body.chat_id or str(uuid.uuid4())
        is_new_chat = chat_id not in active_chats
        is_bypass = request.headers.get("X-Cache-Bypass") == "1"

        # A user who pastes a Stellar secret key must not have it forwarded to
        # the model provider or written into stored history. Everything
        # downstream works from the redacted text; the zakat layer separately
        # detects that one was present and warns the user.
        prompt = redact_secret_keys(body.prompt)
        extra_context = redact_secret_keys(body.context)
        logger.info(f"Received chat request: {prompt[:100]}...")

        # --- Fiqh/intent classification & madhhab ---
        with trace.span("classification"):
            madhhab = normalize_madhhab(body.madhhab)
            is_fiqh = classify_fiqh(prompt)
            fiqh_info = FiqhInfo(is_fiqh_question=is_fiqh, madhhab_requested=madhhab)
            effective_language = normalize_language(body.language)

            # --- Swahili analysis ---
            is_swahili = effective_language == "sw"
            swahili_analysis = None
            if is_swahili or any(
                w in prompt.lower()
                for w in [
                    "je,",
                    "habari",
                    "swala",
                    "udhu",
                    "saumu",
                    "zaka",
                    "hija",
                    "kadhi",
                    "bakwata",
                    "maulidi",
                    "kufunga",
                ]
            ):
                swahili_analysis = analyze_swahili(prompt)
                if not is_swahili and len(swahili_analysis.detected_terms) >= 2:
                    is_swahili = True
                    effective_language = "sw"

        # --- Tafsir and zakat retrieval (grouped as one telemetry stage) ---
        with trace.span("retrieval"):
            # Tafsir detection is offline (regex + the bundled surah index),
            # so a non-tafsir prompt costs nothing.
            tafsir_context = await tafsir_retriever(prompt, body.language or DEFAULT_TAFSIR_LANGUAGE)
            tafsir_info = summarize_tafsir_context(tafsir_context) if tafsir_context else None

            # Zakat detection is offline (keywords plus a key-shaped match), so
            # an ordinary prompt never touches Horizon or the gold-price API.
            zakat_context = await zakat_retriever(body.prompt, body.context)
            zakat_info = zakat_context.info if zakat_context else None

            # Purchase detection is offline (keywords). History comes from an
            # inline summary or a best-effort JWT fetch — never other users'.
            purchase_context = await purchase_retriever(body.prompt, body.transactions, body.auth_token)
            purchase_info = purchase_context.info if purchase_context else None

        # --- Memory lookup ---
        profile: UserProfile | None = None
        summary: ChatSummary | None = None
        if body.user_id:
            profile = await memory_store.get_profile(body.user_id)
            summary = await memory_store.get_chat_summary(f"{body.user_id}:{chat_id}")

        # Determine cache scope: public for anonymous, user:{user_id} for authenticated
        cache_scope = "public" if body.user_id is None else f"user:{body.user_id}"

        # Neither a tafsir-grounded answer nor a zakat/purchase answer goes
        # through the semantic response cache: the first is built from retrieved
        # passages (already cached by ayah key), and the others contain one
        # user's real financial data, which must never be replayed to anyone else.
        is_cacheable = (
            is_new_chat
            and body.context is None
            and tafsir_context is None
            and zakat_context is None
            and purchase_context is None
            and SEMANTIC_CACHE_ENABLED
        )

        # --- Two-tier cache lookup: exact-match first, then semantic ---
        exact_cache = get_chat_exact_cache()
        embedding: Any = None
        normalized: str | None = None

        if is_cacheable and not is_bypass:
            # Exact-match cache lookup (tier 1)
            exact_key = f"{cache_scope}:{normalize_text(prompt)}"
            exact_cached = exact_cache.get(exact_key)
            if exact_cached is not None:
                fastapi_response.headers["X-Cache-Tier"] = "exact"
                fastapi_response.headers["X-Semantic-Cache"] = "hit"
                model = genai.GenerativeModel(
                    telemetry.GEMINI_MODEL,
                    safety_settings=get_safety_settings(),
                )
                chat_session = model.start_chat(
                    history=[
                        {"role": "user", "parts": [{"text": prompt}]},
                        {"role": "model", "parts": [{"text": exact_cached["response"]}]},
                    ]
                )
                active_chats[chat_id] = chat_session
                logger.info("Exact cache HIT for prompt: %s", prompt[:80])
                cached_message_id = _record_answer(chat_id, prompt, exact_cached["response"])
                _finalize()
                _succeeded = True
                return ChatResponse(
                    response=exact_cached["response"],
                    chat_id=chat_id,
                    message_id=cached_message_id,
                    history=exact_cached["history"],
                    fiqh=fiqh_info,
                    hadith_references=annotate_hadith(exact_cached["response"]),
                    language=effective_language,
                )

            # Semantic cache lookup (tier 2)
            normalized = normalize_text(prompt)
            embedding = embed_text(normalized)
            cached = semantic_cache.get(embedding, scope=cache_scope)
            if cached is not None:
                fastapi_response.headers["X-Cache-Tier"] = "semantic"
                fastapi_response.headers["X-Semantic-Cache"] = "hit"
                model = genai.GenerativeModel(
                    telemetry.GEMINI_MODEL,
                    safety_settings=get_safety_settings(),
                )
                chat_session = model.start_chat(
                    history=[
                        {"role": "user", "parts": [{"text": prompt}]},
                        {"role": "model", "parts": [{"text": cached.response}]},
                    ]
                )
                active_chats[chat_id] = chat_session
                logger.info("Semantic cache HIT for prompt: %s", prompt[:80])
                cached_message_id = _record_answer(chat_id, prompt, cached.response)
                _finalize()
                _succeeded = True
                return ChatResponse(
                    response=cached.response,
                    chat_id=chat_id,
                    message_id=cached_message_id,
                    history=cached.history,
                    fiqh=fiqh_info,
                    hadith_references=annotate_hadith(cached.response),
                    language=effective_language,
                )
        elif is_bypass:
            semantic_cache.bypasses += 1

        # --- Token quota enforcement ---
        # Check quota before making any LLM call (cache miss path)
        quota_key = body.user_id if body.user_id else _rate_limit_key(request)
        # Estimate token count for quota check (conservative estimate)
        estimated_tokens = len(prompt.split()) + len(body.context.split()) if body.context else len(prompt.split())
        # Use a conservative multiplier for system context and response
        estimated_tokens = int(estimated_tokens * 3)  # Account for system prompt and response

        quota_allowed, retry_after = token_quota_tracker.is_allowed(quota_key, estimated_tokens)
        if not quota_allowed:
            logger.warning("Token quota exceeded for key %s: retry_after=%d", quota_key, retry_after)
            raise APIException(
                status_code=429,
                detail="Token quota exceeded. Please try again later.",
                hint=f"Hourly token quota limit reached. Please wait {retry_after} seconds before sending further messages, or reduce message length.",
                headers={"Retry-After": str(retry_after)},
            )

        # --- Normal flow (cache miss / bypass / not cacheable) ---
        async def generate(safety_prompt: str) -> str:
            if chat_id not in active_chats:
                logger.info(f"Creating new chat session: {chat_id}")
                model = get_model()
                # Load persisted history if available
                persisted = await session_store.load_history(chat_id)
                history = dicts_to_contents(persisted) if persisted else []
                active_chats[chat_id] = model.start_chat(history=history)

            system_context = ISLAMIC_CONTEXT + HADITH_ADAB_CONTEXT + CITATION_BLOCK_CONTEXT
            if is_fiqh:
                system_context += FIQH_IKHTILAF_CONTEXT
                if madhhab:
                    system_context += MADHHAB_LEAD_INSTRUCTION.format(madhhab=madhhab)
            if tafsir_context is not None:
                system_context += tafsir_system_context(tafsir_context)
            if zakat_context is not None:
                system_context += zakat_context.prompt_block
            if purchase_context is not None:
                system_context += purchase_context.prompt_block
            if is_swahili and swahili_analysis:
                sw_enhancement = swahili_response_enhancer.build_prompt_enhancement(safety_prompt)
                if sw_enhancement.cultural_notes:
                    system_context += "\n\nMuktadha wa Afrika Mashariki (East African Context):\n" + "\n".join(
                        f"- {note}" for note in sw_enhancement.cultural_notes
                    )
            memory_block = render_user_context(profile, summary)
            if memory_block:
                system_context += f"\n\n{memory_block}"
            context = f"Additional context: {extra_context}\n\n" if extra_context else ""
            full_prompt = f"{system_context}\n{context}User question: {safety_prompt}"
            logger.info("Sending message to chat...")
            _t0 = time.perf_counter()
            response = await send_message_with_retry(
                active_chats[chat_id],
                full_prompt,
                generation_config=GENERATION_CONFIG,
            )
            telemetry.record_model_call(
                response,
                telemetry.GEMINI_MODEL,
                (time.perf_counter() - _t0) * 1000.0,
                stage="generation",
                trace=trace,
            )
            text = extract_text_safely(response)
            if not text:
                raise APIException(
                    status_code=500,
                    detail="Empty response from AI model",
                    hint="The upstream AI model returned an empty response. Try rephrasing your prompt or asking again in a few moments.",
                )
            return text

        enabled = os.getenv("SAFETY_PIPELINE_ENABLED", "true").lower() not in {"0", "false", "off"}
        with trace.span("generation"):
            if enabled:
                safety_result = await safety_pipeline.run_async(prompt, generate)
            else:
                safety_result = None
                generated_text = await generate(prompt)

        # Everything from here (history extraction, hadith grading, confidence
        # assessment, and the scholar-review enqueue, which does I/O) is timed
        # as one post-processing stage so a slow tail is attributable.
        _pp_start = time.perf_counter()

        logger.info(
            "safety=%s",
            {
                "policy_id": safety_result.category_id if safety_result else None,
                "action": safety_result.action if safety_result else "disabled",
                "stages_fired": safety_result.stages_fired if safety_result else [],
                "latency_ms": safety_result.latency_ms if safety_result else 0,
            },
        )

        # Get chat history (strip system context from user messages)
        history = []
        chat_session = active_chats.get(chat_id)
        for message in chat_session.history if chat_session else []:
            try:
                if hasattr(message, "parts") and message.parts:
                    content = message.parts[0].text if hasattr(message.parts[0], "text") else str(message.parts[0])
                else:
                    content = str(message)

                if message.role == "user":
                    content = _strip_system_context(content)
                history.append(Message(role="user" if message.role == "user" else "model", content=content))
            except Exception as e:
                logger.warning(f"Error processing message in history: {str(e)}")
                continue

        response_text = safety_result.text if safety_result else generated_text

        # --- Structured citations (#15) ---
        # Parsed off before hadith annotation and before the cache write, so
        # the block never reaches the user and a cached answer replays exactly
        # the prose the original asker saw. Parsing is total: a malformed or
        # truncated block costs the citations, never the answer.
        response_text, citation_extraction = extract_citations(response_text)

        # --- Hadith authenticity grading ---
        # Baked into response_text *before* the cache write so a cached hit
        # replays the same caution the user originally saw.
        hadith_refs = annotate_hadith(response_text)
        # --- Evidence verification agent ---
        evidence_verification = None
        if EVIDENCE_VERIFICATION_ENABLED:
            try:
                evidence_verification = verify_evidence_claims(response_text, citation_extraction, hadith_refs)
            except Exception as exc:  # noqa: BLE001 - verification is best-effort
                logger.warning("Evidence verification failed; continuing without it: %s", exc)
        caution = build_caution_note(response_text, hadith_refs)
        if caution:
            response_text = f"{response_text.rstrip()}\n\n{caution}"

        # --- Confidence, abstention, and scholar escalation ---
        # is_religious and is_high_stakes reuse classification that already ran
        # this turn (the fiqh classifier and the hadith annotator) rather than
        # adding a competing classifier. self_consistency (#ai-18) is still
        # absent from the average until that component lands.
        # citation_verification is now supplied (#15): the share of this
        # answer's citations that validated against the surah index and the
        # hadith collection table. Being an EXTERNAL_SIGNAL it also lifts the
        # UNVERIFIED_CEILING that otherwise caps every answer at 0.65, so a
        # well-cited answer can reach the confident band for the first time.
        signals = build_signals(
            response_text,
            is_religious=is_fiqh or bool(hadith_refs),
            is_high_stakes=is_fiqh,
            citation_verification=citation_extraction.score,
        )
        assessment = assess(signals)
        answer_before_policy = response_text

        if assessment.queued:
            # Queue before shaping the reply, so the user is only told their
            # question reached a scholar if it actually did.
            try:
                item = await enqueue_for_review(
                    question=prompt,
                    answer=answer_before_policy,
                    score=assessment.score,
                    band=assessment.band.value,
                    signals=assessment.signals,
                    chat_id=chat_id,
                )
                assessment.review_id = item.id
            except Exception as exc:  # noqa: BLE001 - the answer still matters
                logger.error("Could not queue answer for scholar review: %s", exc)
                assessment.queued = False

        response_text = apply_policy(response_text, assessment)

        logger.info(
            "confidence=%s",
            {
                "score": assessment.score,
                "band": assessment.band.value,
                "signals": assessment.signals_used,
                "queued": assessment.queued,
            },
        )

        # --- Two-tier cache write ---
        # Only confident answers are cached. Replaying an abstention, or a
        # hedged answer whose warning would outlive the doubt that caused it,
        # would spread one turn's uncertainty to every later asker.
        is_cacheable = is_cacheable and assessment.band is ConfidenceBand.CONFIDENT
        if is_cacheable and (safety_result is None or safety_result.generator_called):
            # Get token count from telemetry for savings tracking
            totals = trace.request_totals()
            token_count = totals.get("total_tokens", 0)

            if embedding is None:
                normalized = normalize_text(prompt)
                embedding = embed_text(normalized)

            # Write to exact-match cache (tier 1)
            exact_key = f"{cache_scope}:{normalized}"
            exact_cache.put(
                exact_key,
                {"response": response_text, "history": history},
                token_count=token_count,
            )

            # Write to semantic cache (tier 2)
            semantic_cache.put(
                embedding,
                response_text,
                chat_id,
                history,
                scope=cache_scope,
                token_count=token_count,
            )
            logger.info("Two-tier cache WRITE for prompt: %s (scope: %s)", prompt[:80], cache_scope)

        fastapi_response.headers["X-Cache-Tier"] = "miss"
        fastapi_response.headers["X-Semantic-Cache"] = "bypass" if is_bypass else "miss"
        if swahili_analysis:
            fastapi_response.headers["X-Swahili-Dialect"] = swahili_analysis.dialect.primary_dialect.value
            fastapi_response.headers["X-Swahili-Terms-Detected"] = str(len(swahili_analysis.detected_terms))
            fastapi_response.headers["X-Swahili-Code-Switching"] = swahili_analysis.code_switch.switch_type.value

        # Assign this answer a stable id and snapshot the displayed text, so a
        # later /feedback call can reference exactly this turn and store what
        # the user actually saw (post safety/hadith/abstention shaping).
        message_id = _record_answer(chat_id, prompt, response_text)
        _tag_history_with_message_ids(chat_id, history)

        trace.add_span("post_processing", (time.perf_counter() - _pp_start) * 1000.0)
        logger.info("Chat response generated successfully")
        # Build the response before finalizing, so a construction/validation
        # failure is handled only by the error path and the request is not
        # counted as both a success (here) and an error (except block).
        response_obj = ChatResponse(
            response=response_text,
            chat_id=chat_id,
            message_id=message_id,
            history=history,
            moderation=Moderation(
                category_id=safety_result.category_id,
                action=safety_result.action,
            )
            if safety_result and safety_result.category_id
            else None,
            fiqh=fiqh_info,
            hadith_references=hadith_refs,
            tafsir=tafsir_info,
            confidence=assessment,
            zakat=zakat_info,
            purchases=purchase_info,
            language=effective_language,
            citations=citation_extraction.citations,
            evidence_verification=evidence_verification,
        )
        _finalize()
        _succeeded = True

        # --- Persist chat history ---
        asyncio.create_task(_persist_chat_history(chat_id, body.user_id, chat_session))

        # --- Background memory extraction and summarization ---
        # Runs as fire-and-forget tasks after the response is sent.
        if body.user_id and body.remember and MEMORY_EXTRACTION_ENABLED:
            asyncio.create_task(
                _extract_and_update_memory(
                    body.user_id,
                    prompt,
                    response_text,
                    chat_id,
                    summary,
                    memory_store,
                )
            )
            logger.info("Memory extraction scheduled for user %s", body.user_id[:8])

        # --- Summary eviction ---
        # After enough turns accumulate, summarize old history and persist.
        if body.user_id and body.remember and MEMORY_EXTRACTION_ENABLED:
            chat_session = active_chats.get(chat_id)
            if chat_session and hasattr(chat_session, "history") and chat_session.history:
                if len(chat_session.history) >= MAX_CHAT_HISTORY_TURNS:
                    asyncio.create_task(
                        _summarize_history(
                            f"{body.user_id}:{chat_id}",
                            chat_session.history,
                            summary,
                            memory_store,
                        )
                    )
                    logger.info("History summarization triggered for %s", body.user_id[:8])

        return response_obj

    except ResourceExhausted as exc:
        logger.warning("Gemini rate limit exceeded for chat %s: %s", chat_id, exc)
        raise APIException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            hint="Upstream Gemini API rate limit reached. Please wait 10-30 seconds before retrying.",
            headers={"X-Trace-Id": trace.trace_id},
        ) from exc
    except InvalidArgument as exc:
        logger.warning("Invalid argument for Gemini call in chat %s: %s", chat_id, exc)
        raise APIException(
            status_code=400,
            detail="Invalid request parameters.",
            hint="Verify request fields (e.g., prompt length must be 1-10000 characters, context must be <= 5000 characters).",
            headers={"X-Trace-Id": trace.trace_id},
        ) from exc
    except (TimeoutError, DeadlineExceeded) as exc:
        logger.warning("Gemini API call timed out for chat %s: %s", chat_id, exc)
        raise APIException(
            status_code=504,
            detail="AI service timed out.",
            hint="The AI provider did not respond within the deadline. Retry in a few seconds or try a shorter prompt.",
            headers={"X-Trace-Id": trace.trace_id},
        ) from exc
    except ServiceUnavailable as exc:
        logger.warning("Gemini service unavailable for chat %s: %s", chat_id, exc)
        raise APIException(
            status_code=503,
            detail="AI service temporarily unavailable.",
            hint="The AI service is temporarily experiencing high load or provider downtime. Please retry in 30-60 seconds.",
            headers={"X-Trace-Id": trace.trace_id},
        ) from exc
    except HTTPException as exc:
        # Attach the trace id to already-typed HTTP errors (e.g. the empty-response
        # 500 raised inside generate) so a failed request stays correlatable.
        exc.headers = {**(exc.headers or {}), "X-Trace-Id": trace.trace_id}
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /chat handler for session %s: %s", chat_id, exc)
        raise APIException(
            status_code=500,
            detail="AI service error",
            hint="An unexpected server error occurred while generating the answer. Please retry your request or contact support if the issue persists.",
            headers={"X-Trace-Id": trace.trace_id},
        ) from exc
    finally:
        # Record the request exactly once: the success path already recorded it
        # via _finalize(); anything that reached an except path is an error.
        if not _succeeded:
            telemetry.registry.record_request((time.perf_counter() - _handler_start) * 1000.0, error=True)
        telemetry.current_trace.reset(_ctx_token)


async def _extract_and_update_memory(
    user_id: str,
    prompt: str,
    response: str,
    chat_id: str,
    existing_summary: ChatSummary | None,
    store: Any,
) -> None:
    """Fire-and-forget memory extraction. Runs via asyncio.create_task."""
    try:
        updates = await extract_updates(prompt, response)
        if updates.get("none"):
            return
        profile = await store.get_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
        profile = apply_updates(profile, updates)
        await store.save_profile(user_id, profile)
        logger.debug("Memory updated for user %s", user_id[:8])
    except Exception:
        logger.warning("Memory extraction failed for user %s", user_id[:8], exc_info=True)


async def _summarize_history(
    chat_id: str,
    history: list,
    existing_summary: ChatSummary | None,
    store: Any,
) -> None:
    """Summarize accumulated conversation turns and persist."""
    try:
        turns = [{"role": m.role, "text": m.parts[0].text if m.parts else ""} for m in history]
        new_summary_text = await summarize_conversation_turns(turns)
        if existing_summary:
            merged = await merge_summaries(existing_summary.content, new_summary_text)
        else:
            merged = new_summary_text
        summary = ChatSummary(chat_id=chat_id, content=merged, turn_count=len(history))
        await store.save_chat_summary(chat_id, summary)
    except Exception:
        logger.warning("History summarization failed for %s", chat_id[:8], exc_info=True)


@app.post("/chat/stream")
@limiter.limit(f"{CHAT_RATE_LIMIT_MAX}/{CHAT_RATE_LIMIT_WINDOW_SECONDS} seconds")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Streaming chat endpoint using Server-Sent Events (SSE).

    Returns incremental ``data:`` events carrying text deltas as JSON
    (``{"type": "content", "delta": "..."}``), followed by a terminal
    ``{"type": "done", ...}`` event with chat_id, history, and metadata.
    Mid-stream upstream errors terminate the stream with an explicit error
    event rather than silently truncating.

    Session behaviour matches the non-streaming ``/chat`` endpoint:
    streamed turns end up in the same ``active_chats`` history, and a
    follow-up non-streamed message in the same ``chat_id`` sees the
    streamed answer as context.

    Accepts the same request schema as ``POST /chat``.
    """
    trace = telemetry.Trace()
    _ctx_token = telemetry.current_trace.set(trace)
    _handler_start = time.perf_counter()

    try:
        chat_id = body.chat_id or str(uuid.uuid4())
        prompt = redact_secret_keys(body.prompt)
        extra_context = redact_secret_keys(body.context)
        logger.info("Received streaming chat request: %s...", prompt[:100])

        # --- Fiqh/intent classification & madhhab ---
        with trace.span("classification"):
            madhhab = normalize_madhhab(body.madhhab)
            is_fiqh = classify_fiqh(prompt)
            fiqh_info = FiqhInfo(is_fiqh_question=is_fiqh, madhhab_requested=madhhab)
            effective_language = normalize_language(body.language)

        # --- Tafsir and zakat retrieval ---
        with trace.span("retrieval"):
            tafsir_context = await tafsir_retriever(prompt, body.language or DEFAULT_TAFSIR_LANGUAGE)
            tafsir_info = summarize_tafsir_context(tafsir_context) if tafsir_context else None
            zakat_context = await zakat_retriever(body.prompt, body.context)
            zakat_info = zakat_context.info if zakat_context else None

        combined_text: str = ""  # accumulated full response for post-processing
        chat_session = None

        safety_enabled = os.getenv("SAFETY_PIPELINE_ENABLED", "true").lower() not in {"0", "false", "off"}

        # --- Purchase history ---
        purchase_context = await purchase_retriever(body.prompt, body.transactions, body.auth_token)

        async def event_generator() -> AsyncGenerator[str, None]:
            """SSE generator: metadata → content deltas → done/error."""
            nonlocal combined_text, chat_session

            # Track the Gemini streaming response for disconnect handling
            stream_response: Any = None
            decision: Any = None
            hadith_refs: list[HadithReference] = []
            assessment: ConfidenceAssessment | None = None
            citation_extraction = CitationExtraction()

            try:
                # --- Metadata event ---
                meta = json.dumps({"type": "metadata", "chat_id": chat_id, "language": effective_language})
                yield f"data: {meta}\n\n"

                # --- Safety input gate ---
                if safety_enabled:
                    with trace.span("safety_input"):
                        decision = await safety_pipeline.input_gate.evaluate_async(prompt)

                    if decision.action == "refuse":
                        err = json.dumps(
                            {
                                "type": "error",
                                "message": "Your message was not processed due to content policy.",
                            }
                        )
                        yield f"data: {err}\n\n"
                        return

                    if decision.action == "allow_with_guidance":
                        generation_prompt = f"{decision.guidance}\n\nUser question: {prompt}"
                    else:
                        generation_prompt = prompt
                else:
                    generation_prompt = prompt

                # --- Create or get chat session ---
                if chat_id not in active_chats:
                    logger.info("Creating new streaming chat session: %s", chat_id)
                    # Resume from persisted history so a returning user (or a
                    # request that arrived after a restart) keeps the context.
                    persisted = await session_store.load_history(chat_id)
                    active_chats[chat_id] = get_model().start_chat(
                        history=dicts_to_contents(persisted) if persisted else []
                    )

                chat_session = active_chats[chat_id]

                # --- Build system context + prompt ---
                system_context = ISLAMIC_CONTEXT + HADITH_ADAB_CONTEXT + CITATION_BLOCK_CONTEXT
                if effective_language:
                    system_context += LANGUAGE_INSTRUCTIONS
                    system_context += f"\nresponse_language: {effective_language}"
                else:
                    system_context += LANGUAGE_INSTRUCTIONS
                    system_context += "\nresponse_language: auto (respond in the user's language)"
                if is_fiqh:
                    system_context += FIQH_IKHTILAF_CONTEXT
                    if madhhab:
                        system_context += MADHHAB_LEAD_INSTRUCTION.format(madhhab=madhhab)
                if tafsir_context is not None:
                    system_context += tafsir_system_context(tafsir_context)
                if zakat_context is not None:
                    system_context += zakat_context.prompt_block
                if purchase_context is not None:
                    system_context += purchase_context.prompt_block

                ctx = f"Additional context: {extra_context}\n\n" if extra_context else ""
                full_prompt = f"{system_context}\n{ctx}User question: {generation_prompt}"

                # --- Async streaming generation ---
                with trace.span("generation"):
                    logger.info("Starting async streaming response...")
                    _t0 = time.perf_counter()

                    stream_response = await active_chats[chat_id].send_message_async(
                        full_prompt,
                        generation_config={
                            "temperature": settings.temperature,
                            "top_p": settings.top_p,
                            "top_k": settings.top_k,
                            "max_output_tokens": settings.max_output_tokens,
                        },
                        stream=True,
                    )

                    # The citation block must never flash in a delta. The filter
                    # withholds any tail that could still turn out to be the
                    # start marker, including one split across two chunks.
                    citation_filter = CitationStreamFilter()
                    async for chunk in stream_response:
                        if chunk.text:
                            visible = citation_filter.feed(chunk.text)
                            if visible:
                                combined_text += visible
                                delta = json.dumps(
                                    {
                                        "type": "content",
                                        "delta": visible,
                                    }
                                )
                                yield f"data: {delta}\n\n"

                    trailing, citation_extraction = citation_filter.finish()
                    if trailing:
                        combined_text += trailing
                        delta = json.dumps(
                            {
                                "type": "content",
                                "delta": trailing,
                            }
                        )
                        yield f"data: {delta}\n\n"

                    telemetry.record_model_call(
                        stream_response,
                        telemetry.GEMINI_MODEL,
                        (time.perf_counter() - _t0) * 1000.0,
                        stage="generation",
                        trace=trace,
                    )

                # If there was an accumulated text (should always be the case
                # after the loop), run post-processing.
                _pp_start = time.perf_counter()

                # --- Safety output check on the final text ---
                if safety_enabled and combined_text:
                    output_result = safety_pipeline.output_check.enforce(
                        combined_text,
                        decision if safety_enabled else None,
                    )
                    combined_text = output_result.text

                # --- Hadith authenticity grading ---
                hadith_refs = annotate_hadith(combined_text)
                evidence_verification = None
                if EVIDENCE_VERIFICATION_ENABLED:
                    try:
                        evidence_verification = verify_evidence_claims(combined_text, citation_extraction, hadith_refs)
                    except Exception as exc:  # noqa: BLE001 - verification is best-effort
                        logger.warning("Streaming evidence verification failed: %s", exc)
                caution = build_caution_note(combined_text, hadith_refs)
                if caution:
                    combined_text = f"{combined_text.rstrip()}\n\n{caution}"

                # --- Confidence assessment & scholar review ---
                signals = build_signals(
                    combined_text,
                    is_religious=is_fiqh or bool(hadith_refs),
                    is_high_stakes=is_fiqh,
                    citation_verification=citation_extraction.score,
                )
                assessment = assess(signals)

                if assessment.queued:
                    try:
                        item = await enqueue_for_review(
                            question=prompt,
                            answer=combined_text,
                            score=assessment.score,
                            band=assessment.band.value,
                            signals=assessment.signals,
                            chat_id=chat_id,
                        )
                        assessment.review_id = item.id
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Could not queue streaming answer for review: %s", exc)
                        assessment.queued = False

                combined_text = apply_policy(combined_text, assessment)

                logger.info(
                    "confidence=%s",
                    {
                        "score": assessment.score,
                        "band": assessment.band.value,
                        "signals": assessment.signals_used,
                        "queued": assessment.queued,
                    },
                )

                # --- Build history ---
                history: list[Message] = []
                for msg in chat_session.history if chat_session else []:
                    try:
                        if hasattr(msg, "parts") and msg.parts:
                            content = msg.parts[0].text if hasattr(msg.parts[0], "text") else str(msg.parts[0])
                        else:
                            content = str(msg)
                        if msg.role == "user":
                            content = _strip_system_context(content)
                        history.append(
                            Message(
                                role="user" if msg.role == "user" else "model",
                                content=content,
                            )
                        )
                    except Exception as exc:
                        logger.warning("Error processing message in history: %s", exc)
                        continue

                trace.add_span(
                    "post_processing",
                    (time.perf_counter() - _pp_start) * 1000.0,
                )

                # --- Terminal done event ---
                done = json.dumps(
                    {
                        "type": "done",
                        "chat_id": chat_id,
                        "history": [m.model_dump() for m in history],
                        "text": combined_text,
                        "confidence": assessment.model_dump() if assessment else None,
                        "hadith_references": ([r.model_dump() for r in hadith_refs] if hadith_refs else None),
                        "fiqh": fiqh_info.model_dump() if fiqh_info else None,
                        "tafsir": tafsir_info.model_dump() if tafsir_info else None,
                        "zakat": zakat_info.model_dump() if zakat_info else None,
                        "citations": [c.model_dump() for c in citation_extraction.citations],
                        "evidence_verification": evidence_verification.model_dump() if evidence_verification else None,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {done}\n\n"

                logger.info("Streaming chat response completed for %s", chat_id)

                # --- Persist chat history ---
                # Awaited (not fire-and-forget) so a client that immediately
                # reloads the chat list sees this turn. Failures are caught
                # inside the helper; the stream is already complete.
                await _persist_chat_history(chat_id, body.user_id, chat_session)

            except asyncio.CancelledError:
                # Client disconnected mid-stream. Consume remaining chunks
                # (if any) so the Gemini SDK finalises chat.history and a
                # follow-up message in this session isn't left broken.
                logger.info("Client disconnected from streaming chat %s", chat_id)
                if stream_response is not None:
                    try:
                        await stream_response.resolve()
                    except Exception:
                        pass
                raise

            except Exception as exc:
                logger.error("Streaming error for %s: %s", chat_id, exc)
                err = json.dumps(
                    {
                        "type": "error",
                        "message": "An error occurred during response generation.",
                    }
                )
                try:
                    yield f"data: {err}\n\n"
                except Exception:
                    # Client may have already disconnected; nothing to do.
                    pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as exc:
        logger.exception(
            "Unexpected error initialising streaming chat %s: %s",
            getattr(request, "chat_id", None),
            exc,
        )
        raise APIException(
            status_code=500,
            detail="AI service error",
            hint="An unexpected server error occurred while initialising streaming chat. Please verify your connection and retry.",
        ) from exc
    finally:
        telemetry.registry.record_request(
            (time.perf_counter() - _handler_start) * 1000.0,
            error=False,
        )
        telemetry.current_trace.reset(_ctx_token)


def _strip_system_context(text: str) -> str:
    """Extract just the user's actual question from the full prompt that includes system context."""
    marker = "\nUser question: "
    if marker in text:
        return text.split(marker, 1)[1]
    return text


async def _persist_chat_history(chat_id: str, user_id: str | None, chat_session: Any) -> None:
    """Persist chat history and user-chat mapping."""
    try:
        if chat_session and hasattr(chat_session, "history") and chat_session.history:
            history_dicts = history_to_dicts(chat_session.history)
            # Strip system context from user messages so the store only contains
            # what the user actually asked, not the internal system prompt blocks.
            for msg in history_dicts:
                if msg.get("role") == "user":
                    msg["text"] = _strip_system_context(msg["text"])
            await session_store.save_history(chat_id, history_dicts)
            if user_id:
                await session_store.add_user_chat(user_id, chat_id)
                logger.info("Persisted chat %s for user %s (%d turns)", chat_id[:8], user_id[:8], len(history_dicts))
            else:
                logger.warning("No user_id — chat %s not linked to any user", chat_id[:8])
    except Exception as exc:
        logger.warning("Failed to persist chat history for %s: %s", chat_id, exc)


@app.get("/user/{user_id}/chats")
async def get_user_chats(user_id: str) -> dict[str, Any]:
    """List all chat IDs for a user."""
    try:
        chat_ids = await session_store.get_user_chats(user_id)
        chats = []
        for cid in chat_ids:
            history = await session_store.load_history(cid)
            # Strip system context from stored user messages
            for msg in history:
                if msg.get("role") == "user":
                    msg["text"] = _strip_system_context(msg.get("text", ""))
            # Use first user message as title, or last message snippet
            title = ""
            snippet = ""
            for msg in history:
                if msg.get("role") == "user" and not title:
                    title = msg.get("text", "")[:80]
                snippet = msg.get("text", "")[:120]
            created_at = await session_store.get_chat_created_at(cid)
            chats.append(
                {
                    "chat_id": cid,
                    "title": title or f"Chat {cid[:8]}",
                    "snippet": snippet,
                    "message_count": len(history),
                    "created_at": created_at,
                }
            )
        # Most recent first
        chats.sort(key=lambda c: float(str(c["created_at"] or 0)), reverse=True)
        return {"chats": chats}
    except Exception as e:
        logger.error("Error listing user chats", exc_info=True)
        raise APIException(
            status_code=500,
            detail="Internal server error",
            hint="Failed to load user chat sessions. Verify user_id format and retry.",
        ) from e


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str) -> list[dict[str, str]]:
    """Get the message history for a specific chat."""
    try:
        history = await session_store.load_history(chat_id)
        # Strip system context from persisted user messages (covers old data
        # stored before the stripping fix was deployed).
        for msg in history:
            if msg.get("role") == "user":
                msg["text"] = _strip_system_context(msg.get("text", ""))
        return history
    except Exception as e:
        logger.error("Error loading chat history", exc_info=True)
        raise APIException(
            status_code=500,
            detail="Internal server error",
            hint="Failed to retrieve chat history. Verify chat_id and retry.",
        ) from e


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str, user_id: str | None = None) -> dict[str, str]:
    try:
        existed = chat_id in active_chats
        active_chats.pop(chat_id, None)
        # Drop this session's feedback bookkeeping too, so the message-id list
        # and answer snapshots do not outlive the conversation they describe.
        for message_id in chat_message_ids.pop(chat_id, []):
            answer_snapshots.pop((chat_id, message_id), None)
        # Remove from persistent store
        await session_store.delete_session(chat_id)
        if user_id:
            await session_store.remove_user_chat(user_id, chat_id)
        if existed:
            logger.info(f"Deleted chat session: {chat_id}")
            return {"message": "Chat session deleted successfully"}
        return {"message": "Chat session not found"}
    except Exception as e:
        logger.error("Error deleting chat", exc_info=True)
        raise APIException(
            status_code=500,
            detail="Internal server error",
            hint="Failed to delete chat session. Please retry.",
        ) from e


@app.post("/evidence/verify", response_model=EvidenceVerificationResult)
async def verify_evidence_endpoint(body: EvidenceVerifyRequest) -> EvidenceVerificationResult:
    """Run the evidence verification agent on a generated answer.

    Extracts any citation block, parses Quran/Hadith references, links them to
    claims, and returns a structured audit of support strength.
    """
    try:
        cleaned_text, citation_extraction = extract_citations(body.text)
        hadith_refs = annotate_hadith(cleaned_text)
        return verify_evidence_claims(cleaned_text, citation_extraction, hadith_refs)
    except Exception as exc:
        logger.error("Evidence verification endpoint failed: %s", exc)
        raise APIException(
            status_code=500,
            detail="Evidence verification failed.",
            hint="The evidence verification agent could not process the supplied text. Verify the text is well-formed and retry.",
        ) from exc


# ---------------------------------------------------------------------------
# Feedback: capture and admin views
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    """Best-effort client IP for rate limiting.

    X-Forwarded-For is honored only when TRUST_PROXY_HEADERS is set — otherwise
    a client can rotate the header to get a fresh bucket every request. Even
    when trusted, take the rightmost hop (the address our proxy saw) rather than
    the leftmost, which the client controls.
    """
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                return hops[-1]
    return request.client.host if request.client else "unknown"


@app.post("/feedback", status_code=200)
async def submit_feedback(request: Request, body: FeedbackRequest) -> dict[str, Any]:
    """Attach a rating and optional failure categories to one model answer.

    The prompt/answer snapshot is resolved server-side from what the user was
    shown for that (chat_id, message_id). If the snapshot is gone — a free-tier
    restart evicts it — the client MUST supply prompt and answer, or the
    request is rejected with 422.

    Rate-limited per IP (in-process sliding window). Idempotent: resubmitting
    for the same (chat_id, message_id) overwrites the earlier record.
    """
    if not rate_limiter.is_allowed(_client_ip(request)):
        raise APIException(
            status_code=429,
            detail="Too many feedback submissions. Please wait before trying again.",
            hint="Feedback submissions are rate-limited to 20 per minute per IP. Please wait up to 60 seconds before submitting again.",
        )

    snapshot = answer_snapshots.get((body.chat_id, body.message_id))
    prompt_text = snapshot["prompt"] if snapshot else body.prompt
    answer_text = snapshot["answer"] if snapshot else body.answer

    if snapshot is None and (not prompt_text or not answer_text):
        raise APIException(
            status_code=422,
            detail=(
                "This answer is no longer in memory. Please supply 'prompt' and "
                "'answer' in the request body so the feedback has context."
            ),
            hint=(
                "Supply both 'prompt' and 'answer' string fields in the JSON body "
                "(e.g. {'chat_id': '...', 'message_id': '...', 'rating': 'up', 'prompt': '...', 'answer': '...'})."
            ),
        )

    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        chat_id=body.chat_id,
        message_id=body.message_id,
        rating=body.rating,
        categories=body.categories or [],
        comment=body.comment,
        prompt=prompt_text,
        answer=answer_text,
        model_name=telemetry.GEMINI_MODEL,
        generation_config=GENERATION_CONFIG,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    )

    try:
        # SQLite/Redis I/O is synchronous; keep it off the event loop.
        await run_in_threadpool(feedback_store.upsert, record)
    except Exception as exc:
        logger.error("Failed to store feedback: %s", exc)
        raise APIException(
            status_code=500,
            detail="Failed to store feedback.",
            hint="Database error storing feedback record. Please retry in a few moments.",
        ) from exc

    logger.info(
        "Feedback stored: chat_id=%s message_id=%s rating=%s",
        body.chat_id,
        body.message_id,
        body.rating,
    )
    return {"status": "ok", "feedback_id": record.feedback_id}


@app.get("/feedback/stats", dependencies=[Depends(require_admin)])
async def feedback_stats() -> dict[str, Any]:
    """Aggregate quality metrics: rating ratios, per-category, per-model, by day.

    Requires the X-Admin-Token header.
    """
    try:
        return await run_in_threadpool(feedback_store.stats)
    except Exception as exc:
        logger.error("Failed to fetch feedback stats: %s", exc)
        raise APIException(
            status_code=500,
            detail="Failed to fetch stats.",
            hint="Database error aggregating feedback metrics. Please retry.",
        ) from exc


@app.get("/feedback/records", dependencies=[Depends(require_admin)])
async def feedback_records(
    rating: str | None = None,
    category: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Recent flagged records, filterable by rating and category.

    Requires the X-Admin-Token header.
    """
    if rating and rating not in ("up", "down"):
        raise APIException(
            status_code=422,
            detail="rating must be 'up' or 'down'",
            hint="Set the 'rating' query parameter to either 'up' or 'down' (e.g., ?rating=down).",
        )
    if category and category not in FEEDBACK_TAXONOMY:
        raise APIException(
            status_code=422,
            detail=f"Unknown category. Valid: {sorted(FEEDBACK_TAXONOMY)}",
            hint=f"Use one of the allowed taxonomy categories: {', '.join(sorted(FEEDBACK_TAXONOMY))} (e.g., ?category=incorrect_information).",
        )
    if not (1 <= limit <= 500):
        raise APIException(
            status_code=422,
            detail="limit must be between 1 and 500",
            hint="Specify an integer limit between 1 and 500 (e.g., ?limit=50). Default is 50.",
        )
    try:
        records = await run_in_threadpool(feedback_store.list_records, rating, category, limit)
        return {"records": [r.to_dict() for r in records]}
    except Exception as exc:
        logger.error("Failed to fetch feedback records: %s", exc)
        raise APIException(
            status_code=500,
            detail="Failed to fetch records.",
            hint="Database error retrieving feedback records. Please retry.",
        ) from exc


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Lightweight liveness probe for container healthchecks and keep-alive pings."""
    return {"status": "ok"}


@app.get("/memory/{user_id}")
async def get_memory(user_id: str) -> dict[str, Any]:
    """Retrieve the stored user profile for transparency.

    TODO(#9): bind to authenticated principal — anyone who knows a user_id
    can currently read another user's memory.
    """
    profile = await memory_store.get_profile(user_id)
    if profile is None:
        raise APIException(
            status_code=404,
            detail="Memory not found",
            hint="No user profile found for the provided user_id. Memory is created automatically when chatting with user_id provided.",
        )
    return profile.model_dump()


@app.delete("/memory/{user_id}")
async def delete_memory(user_id: str) -> dict[str, str]:
    """Completely erase the stored user profile.

    TODO(#9): bind to authenticated principal — anyone who knows a user_id
    can currently erase another user's memory.
    """
    existed = await memory_store.delete_profile(user_id)
    if existed:
        logger.info("Deleted memory for user %s", user_id[:8])
        return {"message": "Memory deleted successfully"}
    return {"message": "Memory not found"}


def get_health_status(deep: bool = False) -> tuple[dict, int]:
    key_configured = bool(GEMINI_API_KEY)
    checks = {"gemini_api_key_configured": "ok" if key_configured else "fail"}
    if deep:
        checks["gemini_api_reachable"] = "skipped"
    if key_configured:
        return {"status": "ok", "version": app.version, "checks": checks}, 200
    return {
        "status": "unhealthy",
        "version": app.version,
        "checks": checks,
        "failing_check": "gemini_api_key_configured",
        "hint": "Configure the GEMINI_API_KEY environment variable with a valid Google Gemini API key.",
    }, 503


@app.get("/health")
async def health(deep: bool = False) -> JSONResponse:
    body, code = get_health_status(deep=deep)
    return JSONResponse(content=body, status_code=code)


@app.get("/cache/stats")
async def cache_stats() -> dict[str, Any]:
    exact_cache = get_chat_exact_cache()
    return {
        "semantic": semantic_cache.get_stats(),
        "exact": exact_cache.get_stats(),
        "combined": {
            "total_hits": semantic_cache.hits + exact_cache.hits,
            "total_misses": semantic_cache.misses + exact_cache.misses,
            "total_tokens_saved": semantic_cache.tokens_saved + exact_cache.tokens_saved,
        },
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Lightweight LLM observability surface: token, cost, and latency
    aggregates plus error rate. Contains only counts, durations, costs, model
    names, and trace-derived aggregates - never prompt or answer content.

    Cache hit-rate is sourced from the semantic cache's own precise counters
    (#27) rather than re-derived here, so the numbers stay consistent with
    /cache/stats. #9 (auth/rate limiting) can consume the cost/token totals
    below without this endpoint enforcing anything itself.
    """
    exact_cache = get_chat_exact_cache()
    snapshot = telemetry.registry.snapshot()
    snapshot["semantic_cache"] = semantic_cache.get_stats()
    snapshot["exact_cache"] = exact_cache.get_stats()
    snapshot["combined_cache"] = {
        "total_hits": semantic_cache.hits + exact_cache.hits,
        "total_misses": semantic_cache.misses + exact_cache.misses,
        "total_tokens_saved": semantic_cache.tokens_saved + exact_cache.tokens_saved,
    }
    return snapshot


@app.get("/confidence/policy")
async def confidence_policy() -> dict[str, Any]:
    """Current confidence thresholds and the queue's durability."""
    return {
        "thresholds": confidence_thresholds(),
        "review_queue": await review_store.stats(),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
