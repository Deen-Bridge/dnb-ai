import json
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
import time

from google.api_core.exceptions import (
    ResourceExhausted,
    InvalidArgument,
    DeadlineExceeded,
    ServiceUnavailable,
)

import telemetry

from stellar import (
    ZakatInfo,
    build_chat_zakat_context,
    redact_secret_keys,
    router as stellar_router,
)
from safety import InputGate, OutputCheck, SafetyPipeline, load_policy
from semantic_cache import (
    SEMANTIC_CACHE_ENABLED,
    embed_text,
    get_cache,
    normalize_text,
)
from fiqh import (
    FIQH_IKHTILAF_CONTEXT,
    MADHHAB_LEAD_INSTRUCTION,
    FiqhInfo,
    classify_fiqh,
    normalize_madhhab,
)
from hadith import HADITH_ADAB_CONTEXT, HadithReference, annotate as annotate_hadith, build_caution_note
from study import router as study_router
from tafsir import (
    TafsirContext,
    TafsirInfo,
    build_chat_tafsir_context,
    router as tafsir_router,
    summarize_tafsir_context,
    tafsir_system_context,
)
from confidence import (
    ConfidenceAssessment,
    ConfidenceBand,
    apply_policy,
    assess,
    build_signals,
    thresholds as confidence_thresholds,
)
from review import enqueue_for_review, router as review_router
from review_store import get_review_store

from memory import ChatSummary, UserProfile, create_memory_store, render_user_context
from memory.extraction import (
    MEMORY_EXTRACTION_ENABLED,
    apply_updates,
    extract_updates,
    merge_summaries,
    summarize_conversation_turns,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="DeenBridge AI API")

# Stellar integration: read-only zakat/balance features on the network
# the rest of the Deen Bridge platform settles on
app.include_router(stellar_router)
app.include_router(study_router)
# Tafsir: grounded, attributed ayah explanations from named classical works
app.include_router(tafsir_router)
# Scholar review: the human end of the abstention loop
app.include_router(review_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://deenbridge.vercel.app",  # Production frontend
        "https://dnb-frontend.vercel.app",  # Your frontend domain
        "http://localhost:8000",  # Local API
        "https://dnb-ai.onrender.com",  # Render deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class CitationVerificationResult(BaseModel):
    source: str  # "quran" | "hadith"
    surah: Optional[int] = None
    ayah: Optional[int] = None
    collection: Optional[str] = None
    number: Optional[str] = None
    status: str  # "verified" | "mismatch" | "unverified" | "not_quoted"
    reason: Optional[str] = None


class ChatRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    context: Optional[str] = None  # Additional context for specific queries
    madhhab: Optional[str] = None  # User's madhhab: hanafi, maliki, shafii, hanbali
    language: Optional[str] = None  # BCP-47 response language (ar, en, ur, etc.); auto-detect when omitted
    user_id: Optional[str] = Field(default=None, max_length=128)  # Opaque user identifier for personalization
    remember: bool = True           # When False, existing memory is read but no new data persisted


class Message(BaseModel):
    role: str
    content: str


class Moderation(BaseModel):
    category_id: Optional[str] = None
    action: str


class ChatResponse(BaseModel):
    response: Optional[str] = None
    text: Optional[str] = None
    chat_id: str
    history: List[Message] = []
    moderation: Optional[Moderation] = None
    fiqh: Optional[FiqhInfo] = None
    hadith_references: Optional[List[HadithReference]] = None
    tafsir: Optional[TafsirInfo] = None
    confidence: Optional[ConfidenceAssessment] = None
    zakat: Optional[ZakatInfo] = None
    language: Optional[str] = None


def classify_for_safety(prompt: str, candidate_ids: List[str]):
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
safety_pipeline = SafetyPipeline(
    InputGate(safety_policy, classify_for_safety), OutputCheck(safety_policy)
)

# Semantic response cache
semantic_cache = get_cache()

# Durable queue for low-confidence religious answers awaiting a scholar
review_store = get_review_store()

# Per-user memory store (Redis-backed or in-memory)
memory_store = create_memory_store()

MAX_CHAT_HISTORY_TURNS = 20

# Tafsir retrieval seam: returns None for prompts that are not
# verse-explanation questions. Offline tests replace this with a stub.
DEFAULT_TAFSIR_LANGUAGE = "en"


async def tafsir_retriever(prompt: str, language: str) -> Optional[TafsirContext]:
    """Retrieve tafsir for a chat turn; never fail the turn over retrieval."""
    try:
        return await build_chat_tafsir_context(prompt, language)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("Tafsir retrieval failed; answering without it: %s", exc)
        return None


async def zakat_retriever(prompt: str, context: Optional[str]):
    """Compute zakat for a chat turn; never fail the turn over the lookup."""
    try:
        return await build_chat_zakat_context(prompt, context)
    except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
        logger.warning("Zakat lookup failed; answering without it: %s", exc)
        return None


def get_safety_settings():
    return [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]


# In-memory session store for demo purposes
sessions: Dict[str, Any] = {}
active_chats: Dict[str, Any] = {}

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
    "- Do NOT mix languages within a single response unless the user explicitly code-switches.\n"
)


def normalize_language(lang: Optional[str]) -> Optional[str]:
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


def get_model():
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=ISLAMIC_CONTEXT,
    )


GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))


def extract_text_safely(response: Any) -> Optional[str]:
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
    generation_config: Optional[Dict[str, Any]] = None,
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
            len(chat_session.history)
            if hasattr(chat_session, "history") and chat_session.history is not None
            else 0
        )
        try:
            kwargs: Dict[str, Any] = {"request_options": {"timeout": timeout}}
            if generation_config:
                kwargs["generation_config"] = generation_config
            response = await chat_session.send_message_async(
                message,
                **kwargs,
            )
            return response
        except (ServiceUnavailable, DeadlineExceeded, asyncio.TimeoutError) as exc:
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
    chat_session,
    user_message: str,
    original_text: str,
    mismatches: List[Dict[str, Any]],
) -> str:
    """Run exactly one corrective regeneration when a citation mismatch occurs in strict mode."""
    corrections_text = []
    for m in mismatches:
        if m.get("source") == "quran" and "correct_text" in m:
            corrections_text.append(
                f"- Surah {m['surah']}:{m['ayah']} text in corpus is: '{m['correct_text']}'. "
                f"Your quote did not match."
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request, fastapi_response: Response):
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
        chat_id = request.chat_id or str(uuid.uuid4())
        is_new_chat = chat_id not in active_chats
        is_bypass = http_request.headers.get("X-Cache-Bypass") == "1"

        # A user who pastes a Stellar secret key must not have it forwarded to
        # the model provider or written into stored history. Everything
        # downstream works from the redacted text; the zakat layer separately
        # detects that one was present and warns the user.
        prompt = redact_secret_keys(request.prompt)
        extra_context = redact_secret_keys(request.context)
        logger.info(f"Received chat request: {prompt[:100]}...")

        # --- Fiqh/intent classification & madhhab ---
        with trace.span("classification"):
            madhhab = normalize_madhhab(request.madhhab)
            is_fiqh = classify_fiqh(prompt)
            fiqh_info = FiqhInfo(is_fiqh_question=is_fiqh, madhhab_requested=madhhab)
            effective_language = normalize_language(request.language)

        # --- Tafsir and zakat retrieval (grouped as one telemetry stage) ---
        with trace.span("retrieval"):
            # Tafsir detection is offline (regex + the bundled surah index),
            # so a non-tafsir prompt costs nothing.
            tafsir_context = await tafsir_retriever(
                prompt, request.language or DEFAULT_TAFSIR_LANGUAGE
            )
            tafsir_info = summarize_tafsir_context(tafsir_context) if tafsir_context else None

            # Zakat detection is offline (keywords plus a key-shaped match), so
            # an ordinary prompt never touches Horizon or the gold-price API.
            zakat_context = await zakat_retriever(request.prompt, request.context)
            zakat_info = zakat_context.info if zakat_context else None

        # --- Memory lookup ---
        profile: Optional[UserProfile] = None
        summary: Optional[ChatSummary] = None
        if request.user_id:
            profile = await memory_store.get_profile(request.user_id)
            summary = await memory_store.get_chat_summary(f"{request.user_id}:{chat_id}")

        # Neither a tafsir-grounded answer nor a zakat answer goes through the
        # semantic response cache: the first is built from retrieved passages
        # (already cached by ayah key), and the second contains one user's real
        # balance, which must never be replayed to anyone else.
        is_cacheable = (
            is_new_chat
            and request.context is None
            and tafsir_context is None
            and zakat_context is None
            and request.user_id is None
            and SEMANTIC_CACHE_ENABLED
        )

        # --- Semantic cache lookup ---
        embedding: Any = None
        normalized: Optional[str] = None
        if is_cacheable and not is_bypass:
            normalized = normalize_text(prompt)
            embedding = embed_text(normalized)
            cached = semantic_cache.get(embedding)
            if cached is not None:
                fastapi_response.headers["X-Semantic-Cache"] = "hit"
                model = genai.GenerativeModel(
                    telemetry.GEMINI_MODEL,
                    safety_settings=get_safety_settings(),
                )
                chat_session = model.start_chat(history=[
                    {"role": "user", "parts": [{"text": prompt}]},
                    {"role": "model", "parts": [{"text": cached.response}]},
                ])
                active_chats[chat_id] = chat_session
                logger.info("Semantic cache HIT for prompt: %s", prompt[:80])
                _finalize()
                _succeeded = True
                return ChatResponse(
                    response=cached.response,
                    chat_id=chat_id,
                    history=cached.history,
                    fiqh=fiqh_info,
                    hadith_references=annotate_hadith(cached.response),
                    language=effective_language,
                )
        elif is_bypass:
            semantic_cache.bypasses += 1

        # --- Normal flow (cache miss / bypass / not cacheable) ---
        async def generate(safety_prompt: str) -> str:
            if chat_id not in active_chats:
                logger.info(f"Creating new chat session: {chat_id}")
                model = get_model()
                active_chats[chat_id] = model.start_chat(history=[])

            system_context = ISLAMIC_CONTEXT + HADITH_ADAB_CONTEXT
            if is_fiqh:
                system_context += FIQH_IKHTILAF_CONTEXT
                if madhhab:
                    system_context += MADHHAB_LEAD_INSTRUCTION.format(madhhab=madhhab)
            if tafsir_context is not None:
                system_context += tafsir_system_context(tafsir_context)
            if zakat_context is not None:
                system_context += zakat_context.prompt_block
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
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
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
                raise HTTPException(status_code=500, detail="Empty response from AI model")
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

        # Get chat history
        history = []
        chat_session = active_chats.get(chat_id)
        for message in chat_session.history if chat_session else []:
            try:
                if hasattr(message, 'parts') and message.parts:
                    content = message.parts[0].text if hasattr(message.parts[0], 'text') else str(message.parts[0])
                else:
                    content = str(message)

                history.append(Message(
                    role="user" if message.role == "user" else "model",
                    content=content
                ))
            except Exception as e:
                logger.warning(f"Error processing message in history: {str(e)}")
                continue

        response_text = safety_result.text if safety_result else generated_text

        # --- Hadith authenticity grading ---
        # Baked into response_text *before* the cache write so a cached hit
        # replays the same caution the user originally saw.
        hadith_refs = annotate_hadith(response_text)
        caution = build_caution_note(response_text, hadith_refs)
        if caution:
            response_text = f"{response_text.rstrip()}\n\n{caution}"

        # --- Confidence, abstention, and scholar escalation ---
        # is_religious and is_high_stakes reuse classification that already ran
        # this turn (the fiqh classifier and the hadith annotator) rather than
        # adding a competing classifier. self_consistency (#ai-18) and
        # citation_verification (#40) are passed through when those components
        # supply them; until then they are simply absent from the average.
        signals = build_signals(
            response_text,
            is_religious=is_fiqh or bool(hadith_refs),
            is_high_stakes=is_fiqh,
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

        # --- Semantic cache write ---
        # Only confident answers are cached. Replaying an abstention, or a
        # hedged answer whose warning would outlive the doubt that caused it,
        # would spread one turn's uncertainty to every later asker.
        is_cacheable = is_cacheable and assessment.band is ConfidenceBand.CONFIDENT
        if is_cacheable and (safety_result is None or safety_result.generator_called):
            if embedding is None:
                normalized = normalize_text(prompt)
                embedding = embed_text(normalized)
            semantic_cache.put(embedding, response_text, chat_id, history)
            logger.info("Semantic cache WRITE for prompt: %s", prompt[:80])

        fastapi_response.headers["X-Semantic-Cache"] = "bypass" if is_bypass else "miss"

        trace.add_span("post_processing", (time.perf_counter() - _pp_start) * 1000.0)
        logger.info("Chat response generated successfully")
        # Build the response before finalizing, so a construction/validation
        # failure is handled only by the error path and the request is not
        # counted as both a success (here) and an error (except block).
        response_obj = ChatResponse(
            response=response_text,
            chat_id=chat_id,
            history=history,
            moderation=Moderation(
                category_id=safety_result.category_id,
                action=safety_result.action,
            ) if safety_result and safety_result.category_id else None,
            fiqh=fiqh_info,
            hadith_references=hadith_refs,
            tafsir=tafsir_info,
            confidence=assessment,
            zakat=zakat_info,
            language=effective_language,
        )
        _finalize()
        _succeeded = True

        # --- Background memory extraction and summarization ---
        # Runs as fire-and-forget tasks after the response is sent.
        if request.user_id and request.remember and MEMORY_EXTRACTION_ENABLED:
            asyncio.create_task(
                _extract_and_update_memory(
                    request.user_id, prompt, response_text, chat_id, summary, memory_store,
                )
            )
            logger.info("Memory extraction scheduled for user %s", request.user_id[:8])

        # --- Summary eviction ---
        # After enough turns accumulate, summarize old history and persist.
        if request.user_id and request.remember and MEMORY_EXTRACTION_ENABLED:
            chat_session = active_chats.get(chat_id)
            if chat_session and hasattr(chat_session, "history") and chat_session.history:
                if len(chat_session.history) >= MAX_CHAT_HISTORY_TURNS:
                    asyncio.create_task(
                        _summarize_history(
                            f"{request.user_id}:{chat_id}",
                            chat_session.history,
                            summary,
                            memory_store,
                        )
                    )
                    logger.info("History summarization triggered for %s", request.user_id[:8])

        return response_obj

    except ResourceExhausted as exc:
        logger.warning("Gemini rate limit exceeded for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"X-Trace-Id": trace.trace_id},
        )
    except InvalidArgument as exc:
        logger.warning("Invalid argument for Gemini call in chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid request parameters.",
            headers={"X-Trace-Id": trace.trace_id},
        )
    except (DeadlineExceeded, asyncio.TimeoutError) as exc:
        logger.warning("Gemini API call timed out for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=504,
            detail="AI service timed out.",
            headers={"X-Trace-Id": trace.trace_id},
        )
    except ServiceUnavailable as exc:
        logger.warning("Gemini service unavailable for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable.",
            headers={"X-Trace-Id": trace.trace_id},
        )
    except HTTPException as exc:
        # Attach the trace id to already-typed HTTP errors (e.g. the empty-response
        # 500 raised inside generate) so a failed request stays correlatable.
        exc.headers = {**(exc.headers or {}), "X-Trace-Id": trace.trace_id}
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /chat handler for session %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=500,
            detail="AI service error",
            headers={"X-Trace-Id": trace.trace_id},
        )
    finally:
        # Record the request exactly once: the success path already recorded it
        # via _finalize(); anything that reached an except path is an error.
        if not _succeeded:
            telemetry.registry.record_request(
                (time.perf_counter() - _handler_start) * 1000.0, error=True
            )
        telemetry.current_trace.reset(_ctx_token)


async def _extract_and_update_memory(
    user_id: str, prompt: str, response: str, chat_id: str,
    existing_summary: Optional[ChatSummary],
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
    chat_id: str, history: list, existing_summary: Optional[ChatSummary],
    store: Any,
) -> None:
    """Summarize accumulated conversation turns and persist."""
    try:
        turns = [
            {"role": m.role, "text": m.parts[0].text if m.parts else ""}
            for m in history
        ]
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
async def chat_stream(request: ChatRequest, http_request: Request):
    """Streaming chat endpoint using Server-Sent Events (SSE)."""
    try:
        chat_id = request.chat_id or str(uuid.uuid4())
        prompt = redact_secret_keys(request.prompt)
        extra_context = redact_secret_keys(request.context)
        logger.info(f"Received streaming chat request: {prompt[:100]}...")

        # --- Fiqh classification & madhhab ---
        madhhab = normalize_madhhab(request.madhhab)
        is_fiqh = classify_fiqh(prompt)
        effective_language = normalize_language(request.language)

        # --- Tafsir retrieval ---
        tafsir_context = await tafsir_retriever(
            prompt, request.language or DEFAULT_TAFSIR_LANGUAGE
        )

        # --- Zakat calculation ---
        zakat_context = await zakat_retriever(request.prompt, request.context)

        async def event_generator():
            try:
                # Send metadata first
                metadata = json.dumps({"type": "metadata", "chat_id": chat_id, "language": effective_language})
                yield f"data: {metadata}\n\n"

                # Create or get chat session
                if chat_id not in active_chats:
                    logger.info(f"Creating new streaming chat session: {chat_id}")
                    model = genai.GenerativeModel(
                        telemetry.GEMINI_MODEL,
                        safety_settings=get_safety_settings()
                    )
                    active_chats[chat_id] = model.start_chat(history=[])

                # Build system context
                system_context = ISLAMIC_CONTEXT + HADITH_ADAB_CONTEXT
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

                context = f"Additional context: {extra_context}\n\n" if extra_context else ""
                full_prompt = f"{system_context}\n{context}User question: {prompt}"

                # Stream the response
                logger.info("Starting streaming response...")
                response_stream = active_chats[chat_id].send_message(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    },
                    stream=True
                )

                # Yield each chunk
                for chunk in response_stream:
                    if chunk.text:
                        content_data = json.dumps({"type": "content", "text": chunk.text})
                        yield f"data: {content_data}\n\n"

                # Build history for response
                history = []
                chat_session = active_chats.get(chat_id)
                for message in chat_session.history if chat_session else []:
                    try:
                        if hasattr(message, 'parts') and message.parts:
                            content = message.parts[0].text if hasattr(message.parts[0], 'text') else str(message.parts[0])
                        else:
                            content = str(message)
                        history.append(Message(
                            role="user" if message.role == "user" else "model",
                            content=content
                        ))
                    except Exception as e:
                        logger.warning(f"Error processing message in history: {str(e)}")
                        continue

                # Send done event
                done_data = json.dumps({"type": "done", "chat_id": chat_id, "history": [m.model_dump() for m in history]})
                yield f"data: {done_data}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                error_data = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        logger.error("Streaming Chat API Error", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    chat_id = request.chat_id or "default"
    if chat_id not in sessions:
        model = get_model()
        sessions[chat_id] = model.start_chat(history=[])

    chat_session = sessions[chat_id]

    async def event_stream():
        full_text_parts: List[str] = []
        try:
            response_stream = None
            if hasattr(chat_session, "send_message_async"):
                response_stream = await chat_session.send_message_async(request.message, stream=True)
            else:
                response_stream = chat_session.send_message(request.message, stream=True)

            async for chunk in response_stream:
                text = getattr(chunk, "text", None)
                if text is None and isinstance(chunk, dict):
                    text = chunk.get("text")
                if text is None:
                    continue

                full_text_parts.append(text)
                payload = json.dumps({"delta": text})
                yield f"event: delta\ndata: {payload}\n\n"
        except Exception as exc:  # pragma: no cover - defensive path
            error_payload = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error_payload}\n\n"
            return

        final_payload = json.dumps({"chat_id": chat_id, "text": "".join(full_text_parts)})
        yield f"event: done\ndata: {final_payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        if chat_id in active_chats:
            del active_chats[chat_id]
            logger.info(f"Deleted chat session: {chat_id}")
            return {"message": "Chat session deleted successfully"}
        return {"message": "Chat session not found"}
    except Exception as e:
        logger.error("Error deleting chat", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/ping")
async def ping():
    """Lightweight liveness probe for container healthchecks and keep-alive pings."""
    return {"status": "ok"}


@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Retrieve the stored user profile for transparency.

    TODO(#9): bind to authenticated principal — anyone who knows a user_id
    can currently read another user's memory.
    """
    profile = await memory_store.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return profile.model_dump()


@app.delete("/memory/{user_id}")
async def delete_memory(user_id: str):
    """Completely erase the stored user profile.

    TODO(#9): bind to authenticated principal — anyone who knows a user_id
    can currently erase another user's memory.
    """
    existed = await memory_store.delete_profile(user_id)
    if existed:
        logger.info("Deleted memory for user %s", user_id[:8])
        return {"message": "Memory deleted successfully"}
    return {"message": "Memory not found"}


@app.get("/cache/stats")
async def cache_stats():
    return semantic_cache.get_stats()


@app.get("/metrics")
async def metrics():
    """Lightweight LLM observability surface: token, cost, and latency
    aggregates plus error rate. Contains only counts, durations, costs, model
    names, and trace-derived aggregates - never prompt or answer content.

    Cache hit-rate is sourced from the semantic cache's own precise counters
    (#27) rather than re-derived here, so the numbers stay consistent with
    /cache/stats. #9 (auth/rate limiting) can consume the cost/token totals
    below without this endpoint enforcing anything itself.
    """
    snapshot = telemetry.registry.snapshot()
    snapshot["semantic_cache"] = semantic_cache.get_stats()
    return snapshot


@app.get("/confidence/policy")
async def confidence_policy():
    """Current confidence thresholds and the queue's durability."""
    return {
        "thresholds": confidence_thresholds(),
        "review_queue": await review_store.stats(),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
