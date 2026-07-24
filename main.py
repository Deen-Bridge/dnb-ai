from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, RootModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import logging
from typing import Any, List, Optional
import uuid

from stellar import router as stellar_router
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_DESCRIPTION = """
The AI service behind **Deen Bridge**, a platform for authentic Islamic
education built on the Stellar network.

`/chat` wraps Google's Gemini model with an Islamic-knowledge system prompt and
several layers that exist to keep answers trustworthy: content-safety
classification, madhhab-aware fiqh handling, hadith authenticity grading,
tafsir-grounded ayah explanations, and a confidence score that makes the
service abstain — or route an answer to a human scholar — rather than guess.

### Sessions

`POST /chat` is the whole conversation API. **Omit `chat_id` to start a new
session**; the response returns the id that was created. **Pass that same
`chat_id` back on the next request to continue the conversation** — history is
kept server-side, so you never resend earlier turns. `DELETE /chat/{chat_id}`
ends a session.

### Response envelope

Every answer carries the text plus optional metadata blocks describing *how*
it was produced — `moderation`, `fiqh`, `hadith_references`, `tafsir`, and
`confidence`. All of them are additive and may be `null`; a client that only
reads `response` and `chat_id` keeps working.
"""

TAGS_METADATA = [
    {
        "name": "chat",
        "description": "Conversation with the assistant, and session lifecycle.",
    },
    {
        "name": "health",
        "description": "Liveness and service-internal metrics.",
    },
    {
        "name": "tafsir",
        "description": (
            "Ayah explanations retrieved from named classical tafsir works, "
            "each attributed to its author."
        ),
    },
    {
        "name": "study",
        "description": "Schema-validated quiz and flashcard generation.",
    },
    {
        "name": "stellar",
        "description": (
            "Read-only Stellar features, including zakat on a wallet's "
            "on-chain USDC balance. Public keys only — secret keys are never "
            "accepted."
        ),
    },
    {
        "name": "scholar-review",
        "description": (
            "Human review queue for low-confidence religious answers. "
            "Requires the `X-Review-Token` header."
        ),
    },
]

app = FastAPI(
    title="DeenBridge AI API",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Deen Bridge",
        "url": "https://github.com/Deen-Bridge/dnb-ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/Deen-Bridge/dnb-ai/blob/main/LICENSE",
    },
)

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

# Configure Gemini
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    logger.info("Configuring Gemini API...")
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Error configuring Gemini: {str(e)}")
    raise

# Store active chats
active_chats = {}

# Islamic context and safety instructions
ISLAMIC_CONTEXT = """You are an AI assistant specialized in providing Islamic knowledge and guidance.
Your responses should:
1. Be based on authentic Islamic sources (Quran and Hadith)
2. Be respectful and appropriate
3. Avoid controversial or divisive topics
4. Focus on promoting understanding and unity
5. Acknowledge when a question is beyond your scope
6. Always maintain Islamic etiquette (adab) in responses

Remember to:
- Cite sources when possible
- Be clear about what is from authentic sources vs. scholarly opinion
- Encourage consulting with local scholars for complex matters
- Promote positive Islamic values and character
"""


class Message(BaseModel):
    """One turn of the conversation, as replayed in `history`."""

    role: str = Field(
        ...,
        description="Who produced this turn: `user` or `model`.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="The text of the turn.",
        examples=["What are the conditions of wudu?"],
    )


class ChatRequest(BaseModel):
    """A question for the assistant, optionally continuing an existing session."""

    prompt: str = Field(
        ...,
        description="The user's question or message.",
        examples=["What are the conditions of wudu?"],
    )
    chat_id: Optional[str] = Field(
        None,
        description=(
            "Session to continue. **Omit to start a new session** — the "
            "response returns the id that was created. Pass that id back on "
            "later requests to keep the conversation going; history is stored "
            "server-side, so earlier turns are never resent."
        ),
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    context: Optional[str] = Field(
        None,
        description=(
            "Extra context for this one question — for example the lesson the "
            "user is reading, or their Stellar public key. It is prepended to "
            "the prompt and is **not** stored in the session history. Sending "
            "it also opts this turn out of the response cache."
        ),
        examples=["The user is reading a lesson on the fiqh of purification."],
    )
    madhhab: Optional[str] = Field(
        None,
        description=(
            "The user's school of jurisprudence, so fiqh answers lead with it: "
            "`hanafi`, `maliki`, `shafii`, or `hanbali`. Other schools' "
            "positions are still presented. An unrecognized value is ignored."
        ),
        examples=["shafii"],
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Preferred language code for retrieved tafsir and translations "
            "(default `en`). A work with no edition in this language is "
            "returned in its original language, labelled as such."
        ),
        examples=["en"],
    )

    # JSON Schema `examples` are plain instances of the model. The labelled
    # variants a reader picks between live on the /chat operation instead, as
    # OpenAPI Example Objects (see CHAT_REQUEST_EXAMPLES).
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"prompt": "What are the conditions of wudu?"},
                {
                    "prompt": "Does touching a cat break it?",
                    "chat_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "madhhab": "shafii",
                },
            ]
        }
    )


# Rendered as the example dropdown on POST /chat.
CHAT_REQUEST_EXAMPLES = {
    "start_session": {
        "summary": "Start a new session",
        "description": "No chat_id — the response returns a newly created one.",
        "value": {"prompt": "What are the conditions of wudu?"},
    },
    "continue_session": {
        "summary": "Continue a session",
        "description": "Reuse the chat_id returned by the first call.",
        "value": {
            "prompt": "Does touching a cat break it?",
            "chat_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "madhhab": "shafii",
        },
    },
}


class Moderation(BaseModel):
    """Present only when the safety policy matched the request."""

    category_id: Optional[str] = Field(
        None,
        description="Policy category that matched, from `safety/policy.yaml`.",
        examples=["self_harm"],
    )
    action: str = Field(
        ...,
        description="What the policy did: `allow`, `allow_with_guidance`, or `refuse`.",
        examples=["allow_with_guidance"],
    )


class DeleteChatResponse(BaseModel):
    """Result of ending a chat session."""

    message: str = Field(
        ...,
        description=(
            "Human-readable outcome. Deleting an unknown or already-deleted "
            "session is not an error — it reports that none was found."
        ),
        examples=["Chat session deleted successfully"],
    )


class PingResponse(RootModel[List[str]]):
    """Liveness probe payload — a single-element array."""

    root: List[str] = Field(
        ...,
        description="Liveness marker.",
        examples=[["************** Ping pong ping pong *************"]],
    )


class ChatResponse(BaseModel):
    """The assistant's answer, plus optional metadata about how it was produced.

    Every metadata block is additive and may be `null`; a client that reads
    only `response` and `chat_id` is unaffected by any of them.
    """

    response: str = Field(
        ...,
        description=(
            "The assistant's answer. This is the text to display — it already "
            "includes any hadith-authenticity caution, uncertainty note, or "
            "abstention message the service decided to attach."
        ),
        examples=["Wudu requires intention (niyyah), washing the face, ..."],
    )
    chat_id: str = Field(
        ...,
        description=(
            "Session id for this conversation. Send it back on the next "
            "request to continue; it is newly created when the request omitted "
            "one."
        ),
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    history: List[Message] = Field(
        ...,
        description="Every turn in the session so far, oldest first.",
    )
    moderation: Optional[Moderation] = Field(
        None,
        description="Safety-policy outcome, when a policy category matched.",
    )
    fiqh: Optional[FiqhInfo] = Field(
        None,
        description="Whether this was treated as a fiqh question, and under which madhhab.",
    )
    hadith_references: Optional[List[HadithReference]] = Field(
        None,
        description=(
            "Hadith cited in the answer, each with its authenticity grade and "
            "grader, so a weak narration is never presented as authentic."
        ),
    )
    tafsir: Optional[TafsirInfo] = Field(
        None,
        description=(
            "For verse-explanation questions: the ayat retrieved and the tafsir "
            "works whose text actually backed the answer."
        ),
    )
    confidence: Optional[ConfidenceAssessment] = Field(
        None,
        description=(
            "How reliable the answer is judged to be (0–1), the band that score "
            "falls in, and whether the service abstained or queued the answer "
            "for a scholar."
        ),
    )


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
        "gemini-2.5-flash-preview-05-20",
        system_instruction=classifier_instruction,
    )
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
        request_options={"timeout": 30},
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


@app.get(
    "/ping",
    response_model=PingResponse,
    tags=["health"],
    summary="Liveness check",
)
async def ping():
    """Confirm the service is up.

    Takes no parameters and touches no dependency — a 200 means this process is
    serving requests, not that Gemini or Redis are reachable.
    """
    logger.info("************** Ping pong ping pong *************")
    return {"************** Ping pong ping pong *************"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ask the assistant a question",
    responses={
        200: {
            "description": "The assistant's answer, with any metadata blocks that applied.",
            "content": {
                "application/json": {
                    "example": {
                        "response": "Wudu requires intention (niyyah), washing the face, ...",
                        "chat_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "history": [
                            {"role": "user", "content": "What are the conditions of wudu?"},
                            {
                                "role": "model",
                                "content": "Wudu requires intention (niyyah), washing the face, ...",
                            },
                        ],
                        "fiqh": {"is_fiqh_question": True, "madhhab_requested": "shafii"},
                        "confidence": {
                            "score": 0.55,
                            "band": "uncertain",
                            "abstained": False,
                            "queued": False,
                            "signals": {"expressed_certainty": 1.0},
                            "signals_used": ["expressed_certainty"],
                            "review_id": None,
                        },
                    }
                }
            },
        },
        422: {
            "description": (
                "The request body failed validation — for example `prompt` is "
                "missing or is not a string."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "missing",
                                "loc": ["body", "prompt"],
                                "msg": "Field required",
                                "input": {"chat_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
                            }
                        ]
                    }
                }
            },
        },
        500: {
            "description": (
                "Generation failed — the upstream model errored or returned an "
                "empty response."
            ),
            "content": {
                "application/json": {
                    "example": {"detail": "❌ Chat API Error: Empty response from AI model"}
                }
            },
        },
    },
)
async def chat(
    http_request: Request,
    fastapi_response: Response,
    request: ChatRequest = Body(..., openapi_examples=CHAT_REQUEST_EXAMPLES),
):
    """Send a message and get the assistant's answer.

    **Starting a session:** omit `chat_id`. The response carries a newly
    created one.

    **Continuing a session:** pass that `chat_id` back. Conversation history is
    held server-side, so only the new message is sent each time; the full
    history is replayed in the response.

    The answer is shaped by several layers before it is returned: the safety
    policy may add guidance or refuse, fiqh questions lead with the requested
    madhhab, cited hadith are graded, verse-explanation questions are grounded
    in real tafsir, and a low-confidence religious answer may be replaced by an
    abstention and queued for a scholar. Everything relevant is reported in the
    optional metadata blocks alongside `response`.
    """
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        chat_id = request.chat_id or str(uuid.uuid4())
        is_new_chat = chat_id not in active_chats
        is_bypass = http_request.headers.get("X-Cache-Bypass") == "1"

        # --- Fiqh classification & madhhab ---
        madhhab = normalize_madhhab(request.madhhab)
        is_fiqh = classify_fiqh(request.prompt)
        fiqh_info = FiqhInfo(is_fiqh_question=is_fiqh, madhhab_requested=madhhab)

        # --- Tafsir retrieval for verse-explanation questions ---
        # Detection is offline (regex + the bundled surah index), so a
        # non-tafsir prompt costs nothing.
        tafsir_context = await tafsir_retriever(
            request.prompt, request.language or DEFAULT_TAFSIR_LANGUAGE
        )
        tafsir_info = summarize_tafsir_context(tafsir_context) if tafsir_context else None

        # A grounded tafsir answer is built from retrieved passages, so it does
        # not go through the semantic response cache — the expensive part, the
        # tafsir text itself, is already cached by exact ayah key in
        # semantic_cache.KeyedCache.
        is_cacheable = (
            is_new_chat
            and request.context is None
            and tafsir_context is None
            and SEMANTIC_CACHE_ENABLED
        )

        # --- Semantic cache lookup ---
        embedding: Any = None
        normalized: Optional[str] = None
        if is_cacheable and not is_bypass:
            normalized = normalize_text(request.prompt)
            embedding = embed_text(normalized)
            cached = semantic_cache.get(embedding)
            if cached is not None:
                fastapi_response.headers["X-Semantic-Cache"] = "hit"
                model = genai.GenerativeModel(
                    'gemini-2.5-flash-preview-05-20',
                    safety_settings=get_safety_settings(),
                )
                chat_session = model.start_chat(history=[
                    {"role": "user", "parts": [{"text": request.prompt}]},
                    {"role": "model", "parts": [{"text": cached.response}]},
                ])
                active_chats[chat_id] = chat_session
                logger.info("Semantic cache HIT for prompt: %s", request.prompt[:80])
                return ChatResponse(
                    response=cached.response,
                    chat_id=chat_id,
                    history=cached.history,
                    fiqh=fiqh_info,
                    hadith_references=annotate_hadith(cached.response),
                )
        elif is_bypass:
            semantic_cache.bypasses += 1

        # --- Normal flow (cache miss / bypass / not cacheable) ---
        def generate(safety_prompt: str) -> str:
            if chat_id not in active_chats:
                logger.info(f"Creating new chat session: {chat_id}")
                model = genai.GenerativeModel(
                    'gemini-2.5-flash-preview-05-20',
                    safety_settings=get_safety_settings()
                )
                active_chats[chat_id] = model.start_chat(history=[])

            system_context = ISLAMIC_CONTEXT + HADITH_ADAB_CONTEXT
            if is_fiqh:
                system_context += FIQH_IKHTILAF_CONTEXT
                if madhhab:
                    system_context += MADHHAB_LEAD_INSTRUCTION.format(madhhab=madhhab)
            if tafsir_context is not None:
                system_context += tafsir_system_context(tafsir_context)
            context = f"Additional context: {request.context}\n\n" if request.context else ""
            full_prompt = f"{system_context}\n{context}User question: {safety_prompt}"
            logger.info("Sending message to chat...")
            response = active_chats[chat_id].send_message(
                full_prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            if not response.text:
                raise HTTPException(status_code=500, detail="Empty response from AI model")
            return response.text

        enabled = os.getenv("SAFETY_PIPELINE_ENABLED", "true").lower() not in {"0", "false", "off"}
        if enabled:
            safety_result = await safety_pipeline.run_async(request.prompt, generate)
        else:
            safety_result = None
            generated_text = generate(request.prompt)

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
                    question=request.prompt,
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
                normalized = normalize_text(request.prompt)
                embedding = embed_text(normalized)
            semantic_cache.put(embedding, response_text, chat_id, history)
            logger.info("Semantic cache WRITE for prompt: %s", request.prompt[:80])

        fastapi_response.headers["X-Semantic-Cache"] = "bypass" if is_bypass else "miss"

        logger.info("Chat response generated successfully")
        return ChatResponse(
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
        )

    except Exception as e:
        error_msg = f"❌ Chat API Error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete(
    "/chat/{chat_id}",
    response_model=DeleteChatResponse,
    tags=["chat"],
    summary="End a chat session",
    responses={
        200: {
            "description": (
                "The session was deleted, or no such session existed. Both "
                "cases return 200 — deleting is idempotent."
            ),
            "content": {
                "application/json": {
                    "examples": {
                        "deleted": {
                            "summary": "Session existed",
                            "value": {"message": "Chat session deleted successfully"},
                        },
                        "not_found": {
                            "summary": "Unknown or already deleted",
                            "value": {"message": "Chat session not found"},
                        },
                    }
                }
            },
        },
        500: {
            "description": "Deletion failed unexpectedly.",
            "content": {
                "application/json": {
                    "example": {"detail": "❌ Error deleting chat: ..."}
                }
            },
        },
    },
)
async def delete_chat(chat_id: str = FastAPIPath(
    ...,
    description="The `chat_id` returned by `POST /chat`.",
    examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
)):
    """Discard a conversation and its stored history.

    Idempotent: deleting a session that does not exist (or was already deleted)
    returns 200 with a "not found" message rather than a 404.
    """
    try:
        if chat_id in active_chats:
            del active_chats[chat_id]
            logger.info(f"Deleted chat session: {chat_id}")
            return {"message": "Chat session deleted successfully"}
        return {"message": "Chat session not found"}
    except Exception as e:
        error_msg = f"❌ Error deleting chat: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get(
    "/cache/stats",
    tags=["health"],
    summary="Semantic cache metrics",
)
async def cache_stats():
    """Hit rate, size, and eviction counts for the semantic response cache.

    Counters are per-process and reset on restart.
    """
    return semantic_cache.get_stats()


@app.get(
    "/confidence/policy",
    tags=["health"],
    summary="Confidence thresholds and review-queue depth",
)
async def confidence_policy():
    """Current confidence thresholds and the queue's durability.

    Useful for confirming what an environment is actually configured to do:
    the abstain/hedge boundaries in force, and whether the scholar-review queue
    is backed by Redis (`durable`) or only by process memory.
    """
    return {
        "thresholds": confidence_thresholds(),
        "review_queue": await review_store.stats(),
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
