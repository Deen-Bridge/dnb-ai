from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    context: Optional[str] = None  # Additional context for specific queries
    madhhab: Optional[str] = None  # User's madhhab: hanafi, maliki, shafii, hanbali
    language: Optional[str] = None  # Preferred language for retrieved tafsir


class Moderation(BaseModel):
    category_id: Optional[str] = None
    action: str


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    history: List[Message]
    moderation: Optional[Moderation] = None
    fiqh: Optional[FiqhInfo] = None
    hadith_references: Optional[List[HadithReference]] = None
    tafsir: Optional[TafsirInfo] = None
    confidence: Optional[ConfidenceAssessment] = None


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


@app.get("/ping")
async def ping():
    logger.info("************** Ping pong ping pong *************")
    return {"************** Ping pong ping pong *************"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request, fastapi_response: Response):
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


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
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


@app.get("/cache/stats")
async def cache_stats():
    return semantic_cache.get_stats()


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
