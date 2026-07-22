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
from study import router as study_router
from rag import RAG_ENABLED, SourceDocument, format_reference_passages, retrieve

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


class Moderation(BaseModel):
    category_id: Optional[str] = None
    action: str


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    history: List[Message]
    moderation: Optional[Moderation] = None
    sources: Optional[List[SourceDocument]] = None


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
    rag_sources: list[SourceDocument] = []

    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        chat_id = request.chat_id or str(uuid.uuid4())
        is_new_chat = chat_id not in active_chats
        is_bypass = http_request.headers.get("X-Cache-Bypass") == "1"
        is_cacheable = is_new_chat and request.context is None and SEMANTIC_CACHE_ENABLED

        # --- RAG retrieval (before generate) ---
        rag_passages = ""
        if RAG_ENABLED:
            rag_sources = retrieve(request.prompt)
            if rag_sources:
                rag_passages = format_reference_passages(rag_sources)
                logger.info("RAG: %d source(s) retrieved", len(rag_sources))
            else:
                logger.info("RAG: no sources retrieved")

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

            context = f"Additional context: {request.context}\n\n" if request.context else ""
            full_prompt = f"{ISLAMIC_CONTEXT}\n{context}User question: {safety_prompt}{rag_passages}"
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

        # --- Semantic cache write ---
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
            sources=rag_sources or None,
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


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
