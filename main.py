from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime, timezone

from stellar import router as stellar_router
from feedback import (
    FeedbackRecord,
    FEEDBACK_TAXONOMY,
    COMMENT_MAX_CHARS,
    rate_limiter,
    store as feedback_store,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="DeenBridge AI API")

# Stellar integration: read-only zakat/balance features on the network
# the rest of the Deen Bridge platform settles on
app.include_router(stellar_router)

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

# Admin token — stopgap until issue #9 provides real auth/rate-limiting
# infrastructure.  Keep this seam isolated; do not add auth logic here.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def require_admin(token: Optional[str] = Depends(_admin_header)) -> None:
    """Dependency: require a valid ADMIN_TOKEN header."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_TOKEN is not configured on this server.",
        )
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token.")


# Store active chats
# {chat_id: {"session": genai.ChatSession, "model_name": str, "gen_config": dict,
#            "message_ids": [str, ...]}}
active_chats: Dict[str, Any] = {}

# Model name and generation config — captured into feedback records so a
# flagged answer is reproducible evidence.
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GENERATION_CONFIG: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str
    message_id: Optional[str] = None  # present only on model turns


class ChatRequest(BaseModel):
    prompt: str
    chat_id: Optional[str] = None
    context: Optional[str] = None  # Additional context for specific queries


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    message_id: str          # stable ID for the model answer just returned
    history: List[Message]


class FeedbackRequest(BaseModel):
    chat_id: str
    message_id: str
    rating: str = Field(..., description="'up' or 'down'")
    categories: Optional[List[str]] = None
    comment: Optional[str] = None
    # Client MUST supply prompt/answer when the session is no longer live
    prompt: Optional[str] = None
    answer: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        return v

    @field_validator("categories")
    @classmethod
    def categories_must_be_valid(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        invalid = set(v) - FEEDBACK_TAXONOMY
        if invalid:
            raise ValueError(
                f"Unknown categories: {sorted(invalid)}. "
                f"Valid choices: {sorted(FEEDBACK_TAXONOMY)}"
            )
        return v

    @field_validator("comment")
    @classmethod
    def comment_length(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > COMMENT_MAX_CHARS:
            raise ValueError(
                f"comment must not exceed {COMMENT_MAX_CHARS} characters "
                f"(got {len(v)})"
            )
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping")
async def ping():
    logger.info("************** Ping pong ping pong *************")
    return {"************** Ping pong ping pong *************"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        # Get or create chat session
        chat_id = request.chat_id or str(uuid.uuid4())
        if chat_id not in active_chats:
            logger.info(f"Creating new chat session: {chat_id}")
            model = genai.GenerativeModel(
                MODEL_NAME,
                safety_settings=get_safety_settings()
            )
            # Initialize chat without sending context message
            active_chats[chat_id] = {
                "session": model.start_chat(history=[]),
                "model_name": MODEL_NAME,
                "gen_config": GENERATION_CONFIG,
                "message_ids": [],  # parallel list to session.history (model turns only)
            }

        entry = active_chats[chat_id]
        chat_session = entry["session"]

        # Prepare the prompt with context if provided
        full_prompt = request.prompt
        if request.context:
            full_prompt = f"Context: {request.context}\n\nQuestion: {ISLAMIC_CONTEXT, request.prompt}"

        # Send message and get response
        logger.info("Sending message to chat...")
        response = chat_session.send_message(
            full_prompt,
            generation_config=GENERATION_CONFIG,
        )

        if not response.text:
            logger.error("Empty response received from model")
            raise HTTPException(status_code=500, detail="Empty response from AI model")

        # Assign a stable ID to this model answer
        new_message_id = str(uuid.uuid4())
        entry["message_ids"].append(new_message_id)

        # Build history — model turns carry the message_id that was assigned
        # when they were created.  The message_ids list mirrors the model turns
        # in chat_session.history (every other entry starting at index 1).
        history: List[Message] = []
        model_turn_idx = 0
        for message in chat_session.history:
            try:
                if hasattr(message, 'parts') and message.parts:
                    content = message.parts[0].text if hasattr(message.parts[0], 'text') else str(message.parts[0])
                else:
                    content = str(message)

                is_model = message.role != "user"
                mid: Optional[str] = None
                if is_model:
                    if model_turn_idx < len(entry["message_ids"]):
                        mid = entry["message_ids"][model_turn_idx]
                    model_turn_idx += 1

                history.append(Message(
                    role="user" if message.role == "user" else "model",
                    content=content,
                    message_id=mid,
                ))
            except Exception as e:
                logger.warning(f"Error processing message in history: {str(e)}")
                continue

        logger.info("Chat response generated successfully")
        return ChatResponse(
            response=response.text,
            chat_id=chat_id,
            message_id=new_message_id,
            history=history,
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


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

@app.post("/feedback", status_code=200)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Attach a rating and optional failure categories to a specific model answer.

    The endpoint resolves prompt/answer text from the live session when it is
    still in memory.  If the session has already been evicted (free-tier Render
    restarts frequently), the client MUST supply prompt and answer in the
    request body.  Both paths are documented in the contract below.

    Contract:
      - session alive  → prompt/answer resolved server-side; client values ignored
      - session gone   → client MUST send prompt and answer; 422 returned if missing

    Rate limiting: {RATE_LIMIT_MAX} requests per IP per {RATE_LIMIT_WINDOW_SECONDS}s
    (in-process sliding window — stopgap until issue #9).
    Idempotent: resubmitting for the same (chat_id, message_id) overwrites.
    """
    ip = _client_ip(request)
    if not rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail="Too many feedback submissions. Please wait before trying again.",
        )

    # Resolve prompt/answer snapshot
    prompt_text: Optional[str] = None
    answer_text: Optional[str] = None

    entry = active_chats.get(body.chat_id)
    if entry:
        chat_session = entry["session"]
        model_name = entry["model_name"]
        gen_config = entry["gen_config"]
        # Find the model turn that matches message_id
        message_ids = entry["message_ids"]
        try:
            idx = message_ids.index(body.message_id)
            # history is [user, model, user, model, …]
            # model turn at list index is: history index = 2*idx + 1
            history = chat_session.history
            model_hist_idx = 2 * idx + 1
            if model_hist_idx < len(history):
                m = history[model_hist_idx]
                answer_text = m.parts[0].text if (hasattr(m, 'parts') and m.parts) else str(m)
            user_hist_idx = 2 * idx
            if user_hist_idx < len(history):
                u = history[user_hist_idx]
                prompt_text = u.parts[0].text if (hasattr(u, 'parts') and u.parts) else str(u)
        except (ValueError, IndexError):
            pass  # message_id not found in this session; fall through to client-supplied
    else:
        model_name = MODEL_NAME
        gen_config = GENERATION_CONFIG

    # Fall back to client-supplied text when session is gone
    if prompt_text is None:
        prompt_text = body.prompt
    if answer_text is None:
        answer_text = body.answer

    # If session is not alive and client didn't supply the pair, reject
    if entry is None and (not prompt_text or not answer_text):
        raise HTTPException(
            status_code=422,
            detail=(
                "The chat session is no longer in memory. "
                "Please supply 'prompt' and 'answer' fields in the request body."
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
        model_name=model_name,
        generation_config=gen_config,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        feedback_store.upsert(record)
    except Exception as exc:
        logger.error("Failed to store feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store feedback.")

    logger.info(
        "Feedback stored: chat_id=%s message_id=%s rating=%s",
        body.chat_id, body.message_id, body.rating,
    )
    return {"status": "ok", "feedback_id": record.feedback_id}


# ---------------------------------------------------------------------------
# Admin / maintainer views (protected by ADMIN_TOKEN — stopgap for #9)
# ---------------------------------------------------------------------------

@app.get("/feedback/stats", dependencies=[Depends(require_admin)])
async def feedback_stats():
    """Aggregate quality metrics: rating ratios, per-category, per-model, by day.

    Requires X-Admin-Token header.
    """
    try:
        return feedback_store.stats()
    except Exception as exc:
        logger.error("Failed to fetch feedback stats: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch stats.")


@app.get("/feedback/records", dependencies=[Depends(require_admin)])
async def feedback_records(
    rating: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
):
    """Return recent flagged feedback records, filterable by rating and category.

    Requires X-Admin-Token header.
    Query params: rating=up|down, category=<taxonomy value>, limit=<int>
    """
    if rating and rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")
    if category and category not in FEEDBACK_TAXONOMY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category. Valid: {sorted(FEEDBACK_TAXONOMY)}",
        )
    if not (1 <= limit <= 500):
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    try:
        records = feedback_store.list_records(rating=rating, category=category, limit=limit)
        return {"records": [r.to_dict() for r in records]}
    except Exception as exc:
        logger.error("Failed to fetch feedback records: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch records.")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
