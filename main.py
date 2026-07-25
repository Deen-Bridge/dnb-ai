import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    InvalidArgument,
    DeadlineExceeded,
    ServiceUnavailable,
)

from verifier import (
    extract_and_verify_all,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Deen Bridge AI Assistant", version="1.0.0")

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CITATION_VERIFY_MODE = os.getenv("CITATION_VERIFY", "annotate").lower()
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

ISLAMIC_CONTEXT = (
    "You are an AI assistant for Deen Bridge, a platform for authentic Islamic education. "
    "Provide respectful, accurate, and context-aware responses grounded in authentic Islamic knowledge.\n\n"
    "POLICY ON CITATIONS:\n"
    "- Cite sources when possible (Quran surah:ayah and authentic Hadith collections).\n"
    "- Ensure exact accuracy of surah/ayah numbers and quoted text.\n"
    "- If you cannot cite a verifiable source for a claim, state the point as general scholarly consensus or "
    "general knowledge—do NOT fabricate references."
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
    message: str
    chat_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    text: str
    chat_id: str
    citations_verified: bool = True
    verification_results: List[CitationVerificationResult] = []


# In-memory session store for demo purposes
sessions: Dict[str, Any] = {}


def get_model():
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=ISLAMIC_CONTEXT,
    )


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
            response = await chat_session.send_message_async(
                message,
                request_options={"timeout": timeout},
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
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    chat_id = request.chat_id or "default"
    if chat_id not in sessions:
        model = get_model()
        sessions[chat_id] = model.start_chat(history=[])

    chat_session = sessions[chat_id]

    try:
        response = await send_message_with_retry(chat_session, request.message)
    except ResourceExhausted as exc:
        logger.warning("Gemini rate limit exceeded for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )
    except InvalidArgument as exc:
        logger.warning("Invalid argument for Gemini call in chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid request parameters.",
        )
    except (DeadlineExceeded, asyncio.TimeoutError) as exc:
        logger.warning("Gemini API call timed out for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=504,
            detail="AI service timed out.",
        )
    except ServiceUnavailable as exc:
        logger.warning("Gemini service unavailable for chat %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /chat handler for session %s: %s", chat_id, exc)
        raise HTTPException(
            status_code=500,
            detail="AI service error",
        )

    response_text = extract_text_safely(response)
    if response_text is None:
        return ChatResponse(
            text="I cannot fulfill this request due to safety guidelines.",
            chat_id=chat_id,
            citations_verified=True,
            verification_results=[],
        )

    # Mode: off -> return verbatim without verification
    if CITATION_VERIFY_MODE == "off":
        return ChatResponse(
            text=response_text,
            chat_id=chat_id,
            citations_verified=True,
            verification_results=[],
        )

    # Verification Step
    verification_results = extract_and_verify_all(response_text)
    mismatches = [
        res for res in verification_results if res.get("status") == VerificationStatus.MISMATCH
    ]

    # Strict Mode: Run single corrective loop if mismatches are found
    if CITATION_VERIFY_MODE == "strict" and mismatches:
        try:
            response_text = await run_strict_corrective_loop(
                chat_session, request.message, response_text, mismatches
            )
        except Exception as exc:
            logger.warning(
                "Strict corrective loop failed: %s; falling back to original response",
                exc,
            )

        # Re-verify updated text
        verification_results = extract_and_verify_all(response_text)
        mismatches = [
            res for res in verification_results if res.get("status") == VerificationStatus.MISMATCH
        ]

    citations_verified = len(mismatches) == 0

    formatted_results = [
        CitationVerificationResult(
            source=res["source"],
            surah=res.get("surah"),
            ayah=res.get("ayah"),
            collection=res.get("collection"),
            number=res.get("number"),
            status=res["status"],
            reason=res.get("reason"),
        )
        for res in verification_results
    ]

    return ChatResponse(
        text=response_text,
        chat_id=chat_id,
        citations_verified=citations_verified,
        verification_results=formatted_results,
    )


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id in sessions:
        del sessions[chat_id]
        return {"status": "success", "message": f"Session {chat_id} deleted."}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/ping")
async def ping():
    return {"status": "ok"}
