import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

from verifier import (
    extract_and_verify_all,
    VerificationStatus,
)

app = FastAPI(title="Deen Bridge AI Assistant", version="1.0.0")

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CITATION_VERIFY_MODE = os.getenv("CITATION_VERIFY", "annotate").lower()

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

    corrective_response = chat_session.send_message(correction_prompt)
    return corrective_response.text


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    chat_id = request.chat_id or "default"
    if chat_id not in sessions:
        model = get_model()
        sessions[chat_id] = model.start_chat(history=[])

    chat_session = sessions[chat_id]
    response = chat_session.send_message(request.message)
    response_text = response.text

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
        response_text = await run_strict_corrective_loop(
            chat_session, request.message, response_text, mismatches
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
