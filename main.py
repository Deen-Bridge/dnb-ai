from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional
import uuid

from stellar import router as stellar_router
from safety import InputGate, OutputCheck, SafetyPipeline, load_policy
from study import router as study_router

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
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        chat_id = request.chat_id or str(uuid.uuid4())

        def generate(safety_prompt: str) -> str:
            if chat_id not in active_chats:
                logger.info(f"Creating new chat session: {chat_id}")
                model = genai.GenerativeModel(
                    'gemini-2.5-flash-preview-05-20',
                    safety_settings=get_safety_settings()
                )
                active_chats[chat_id] = model.start_chat(history=[])

            context = f"Additional context: {request.context}\n\n" if request.context else ""
            full_prompt = f"{ISLAMIC_CONTEXT}\n{context}User question: {safety_prompt}"
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

        logger.info("Chat response generated successfully")
        return ChatResponse(
            response=safety_result.text if safety_result else generated_text,
            chat_id=chat_id,
            history=history,
            moderation=Moderation(
                category_id=safety_result.category_id,
                action=safety_result.action,
            ) if safety_result and safety_result.category_id else None,
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
