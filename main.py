from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional
import uuid

from stellar import router as stellar_router
from store import SessionStore, history_to_dicts, dicts_to_contents
from worship import router as worship_router
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="DeenBridge AI API")

# Stellar integration: read-only zakat/balance features on the network
# the rest of the Deen Bridge platform settles on
app.include_router(stellar_router)

# Worship utilities: prayer times and Islamic calendars
app.include_router(worship_router)

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

# Session store (Redis-backed, with in-memory fallback)
session_store = SessionStore()

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


class ChatResponse(BaseModel):
    response: str
    chat_id: str
    history: List[Message]


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
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")

        chat_id = request.chat_id or str(uuid.uuid4())

        # Load persisted history (empty list for new sessions)
        history_dicts = await session_store.load_history(chat_id)
        if history_dicts:
            logger.info("Loaded %d prior turns for chat %s", len(history_dicts) // 2, chat_id)
        else:
            logger.info("Creating new chat session: %s", chat_id)

        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-05-20',
            system_instruction=ISLAMIC_CONTEXT,
            safety_settings=get_safety_settings()
        )

        # Rebuild chat session from persisted history
        contents = dicts_to_contents(history_dicts)
        chat = model.start_chat(history=contents)

        # Prepare the prompt with context if provided
        full_prompt = request.prompt
        if request.context:
            full_prompt = f"Context: {request.context}\n\nQuestion: {request.prompt}"

        # Send message and get response
        logger.info("Sending message to chat...")
        response = chat.send_message(
            full_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        if not response.text:
            logger.error("Empty response received from model")
            raise HTTPException(status_code=500, detail="Empty response from AI model")

        # Persist the updated history (Gemini SDK has appended the new turn)
        await session_store.save_history(chat_id, history_to_dicts(chat.history))

        # Build response history
        history = []
        for message in chat.history:
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
            response=response.text,
            chat_id=chat_id,
            history=history
        )

    except Exception as e:
        error_msg = f"❌ Chat API Error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        existed = await session_store.delete_session(chat_id)
        if existed:
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
