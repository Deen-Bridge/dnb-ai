from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="DeenBridge AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://deenbridge.vercel.app",  # Production frontend
        "https://*.vercel.app",  # Vercel preview deployments
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.prompt[:100]}...")
        
        # Get or create chat session
        chat_id = request.chat_id or str(uuid.uuid4())
        if chat_id not in active_chats:
            logger.info(f"Creating new chat session: {chat_id}")
            model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20',
                safety_settings=get_safety_settings()
            )
            # Initialize chat with Islamic context
            active_chats[chat_id] = model.start_chat(history=[])
            # Send initial context message
            active_chats[chat_id].send_message(ISLAMIC_CONTEXT)
        
        chat = active_chats[chat_id]
        
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
        
        # Get chat history (excluding the initial context message)
        history = []
        for message in chat.history[1:]:  # Skip the initial context message
            history.append(Message(
                role="user" if message.role == "user" else "model",
                content=message.parts[0].text
            ))
        
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

