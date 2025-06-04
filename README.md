# DeenBridge AI API

An Islamic AI chat API built with FastAPI and Google's Gemini model, designed to provide accurate and respectful Islamic knowledge and guidance.

## Features

- Islamic context-aware responses
- Conversation history tracking
- Content safety filters
- Source citation support
- Respectful and appropriate responses

## Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd dnb-ai
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

5. Run the server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chat

- `POST /chat`: Start or continue a chat session
- `DELETE /chat/{chat_id}`: Delete a chat session

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key

## Security

- Never commit your `.env` file
- Keep your API keys secure
- Use proper authentication in production

## License

[Your chosen license]
