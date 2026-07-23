"""Conversation history budget management.

Provides token estimation and history trimming logic that operates on
Gemini chat session history lists — no Gemini SDK dependency needed.
"""

import os
import logging


logger = logging.getLogger(__name__)

MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "16000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "50"))


def estimate_tokens(text: str) -> int:
    """Cheap local token estimate — ~4 chars per token on average."""
    return max(1, len(text) // 4)


def trim_history(chat_session) -> bool:
    """Enforce token budget and turn cap on chat history.

    Drops oldest turn-pairs (user + model) until both budgets are satisfied.
    Returns True if any turns were dropped.
    """
    if not chat_session or not chat_session.history:
        return False

    original_len = len(chat_session.history)
    truncated = False

    while len(chat_session.history) > MAX_HISTORY_TURNS * 2:
        chat_session.history.pop(0)
        chat_session.history.pop(0)
        truncated = True

    while len(chat_session.history) >= 2:
        total = sum(
            estimate_tokens(m.parts[0].text)
            if hasattr(m, "parts") and m.parts and hasattr(m.parts[0], "text")
            else 0
            for m in chat_session.history
        )
        if total <= MAX_HISTORY_TOKENS:
            break
        chat_session.history.pop(0)
        chat_session.history.pop(0)
        truncated = True

    if truncated:
        turns_dropped = (original_len - len(chat_session.history)) // 2
        logger.info("Trimmed chat history: dropped %d turn pair(s)", turns_dropped)

    return truncated
