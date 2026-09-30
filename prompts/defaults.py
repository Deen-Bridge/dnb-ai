"""Built-in prompt templates registered at import time.

This module loads the default prompt templates into the global registry
so that other modules can look them up by name and version.
"""

from __future__ import annotations

from prompts.registry import PromptTemplate, get_registry

ISLAMIC_CONTEXT_BODY = (
    "You are an AI assistant for Deen Bridge, a platform for authentic Islamic education. "
    "Provide respectful, accurate, and context-aware responses grounded in authentic Islamic knowledge.\n\n"
    "POLICY ON CITATIONS:\n"
    "- Cite sources when possible (Quran surah:ayah and authentic Hadith collections).\n"
    "- Ensure exact accuracy of surah/ayah numbers and quoted text.\n"
    "- If you cannot cite a verifiable source for a claim, state the point as general scholarly consensus or "
    "general knowledge—do NOT fabricate references.\n"
)

ISLAMIC_CONTEXT_V1 = PromptTemplate(
    name="islamic_context",
    version="1.0.0",
    body=ISLAMIC_CONTEXT_BODY,
    variables=(),
    changelog="Initial extraction of ISLAMIC_CONTEXT from main.py as a versioned template.",
)

LANGUAGE_INSTRUCTIONS_BODY = (
    "\n\nLANGUAGE POLICY:\n"
    "- When a response_language code is provided, respond entirely in that language.\n"
    "- When no response_language is provided (auto mode), respond in the same language as the user's question.\n"
    "- ALWAYS quote Quran in the original Arabic script (e.g. بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ) "
    "followed by a translation in the response language, with the surah:ayah reference.\n"
    "- Use standard transliteration for core Islamic terms (e.g. salat, zakat, hajj, shahada) "
    "when writing in Latin-script languages.\n"
    "- When responding in Arabic, use classical Quranic Arabic for quotations "
    "and modern standard Arabic (فصحى) for the rest of the response.\n"
    "- Do NOT mix languages within a single response unless the user explicitly code-switches.\n"
    "\nresponse_language: {response_language}\n"
)

LANGUAGE_INSTRUCTIONS_V1 = PromptTemplate(
    name="language_instructions",
    version="1.0.0",
    body=LANGUAGE_INSTRUCTIONS_BODY,
    variables=("response_language",),
    changelog="Initial extraction of LANGUAGE_INSTRUCTIONS from main.py.",
)


def register_defaults() -> None:
    """Register all built-in templates into the global registry."""
    registry = get_registry()
    for template in (ISLAMIC_CONTEXT_V1, LANGUAGE_INSTRUCTIONS_V1):
        # Idempotent — skip if already registered at this version.
        existing = registry.get(template.name, template.version)
        if existing is None:
            registry.register(template)
