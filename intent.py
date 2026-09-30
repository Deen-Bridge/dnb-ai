"""
Question-understanding pipeline: intent classification, clarifying questions,
answer-length calibration, and suggested follow-ups.

Provides a deterministic short-circuit for trivial messages (salam variants,
very short greetings), a Gemini structured-output call for everything else,
per-intent generation-parameter tables, ambiguity detection, and
defensive follow-up parsing.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent Taxonomy
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    GREETING_SMALLTALK = "greeting_smalltalk"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    FIQH_RULING = "fiqh_ruling"
    PERSONAL_GUIDANCE = "personal_guidance"
    PLATFORM_QUESTION = "platform_question"
    OUT_OF_SCOPE = "out_of_scope"


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    intent: Intent
    needs_clarification: bool = False
    clarifying_question: str = ""
    ambiguity_confidence: float = 0.0
    """How likely the question is ambiguous, 0..1. Only meaningful when
    needs_clarification is True."""


# ---------------------------------------------------------------------------
# Deterministic short-circuit — trivial messages are classified without an
# LLM call so the common greeting case adds nearly zero latency.
# ---------------------------------------------------------------------------

# Patterns that match salam / greeting variants
_SALAM_PATTERNS = [
    re.compile(r"^\s*(?:as[sz]ala[mu]|salam|salaam|sallam)",
               re.IGNORECASE),
    re.compile(r"^\s*(?:wa\s*)?(?:'?alaykum|'?aleikum)",
               re.IGNORECASE),
    re.compile(r"^\s*(?:hello|hi\b|hey|greetings|good\s+(?:morning|afternoon|evening|day))",
               re.IGNORECASE),
    re.compile(r"^\s*(?:assalamu[_\s]?alaykum|assalamo[_\s]?alaikum|as-salam)",
               re.IGNORECASE),
]

# Very short messages (<=3 words) that are clearly just greetings
_GREETING_WORDS = {
    "hi", "hello", "hey", "salam", "salaam", "assalamu", "alaykum",
    "assalamo", "alaikum", "as-salam", "marhaba", "ahlan",
    "good", "morning", "afternoon", "evening", "peace",
}

# Short message length threshold — messages at or below this word count
# go through the deterministic check if they match greeting-like patterns.
_SHORT_MSG_THRESHOLD = 5


def _is_trivial_greeting(message: str) -> bool:
    """Return True if *message* is almost certainly a greeting / salam.

    This is intentionally conservative — borderline cases fall through to
    the LLM classifier so we never misclassify a real question as a greeting.
    """
    stripped = message.strip()
    if not stripped:
        return False

    words = stripped.split()
    word_count = len(words)

    # Very short pure-greeting messages
    if word_count <= 3 and words[0].lower().strip("?!.,") in _GREETING_WORDS:
        return True

    # Salam pattern match
    if word_count <= _SHORT_MSG_THRESHOLD:
        for pat in _SALAM_PATTERNS:
            if pat.match(stripped):
                return True

    return False


# ---------------------------------------------------------------------------
# Ambiguity hints — keywords/phrases that suggest the question could have
# multiple valid interpretations.
# ---------------------------------------------------------------------------

_AMBIGUITY_TRIGGERS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:is\s+\w+\s+haram|is\s+\w+\s+halal)\b", re.IGNORECASE),
     "This question depends on context and scholarly school of thought. "
     "Could you specify which madhhab or situation you are asking about?"),
    (re.compile(r"\bwhat\s+breaks\b", re.IGNORECASE),
     "Are you asking about what invalidates wudu, breaks the fast, or "
     "nullifies something else?"),
    (re.compile(r"\b(?:can\s+I|is\s+it\s+permissible)\b", re.IGNORECASE),
     "Could you provide more context about the specific situation you "
     "are asking about?"),
    (re.compile(r"\bhow\s+(?:to|do\s+I)\b", re.IGNORECASE),
     "Which specific act or practice are you asking about?"),
]


def _detect_ambiguity(message: str) -> Tuple[bool, str]:
    """Determine whether *message* seems underspecified / ambiguous.

    Returns (needs_clarification, clarifying_question).
    """
    for pat, question in _AMBIGUITY_TRIGGERS:
        if pat.search(message):
            return True, question
    return False, ""


# Default fallback clarifying question
_FALLBACK_CLARIFY = (
    "Could you please provide more detail so I can give you the most "
    "accurate answer?"
)


# ---------------------------------------------------------------------------
# Classifier — uses Gemini structured output for non-trivial messages.
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = """You are an intent classifier for an Islamic-knowledge AI assistant.
Analyze the user's message and output a JSON object with the following fields:

1. "intent": one of "greeting_smalltalk", "factual_knowledge", "fiqh_ruling",
   "personal_guidance", "platform_question", "out_of_scope"
2. "needs_clarification": true or false — set to true ONLY if the question is
   genuinely ambiguous or underspecified. Be conservative: when in doubt, set false.
3. "clarifying_question": if needs_clarification is true, provide a single short
   clarifying question. Otherwise, set to "".
4. "ambiguity_confidence": a float from 0.0 to 1.0 indicating how ambiguous
   the message is. 0.0 = not ambiguous, 1.0 = very ambiguous.

Intent descriptions:
- greeting_smalltalk: Simple greetings, salam exchanges, casual talk
- factual_knowledge: Questions about aqeedah, seerah, Islamic definitions, history
- fiqh_ruling: Questions about halal/haram, rulings, jurisprudence
- personal_guidance: Emotional/spiritual support, du'a requests, personal advice
- platform_question: Questions about the Deen Bridge platform, courses, zakat feature
- out_of_scope: Non-Islamic topics, harmful requests, nonsense

Respond with ONLY a valid JSON object, no other text.

User message: {message}"""

# Threshold for "would adding the LLM call be worthwhile"? If the message
# passes the deterministic greeting check, we skip LLM entirely.
# Default ambiguity threshold — needs to be high to avoid over-clarifying.
AMBIGUITY_THRESHOLD = float(os.getenv("AMBIGUITY_THRESHOLD", "0.8"))


def classify_intent(message: str, model_callable=None) -> ClassificationResult:
    """Classify *message* into an Intent.

    *model_callable* should be a callable that accepts a prompt string and
    returns a response object with a ``.text`` attribute (e.g. a
    ``genai.GenerativeModel.generate_content`` method).  If ``None``, the
    function still works for trivial messages but will raise for non-trivial
    ones (useful for testing the short-circuit path).
    """
    # 1. Deterministic short-circuit for trivial greetings
    if _is_trivial_greeting(message):
        return ClassificationResult(
            intent=Intent.GREETING_SMALLTALK,
        )

    # 2. LLM-based classification
    return _llm_classify(message, model_callable)


def _llm_classify(message: str, model_callable) -> ClassificationResult:
    """Call the LLM to classify *message*."""
    prompt = _CLASSIFIER_PROMPT.format(message=message)

    if model_callable is None:
        raise RuntimeError(
            "classify_intent requires a model_callable for non-trivial messages."
        )

    try:
        response = model_callable(prompt)
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            # Find the first ``` and the closing ```
            start = raw.find("\n") + 1 if "\n" in raw else raw.find("```") + 3
            end = raw.rfind("```")
            if end > start:
                raw = raw[start:end].strip()

        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError, ValueError) as exc:
        logger.warning("Intent classifier returned unparseable output: %s", exc)
        return ClassificationResult(
            intent=Intent.FACTUAL_KNOWLEDGE,  # safe fallback
        )

    # Extract and validate intent
    intent_str = data.get("intent", "factual_knowledge")
    try:
        intent = Intent(intent_str)
    except ValueError:
        logger.warning("Unknown intent '%s'; falling back to factual_knowledge", intent_str)
        intent = Intent.FACTUAL_KNOWLEDGE

    # Ambiguity
    needs_clarification = data.get("needs_clarification", False)
    clarifying_question = data.get("clarifying_question", "")
    ambiguity_confidence = float(data.get("ambiguity_confidence", 0.0))

    # Apply threshold — if confidence is below threshold, don't clarify
    if ambiguity_confidence < AMBIGUITY_THRESHOLD:
        needs_clarification = False
        clarifying_question = ""

    # Fallback if needs_clarification is True but no question provided
    if needs_clarification and not clarifying_question:
        clarifying_question = _FALLBACK_CLARIFY

    return ClassificationResult(
        intent=intent,
        needs_clarification=needs_clarification,
        clarifying_question=clarifying_question,
        ambiguity_confidence=ambiguity_confidence,
    )


# ---------------------------------------------------------------------------
# Per-intent generation parameters and instruction snippets
# ---------------------------------------------------------------------------


@dataclass
class IntentConfig:
    instruction_snippet: str
    """Additional instruction appended to ISLAMIC_CONTEXT for this intent."""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048
    """Default generation parameters, all env-overridable per intent."""

    def effective_params(self) -> Dict[str, Any]:
        """Return generation config dict, respecting env overrides."""
        return {
            "temperature": float(os.getenv(
                f"TEMP_{self.max_output_tokens}",
                str(self.temperature),
            )),
            "top_p": float(os.getenv(
                f"TOP_P_{self.max_output_tokens}",
                str(self.top_p),
            )),
            "top_k": int(os.getenv(
                f"TOP_K_{self.max_output_tokens}",
                str(self.top_k),
            )),
            "max_output_tokens": int(os.getenv(
                f"MAX_TOKENS_{self.max_output_tokens}",
                str(self.max_output_tokens),
            )),
        }


# Default per-intent config table
INTENT_CONFIGS: Dict[Intent, IntentConfig] = {
    Intent.GREETING_SMALLTALK: IntentConfig(
        instruction_snippet=(
            "This is a greeting or casual small-talk. Respond warmly in "
            "1-2 sentences, return the salam/salutation properly, and "
            "invite the user to ask an Islamic knowledge question."
        ),
        temperature=0.5,
        max_output_tokens=256,
    ),
    Intent.FACTUAL_KNOWLEDGE: IntentConfig(
        instruction_snippet=(
            "Provide a structured, thorough answer grounded in authentic "
            "Islamic sources. Organize the answer clearly with headings "
            "or sections where appropriate. Cite Quran surah:ayah and "
            "authentic Hadith where possible. Follow with 2-3 suggested "
            "follow-up questions in a delimited block."
        ),
        temperature=0.7,
        max_output_tokens=2048,
    ),
    Intent.FIQH_RULING: IntentConfig(
        instruction_snippet=(
            "Provide a fiqh ruling based on authentic Islamic sources. "
            "Acknowledge differences of scholarly opinion where relevant. "
            "Be balanced and cite sources. Follow with 2-3 suggested "
            "follow-up questions in a delimited block."
        ),
        temperature=0.6,
        max_output_tokens=2048,
    ),
    Intent.PERSONAL_GUIDANCE: IntentConfig(
        instruction_snippet=(
            "Respond with a compassionate, supportive tone. Make du'a "
            "for the user where fitting. For serious personal matters, "
            "gently recommend consulting a qualified scholar or "
            "professional. Keep the answer concise and comforting."
        ),
        temperature=0.8,
        max_output_tokens=1024,
    ),
    Intent.PLATFORM_QUESTION: IntentConfig(
        instruction_snippet=(
            "Answer concisely and practically about the Deen Bridge "
            "platform, its courses, features, or the zakat calculator. "
            "Keep it brief and actionable."
        ),
        temperature=0.5,
        max_output_tokens=1024,
    ),
    Intent.OUT_OF_SCOPE: IntentConfig(
        instruction_snippet=(
            "Politely explain that this question is outside your scope "
            "as an Islamic-knowledge assistant. If the question seems "
            "harmful or inappropriate, gently redirect. Do not attempt "
            "to answer the question."
        ),
        temperature=0.5,
        max_output_tokens=512,
    ),
}


def get_intent_config(intent: Intent) -> IntentConfig:
    """Return the generation config for *intent*, falling back to factual_knowledge."""
    return INTENT_CONFIGS.get(intent, INTENT_CONFIGS[Intent.FACTUAL_KNOWLEDGE])


# ---------------------------------------------------------------------------
# Suggested follow-ups — parse a delimited block from the model response
# ---------------------------------------------------------------------------

# Delimiter used by the model to wrap follow-up suggestions.
# Using a distinctive marker unlikely to appear in natural prose.
FOLLOWUP_START = "<!-- FOLLOWUPS -->"
FOLLOWUP_END = "<!-- /FOLLOWUPS -->"
# Alternative delimiter (XML-style) for robustness
FOLLOWUP_START_ALT = "[[FOLLOWUPS]]"
FOLLOWUP_END_ALT = "[[/FOLLOWUPS]]"


def parse_followups(response_text: str) -> List[str]:
    """Extract 2-3 suggested follow-up questions from *response_text*.

    Returns an empty list if parsing fails — never raises.
    """
    if not response_text:
        return []

    text = response_text

    # Try primary delimiter
    followups = _extract_delimited_block(text, FOLLOWUP_START, FOLLOWUP_END)

    # Try alternative delimiter
    if not followups:
        followups = _extract_delimited_block(text, FOLLOWUP_START_ALT, FOLLOWUP_END_ALT)

    return followups


def _extract_delimited_block(text: str, start_delim: str, end_delim: str) -> List[str]:
    """Extract and parse a delimited block of follow-up questions."""
    start_idx = text.find(start_delim)
    if start_idx == -1:
        return []

    end_idx = text.find(end_delim, start_idx + len(start_delim))
    if end_idx == -1:
        return []

    block = text[start_idx + len(start_delim):end_idx].strip()
    if not block:
        return []

    # Split by newlines and extract list items
    items = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering, bullets, quotes
        cleaned = _clean_followup_line(line)
        if cleaned:
            items.append(cleaned)

    # Return at most 5 items
    return items[:5]


def _clean_followup_line(line: str) -> str:
    """Clean a single follow-up line: strip numbering, bullets, quotes."""
    # Remove leading numbering like "1.", "2)", "1."
    line = re.sub(r'^\s*\d+[.)]\s*', '', line).strip()
    # Remove leading bullets (including Unicode bullet \u2022)
    line = re.sub(r'^\s*[\*\-\u2022]\s*', '', line).strip()
    # Remove surrounding quotes
    line = re.sub(r'^["\'](.*)["\']$', r'\1', line).strip()
    return line


def strip_followup_block(response_text: str) -> str:
    """Remove the follow-up delimited block from *response_text* so it never
    leaks into the visible response.
    """
    text = response_text

    # Primary delimiter
    text = _remove_block(text, FOLLOWUP_START, FOLLOWUP_END)
    # Alternative delimiter
    text = _remove_block(text, FOLLOWUP_START_ALT, FOLLOWUP_END_ALT)

    return text.strip()


def _remove_block(text: str, start_delim: str, end_delim: str) -> str:
    start_idx = text.find(start_delim)
    if start_idx == -1:
        return text

    end_idx = text.find(end_delim, start_idx + len(start_delim))
    if end_idx == -1:
        # No closing delimiter — remove from start to end
        return text[:start_idx].strip()

    # Remove the entire block including delimiters
    return text[:start_idx].strip() + text[end_idx + len(end_delim):]


# ---------------------------------------------------------------------------
# No-double-clarification guard
# ---------------------------------------------------------------------------


def should_clarify(
    classification: ClassificationResult,
    last_classification: Optional[ClassificationResult],
) -> bool:
    """Return True if the assistant should ask a clarifying question.

    Never clarifies twice in a row for the same session: if the *last*
    classification already had ``needs_clarification`` set, this returns
    False so the assistant answers the follow-up reply directly.
    """
    if not classification.needs_clarification:
        return False

    # Don't clarify twice in a row
    if last_classification and last_classification.needs_clarification:
        return False

    return True
