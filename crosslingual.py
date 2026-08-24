"""Arabic–English cross-lingual retrieval (#232).

Understands Arabic-script, romanized and code-switched Islamic queries and
retrieves from a bilingual document collection in one normalized vector space.

Design
------
* Language handling is pure Unicode-range inspection — no model calls.
* ``data/transliteration_glossary.json`` bridges romanized spellings (ALA-LC,
  DIN, Hunterian-style) to canonical Arabic terms; it powers term protection
  during translation and the offline cross-script embedding bridge.
* Query translation uses Gemini when a real key is configured and degrades to
  glossary-only substitution otherwise; every result carries its
  ``translation_source`` so callers can see which mode produced the text.
* Embeddings implement :class:`MultilingualEmbedder`: a Gemini adapter when
  configured, else a deterministic script-aware character-n-gram hashing
  embedder that gains cross-script affinity through the glossary bridge.
* Default paths are fully offline-capable — CI has no secrets.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, Protocol

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "CrosslingualIndex",
    "CrosslingualSearchRequest",
    "CrosslingualSearchResponse",
    "Document",
    "GeminiEmbedder",
    "GlossaryTerm",
    "HashingCrossScriptEmbedder",
    "MultilingualEmbedder",
    "QueryTranslation",
    "ScriptDetection",
    "TermMatch",
    "TransliterationGlossary",
    "crosslingual_search",
    "detect_script",
    "gemini_configured",
    "get_embedder",
    "get_glossary",
    "normalize_arabic_token",
    "normalize_english_text",
    "translate_query",
]

Lang = Literal["ar", "en", "mixed"]
LangPref = Literal["any", "ar", "en"]
TranslationSource = Literal["gemini", "glossary", "none"]

GLOSSARY_PATH = Path(__file__).parent / "data" / "transliteration_glossary.json"
DEFAULT_CORPUS_PATH = Path(__file__).parent / "data" / "quran_uthmani.json"

# Keys shaped like placeholders (CI exports GEMINI_API_KEY=dummy) must never
# trigger network attempts: default paths have to stay fully offline-capable.
_PLACEHOLDER_KEYS = frozenset(
    {"", "dummy", "test", "test-key", "testing", "changeme", "your_api_key_here", "none"}
)
_EMBEDDER_MODE_ENV = "CROSS_LINGUAL_EMBEDDER"

_SNIPPET_CHARS = 240


def gemini_configured() -> bool:
    """True only when a real-looking Gemini API key is present."""
    return os.getenv("GEMINI_API_KEY", "").strip().casefold() not in _PLACEHOLDER_KEYS


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


class TokenLang(BaseModel):
    token: str
    lang: Lang | None = None  # None for digits/symbols


class ScriptDetection(BaseModel):
    lang: Lang
    token_langs: list[TokenLang]


def _char_is_arabic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def _token_has_arabic(token: str) -> bool:
    return any(_char_is_arabic(ch) for ch in token)


def detect_script(text: str) -> ScriptDetection:
    """Classify *text* per token; ``mixed`` when both scripts appear.

    Tokens made only of digits/symbols get ``lang=None``. A letterless query
    falls back to ``"en"`` (the Latin default).
    """
    token_langs: list[TokenLang] = []
    ar_count = 0
    en_count = 0
    for raw in text.split():
        if _token_has_arabic(raw):
            token_langs.append(TokenLang(token=raw, lang="ar"))
            ar_count += 1
        elif any(ch.isascii() and ch.isalpha() for ch in raw):
            token_langs.append(TokenLang(token=raw, lang="en"))
            en_count += 1
        else:
            token_langs.append(TokenLang(token=raw, lang=None))
    if ar_count and en_count:
        lang: Lang = "mixed"
    elif ar_count:
        lang = "ar"
    else:
        lang = "en"
    return ScriptDetection(lang=lang, token_langs=token_langs)


# ---------------------------------------------------------------------------
# Light-touch morphology normalization
#
# Deliberately small heuristics for matching robustness only — never shown to
# users. Arabic: strip harakat/tatweel, fold alef/teh-marbuta/alef-maksura,
# then strip up to two common clitic prefixes (longest first), keeping a
# remainder of at least two letters so short words are never destroyed.
# English: casefold + punctuation strip + a tiny stopword list used only when
# building embedding features.
# ---------------------------------------------------------------------------

# Tatweel (0640), harakat (064B-065F), dagger alif (0670).
_AR_MARKS_RE = re.compile("[\u0640\u064B-\u065F\u0670]")


_AR_NORMALIZE_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})

# Ordered longest-first so وال/بال/... win over their bare-clitic prefixes.
_AR_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ب", "ف", "ك", "ل")

_EN_PUNCT = ".,!?;:\"'()[]{}<>/\\|`~@#$%^&*_+=«»“”‘’…،؛؟"
_EN_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
        "do", "does", "did", "to", "of", "in", "on", "for", "and", "or",
        "i", "you", "my", "me", "it", "its", "that", "this", "these", "those",
    }
)


def normalize_arabic_token(token: str) -> str:
    """Diacritic-fold and clitic-strip one Arabic token."""
    t = _AR_MARKS_RE.sub("", token.translate(_AR_NORMALIZE_TABLE))
    changed = True
    while changed:
        changed = False
        for prefix in _AR_PREFIXES:
            if t.startswith(prefix) and len(t) - len(prefix) >= 2:
                t = t[len(prefix):]
                changed = True
                break
    return t


def normalize_english_text(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercase, punctuation-strip and optionally stopword-filter English."""
    tokens = [chunk.strip(_EN_PUNCT).casefold() for chunk in text.split()]
    tokens = [t for t in tokens if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _EN_STOPWORDS]
    return tokens


def _match_normalize(s: str) -> str:
    """Fold case, apostrophe variants and Latin diacritics for glossary matching.

    Length may differ from the input (NFD combining marks are dropped); all
    downstream matching operates on this normalized form, while user-visible
    fields keep the original string.
    """
    s = s.casefold()
    s = s.replace("’", "'").replace("ʼ", "'")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------------------
# Transliteration glossary
# ---------------------------------------------------------------------------


class GlossaryTerm(BaseModel):
    id: str
    ar: str
    en: str
    category: str = "general"
    variants: list[str] = Field(default_factory=list)


class TermMatch(BaseModel):
    term_id: str
    matched_text: str
    canonical_ar: str
    canonical_en: str
    start: int
    end: int

    @property
    def note(self) -> str:
        return f"'{self.matched_text}' ↔ '{self.canonical_ar}' ({self.canonical_en})"


class TransliterationGlossary:
    """Loader and lookup helpers over the bundled transliteration glossary."""

    def __init__(self, data_file: Path = GLOSSARY_PATH):
        self.data_file = data_file
        self.terms: list[GlossaryTerm] = []
        self._by_variant: dict[str, GlossaryTerm] = {}
        self._patterns: list[tuple[re.Pattern[str], GlossaryTerm]] = []
        self._load()

    def _load(self) -> None:
        if not self.data_file.exists():
            logger.warning("Transliteration glossary missing at %s; bridging disabled", self.data_file)
            return
        with open(self.data_file, encoding="utf-8") as f:
            payload = json.load(f)
        for entry in payload.get("terms", []):
            term = GlossaryTerm(**entry)
            self.terms.append(term)
            # The canonical Arabic form itself acts as an implicit variant so
            # Arabic-side text also fires the bridge.
            for variant in {term.ar, *term.variants}:
                key = _match_normalize(variant)
                self._by_variant.setdefault(key, term)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        # Longer keys first so e.g. "surah al-fatiha" wins over bare "fatiha".
        for key, term in sorted(self._by_variant.items(), key=lambda kv: -len(kv[0])):
            pattern = self._pattern_for(key, arabic_script=_token_has_arabic(key))
            self._patterns.append((re.compile(pattern), term))

    @staticmethod
    def _pattern_for(key: str, *, arabic_script: bool) -> str:
        escaped = re.escape(key)
        # Apostrophes people type or omit: bidah / bid'ah / bidiʼah …
        escaped = escaped.replace("'", "['ʼ’]?")
        # Spaces and hyphens are interchangeable separators in phrases. Some
        # Pythons regex-escape them ("\ ", "\-"), others leave them raw —
        # consume the escaped forms first so no stray backslash survives.
        sep = "\x00"
        escaped = (
            escaped.replace(r"\ ", sep)
            .replace(r"\-", sep)
            .replace(" ", sep)
            .replace("-", sep)
            .replace(sep, r"[\s\-]+")
        )
        if arabic_script:
            # Tolerate clitics such as ال، وال، بال before an Arabic term.
            prefix = r"(?:وال|بال|فال|كال|لل|ال|[وفبكل])?"
            return f"(?<![A-Za-z0-9\\u0600-\\u06FF]){prefix}{escaped}(?![A-Za-z\\u0600-\\u06FF])"
        return f"(?<![A-Za-z0-9]){escaped}(?![A-Za-z])"

    def __len__(self) -> int:
        return len(self.terms)

    def lookup(self, surface: str) -> GlossaryTerm | None:
        """Exact (normalized) lookup of one romanized or Arabic surface form."""
        return self._by_variant.get(_match_normalize(surface.strip()))

    def find_terms(self, text: str) -> list[TermMatch]:
        """Find glossary terms in *text*, longest-match first, non-overlapping.

        Spans refer to ``_match_normalize(text)``.
        """
        working = _match_normalize(text)
        candidates: list[TermMatch] = []
        for pattern, term in self._patterns:
            for m in pattern.finditer(working):
                matched = m.group(0)
                # Skip zero-width safety (an optional-apostrophe edge cannot
                # produce one because every key has at least one literal char).
                if not matched:
                    continue
                candidates.append(
                    TermMatch(
                        term_id=term.id,
                        matched_text=matched,
                        canonical_ar=term.ar,
                        canonical_en=term.en,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        selected: list[TermMatch] = []
        taken: list[tuple[int, int]] = []
        for cand in sorted(candidates, key=lambda c: (-(c.end - c.start), c.start)):
            if any(cand.start < end and cand.end > start for start, end in taken):
                continue
            selected.append(cand)
            taken.append((cand.start, cand.end))
        selected.sort(key=lambda c: c.start)
        return selected


_glossary: TransliterationGlossary | None = None


def get_glossary() -> TransliterationGlossary:
    global _glossary
    if _glossary is None:
        _glossary = TransliterationGlossary()
    return _glossary


# ---------------------------------------------------------------------------
# Query translation adapter
# ---------------------------------------------------------------------------


class TermProtection(BaseModel):
    matched_text: str
    canonical_ar: str
    canonical_en: str


class QueryTranslation(BaseModel):
    source_lang: Lang
    target_lang: Literal["ar", "en"]
    translated_text: str
    translation_source: TranslationSource
    protected_terms: list[TermProtection] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _translation_target(detection: ScriptDetection, override: Literal["ar", "en"] | None) -> Literal["ar", "en"]:
    if override is not None:
        return override
    if detection.lang == "ar":
        return "en"
    if detection.lang == "en":
        return "ar"
    # Code-switched: mirror toward the script the query leans away from.
    ar_n = sum(1 for t in detection.token_langs if t.lang == "ar")
    en_n = sum(1 for t in detection.token_langs if t.lang == "en")
    return "en" if ar_n >= en_n else "ar"


def _build_translation_prompt(text: str, target_lang: Literal["ar", "en"], protections: list[TermProtection]) -> str:
    lang_name = "Arabic" if target_lang == "ar" else "English"
    lines = [
        "You are a careful translator for Islamic questions.",
        "Preserve Islamic terminology precisely; never paraphrase core fiqh or Qur'anic terms loosely.",
    ]
    if protections:
        rendered = "; ".join(f"{p.canonical_ar} ({p.canonical_en})" for p in protections)
        lines.append(f"Render these terms exactly in their canonical Arabic form: {rendered}.")
    lines += [f"Translate the following query into {lang_name}.", "Reply with ONLY the translation.", "", text]
    return "\n".join(lines)


def _gemini_translate_sync(text: str, target_lang: Literal["ar", "en"], protections: list[TermProtection]) -> str:
    """Blocking Gemini call; raises on any failure so callers can fall back."""
    import google.generativeai as genai

    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "gemini-1.5-flash"))
    timeout = int(os.getenv("GEMINI_TIMEOUT", "30"))
    response = model.generate_content(
        _build_translation_prompt(text, target_lang, protections),
        generation_config={"temperature": 0.0, "max_output_tokens": 256},
        request_options={"timeout": timeout},
    )
    translated = (getattr(response, "text", "") or "").strip()
    if not translated:
        raise ValueError("empty translation response")
    return translated


def apply_term_protection(text: str, matches: Iterable[TermMatch], target_lang: Literal["ar", "en"]) -> tuple[str, list[TermProtection]]:
    """Substitute matched glossary surfaces with their canonical form.

    Toward Arabic the romanization is replaced by the canonical Arabic term;
    toward English by the canonical English gloss. Returns the rewritten text
    plus the protections applied (in order of appearance).
    """
    protections: list[TermProtection] = []
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor:m.start])
        out.append(m.canonical_ar if target_lang == "ar" else m.canonical_en)
        cursor = m.end
        protections.append(TermProtection(matched_text=m.matched_text, canonical_ar=m.canonical_ar, canonical_en=m.canonical_en))
    out.append(text[cursor:])
    return "".join(out), protections


async def translate_query(
    query: str,
    *,
    target_lang: Literal["ar", "en"] | None = None,
    enabled: bool = True,
) -> QueryTranslation:
    """Translate a query across the Arabic–English pair.

    Tries Gemini when configured; otherwise substitutes known glossary terms
    and passes everything else through untouched. The mode is reported in
    ``translation_source`` so results can be flagged honestly.
    """
    detection = detect_script(query)
    src = detection.lang
    tgt = _translation_target(detection, target_lang)
    base_notes: list[str] = []

    matches = get_glossary().find_terms(query)

    if src == tgt:
        base_notes.append("query already in target language; no translation needed")
        return QueryTranslation(
            source_lang=src,
            target_lang=tgt,
            translated_text=query,
            translation_source="none",
            protected_terms=[],
            notes=base_notes,
        )

    if enabled and gemini_configured():
        protections = [
            TermProtection(matched_text=m.matched_text, canonical_ar=m.canonical_ar, canonical_en=m.canonical_en)
            for m in matches
        ]
        try:
            translated = await asyncio.to_thread(_gemini_translate_sync, query, tgt, protections)
            return QueryTranslation(
                source_lang=src,
                target_lang=tgt,
                translated_text=translated,
                translation_source="gemini",
                protected_terms=protections,
                notes=["Islamic terminology protected via glossary during translation"] if protections else [],
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully to offline path
            logger.warning("Gemini translation failed (%s); falling back to glossary-only", exc)
            base_notes.append(f"Gemini unavailable ({type(exc).__name__}); used glossary fallback")

    # TermMatch spans refer to the match-normalized string, so substitution
    # runs on that same form (translation output does not preserve casing).
    working_query = _match_normalize(query)
    substituted, protections = apply_term_protection(working_query, matches, tgt)
    notes = [*base_notes]
    if substituted != query:
        notes.append("glossary-only substitution: unmatched words passed through unchanged")
        source: TranslationSource = "glossary"
    else:
        source = "none"
        if not matches:
            notes.append("no glossary terms recognized; passthrough without translation")
    return QueryTranslation(
        source_lang=src,
        target_lang=tgt,
        translated_text=substituted,
        translation_source=source,
        protected_terms=protections,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multilingual embeddings
# ---------------------------------------------------------------------------


class MultilingualEmbedder(Protocol):
    """Anything that maps text into the shared L2-normalized vector space."""

    def embed(self, text: str) -> np.ndarray: ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


class HashingCrossScriptEmbedder:
    """Deterministic offline embedder with glossary-driven cross-script affinity.

    Features are hashed character n-grams (n=3,4) plus whole-word tokens.
    Whenever the glossary recognizes a term — in either script — the canonical
    Arabic *and* English forms contribute features too, so ``"salaat"``,
    ``"صلاة"`` and ``"prayer"`` land near each other without any model call.
    Vectors are signed-hashed into ``dim`` buckets and L2-normalized; hashing
    uses blake2b so results are stable across processes (unlike builtin hash).
    """

    def __init__(self, dim: int = 256, ngram_sizes: tuple[int, ...] = (3, 4)) -> None:
        self.dim = dim
        self.ngram_sizes = ngram_sizes

    # -- internals ----------------------------------------------------------

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:6], "big") % self.dim
        sign = 1.0 if digest[6] & 1 else -1.0
        return index, sign

    def _add_word_grams(self, features: dict[int, float], word: str, weight: float) -> None:
        padded = f"#{word}#"
        idx, _ = self._hash_feature(f"w:{word}")
        features[idx] = features.get(idx, 0.0) + weight
        for size in self.ngram_sizes:
            if len(padded) < size:
                continue
            for i in range(len(padded) - size + 1):
                gram = f"g{size}:{padded[i:i + size]}"
                idx, sign = self._hash_feature(gram)
                features[idx] = features.get(idx, 0.0) + weight * sign

    def _extract_features(self, text: str) -> dict[int, float]:
        features: dict[int, float] = {}
        working = _match_normalize(text)
        for token in working.split():
            if _token_has_arabic(token):
                normalized = normalize_arabic_token(token)
                if len(normalized) >= 2:
                    self._add_word_grams(features, normalized, 2.0)
            else:
                stripped = token.strip(_EN_PUNCT)
                if stripped and stripped not in _EN_STOPWORDS:
                    self._add_word_grams(features, stripped, 2.0)
        # Glossary bridge: canonical AR + EN forms of every recognized term
        # enter the same feature space regardless of the query's script.
        for match in get_glossary().find_terms(text):
            for token in normalize_arabic_token(_match_normalize(match.canonical_ar)).split():
                self._add_word_grams(features, token, 2.5)
            for token in normalize_english_text(match.canonical_en, drop_stopwords=True):
                self._add_word_grams(features, token, 2.0)
        return features

    # -- public API ---------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for idx, weight in self._extract_features(text).items():
            vec[idx] += weight
        return l2_normalize(vec)


class GeminiEmbedder:
    """Gemini text-embedding adapter (network on first use, not at init)."""

    MODEL = "models/text-embedding-004"

    def __init__(self) -> None:
        self.dim: int | None = None

    def embed(self, text: str) -> np.ndarray:
        import google.generativeai as genai

        result = genai.embed_content(model=self.MODEL, content=text)
        vec = np.asarray(result["embedding"], dtype=np.float32)
        self.dim = int(vec.shape[0])
        return l2_normalize(vec)


def get_embedder() -> MultilingualEmbedder:
    """Pick the active embedder: explicit env override > Gemini > offline."""
    mode = os.getenv(_EMBEDDER_MODE_ENV, "auto").strip().casefold()
    if mode in {"hashing", "offline"}:
        return HashingCrossScriptEmbedder()
    if gemini_configured():
        if mode in {"auto", "gemini"}:
            return GeminiEmbedder()
        logger.warning("Unknown %s=%r; using offline embedder", _EMBEDDER_MODE_ENV, mode)
    logger.info("No usable Gemini key; using offline cross-script embedder")
    return HashingCrossScriptEmbedder()


# ---------------------------------------------------------------------------
# Document collection and index
# ---------------------------------------------------------------------------


class Document(BaseModel):
    doc_id: str
    ar: str | None = None
    en: str | None = None


class _Side(NamedTuple):
    doc_index: int
    lang: Literal["ar", "en"]
    vector_index: int
    text: str


def load_default_documents(data_file: Path = DEFAULT_CORPUS_PATH) -> list[Document]:
    """Bilingual ayat from the bundled corpus; empty collection if absent."""
    docs: list[Document] = []
    if not data_file.exists():
        logger.warning("Default cross-lingual corpus missing at %s", data_file)
        return docs
    with open(data_file, encoding="utf-8") as f:
        payload = json.load(f)
    for key, entry in payload.get("ayat", {}).items():
        ar = entry.get("arabic") or None
        en = entry.get("english") or None
        if ar or en:
            docs.append(Document(doc_id=f"quran:{key}", ar=ar, en=en))
    return docs


class CrosslingualIndex:
    """Embeds each language side of every document once, lazily."""

    def __init__(self, documents: Iterable[Document], embedder: MultilingualEmbedder | None = None):
        self.documents = list(documents)
        self.embedder = embedder or get_embedder()
        self._matrix: np.ndarray | None = None
        self._sides: list[_Side] = []

    def _build(self) -> None:
        vectors: list[np.ndarray] = []
        sides: list[_Side] = []
        for doc_index, doc in enumerate(self.documents):
            sides_of_doc: list[tuple[Literal["ar", "en"], str]] = [("ar", doc.ar)] if doc.ar else []
            if doc.en:
                sides_of_doc.append(("en", doc.en))
            for lang, text in sides_of_doc:
                sides.append(_Side(doc_index, lang, len(vectors), text))
                vectors.append(self.embedder.embed(text))
        self._matrix = np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        self._sides = sides

    def _ensure_built(self) -> None:
        if self._matrix is None:
            self._build()

    def search_vectors(
        self,
        query_vec: np.ndarray,
        k: int,
        lang_pref: LangPref = "any",
    ) -> list[tuple[Document, _Side, float]]:
        """Best side per document under the language preference, top-*k*."""
        self._ensure_built()
        assert self._matrix is not None
        if self._matrix.shape[0] == 0:
            return []
        scores = self._matrix @ query_vec
        ranked = sorted(range(len(self._sides)), key=lambda i: float(scores[i]), reverse=True)
        results: list[tuple[Document, _Side, float]] = []
        seen_docs: set[int] = set()
        for side_index in ranked:
            side = self._sides[side_index]
            if lang_pref != "any" and side.lang != lang_pref:
                continue
            if side.doc_index in seen_docs:
                continue
            seen_docs.add(side.doc_index)
            results.append((self.documents[side.doc_index], side, round(float(scores[side_index]), 4)))
            if len(results) >= k:
                break
        return results


_default_index: CrosslingualIndex | None = None


def get_default_index() -> CrosslingualIndex:
    global _default_index
    if _default_index is None:
        _default_index = CrosslingualIndex(load_default_documents())
    return _default_index


# ---------------------------------------------------------------------------
# Retrieval orchestration
# ---------------------------------------------------------------------------


class CrosslingualSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    lang_pref: LangPref = "any"


class CrosslingualHit(BaseModel):
    doc_id: str
    text: str
    text_lang: Literal["ar", "en"]
    score: float
    arabic: str | None = None
    english: str | None = None
    mirrored_snippet: str | None = None
    equivalence_notes: list[str] = Field(default_factory=list)


class CrosslingualSearchResponse(BaseModel):
    query: str
    query_lang: Lang
    results_in: LangPref
    translation: QueryTranslation
    results: list[CrosslingualHit]


def _truncate(text: str, limit: int = _SNIPPET_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _equivalence_notes(matches: list[TermMatch], side_lang: Literal["ar", "en"], side_text: str) -> list[str]:
    """Explain where glossary bridging fired between query and result."""
    notes: list[str] = []
    for match in matches[:3]:
        crossed_scripts = _token_has_arabic(match.matched_text) != (side_lang == "ar")
        if not crossed_scripts:
            continue
        probe = (
            " ".join(normalize_arabic_token(tok) for tok in match.canonical_ar.split())
            if side_lang == "ar"
            else match.canonical_en.casefold()
        )
        if probe and probe in _match_normalize(side_text):
            notes.append(f"glossary bridge: {match.note} — present in this result")
        else:
            notes.append(f"glossary bridge fired: {match.note}")
    return notes


async def crosslingual_search(
    query: str,
    k: int = 5,
    lang_pref: LangPref = "any",
    *,
    documents: list[Document] | None = None,
    embedder: MultilingualEmbedder | None = None,
    translate: bool = True,
) -> CrosslingualSearchResponse:
    """Retrieve bilingual documents for an Arabic, English or mixed query.

    ``lang_pref`` filters which language side may satisfy a hit (``any`` lets
    either side match). When documents/embedder are omitted the bundled
    corpus and the auto-selected embedder are used (cached singleton).
    """
    if lang_pref not in {"any", "ar", "en"}:
        raise ValueError(f"lang_pref must be any|ar|en, got {lang_pref!r}")
    k = max(1, min(int(k), 50))

    detection = detect_script(query)
    translation = await translate_query(query, enabled=translate)

    if documents is None and embedder is None:
        index = get_default_index()
    else:
        index = CrosslingualIndex(documents or load_default_documents(), embedder)

    # Embedding the translated form (when one exists) puts the query closer to
    # the mirrored language's side of the space; otherwise embed as typed.
    embed_text = translation.translated_text if translation.translation_source in {"gemini", "glossary"} else query
    query_vec = await asyncio.to_thread(index.embedder.embed, embed_text)
    found = await asyncio.to_thread(index.search_vectors, query_vec, k, lang_pref)

    query_matches = get_glossary().find_terms(query)
    hits: list[CrosslingualHit] = []
    for doc, side, score in found:
        other_text = doc.en if side.lang == "ar" else doc.ar
        hits.append(
            CrosslingualHit(
                doc_id=doc.doc_id,
                text=side.text,
                text_lang=side.lang,
                score=score,
                arabic=doc.ar,
                english=doc.en,
                mirrored_snippet=_truncate(other_text) if other_text else None,
                equivalence_notes=_equivalence_notes(query_matches, side.lang, side.text),
            )
        )

    return CrosslingualSearchResponse(
        query=query,
        query_lang=detection.lang,
        results_in=lang_pref,
        translation=translation,
        results=hits,
    )
