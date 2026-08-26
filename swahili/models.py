"""Data models and schemas for the Swahili Islamic Language Subsystem."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SwahiliDialect(str, Enum):
    """Regional Swahili dialects and linguistic variants in East Africa."""

    SANIFU = "sanifu"  # Standard Swahili (Tanzania & Kenya official standard)
    PWANI_MVITA = "pwani_mvita"  # Coastal Kenya (Mombasa/Mvita)
    PWANI_UNGUJA = "pwani_unguja"  # Zanzibar Town & Islands (Unguja)
    PWANI_AMU = "pwani_amu"  # Lamu Archipelago (Classical literary Swahili)
    PWANI_PEMBA = "pwani_pemba"  # Pemba Island dialect
    BARA_INLAND = "bara_inland"  # Upcountry / Inland Swahili (Tanzania, Kenya, Uganda, DRC)
    SHENG_URBAN = "sheng_urban"  # Urban code-mixed slang (Nairobi, Dar es Salaam)
    UNKNOWN = "unknown"


class CodeSwitchType(str, Enum):
    """Types of code-switching observed in East African Islamic communications."""

    MONOLINGUAL_SWAHILI = "monolingual_swahili"  # Pure Swahili
    SWAHILI_ARABIC_MIXED = "swahili_arabic_mixed"  # Swahili with Arabic religious phrases
    SWAHILI_ENGLISH_MIXED = "swahili_english_mixed"  # Swahili with English loanwords / phrasing
    TRILINGUAL_MIXED = "trilingual_mixed"  # Swahili + English + Arabic


class IslamicDomain(str, Enum):
    """Islamic thematic domains for terminology categorization."""

    IBADA = "ibada"  # Worship & Rituals (Swala, Saumu, Zaka, Hija, Udhu)
    AQIDAH = "aqidah"  # Creed & Theology (Tawhidi, Imani, Malaika, Akhera)
    FIQHI = "fiqhi"  # Jurisprudence & Legal Categories (Halali, Haramu, Wajibu)
    MUAMALAT = "muamalat"  # Commercial, Financial & Ethical Transactions (Riba, Wakfu)
    NDOA_MIRATHI = "ndoa_mirathi"  # Family Law, Marriage & Inheritance (Ndoa, Talaka, Mirathi)
    QURAN_HADITHI = "quran_hadithi"  # Scripture & Prophetic Tradition (Qur'ani, Hadithi, Tafsiri)
    MAADILI = "maadili"  # Ethics, Morality & Spirituality (Taqwa, Ihsani, Subira)
    UTAMADUNI_HISTORIA = "utamaduni_historia"  # East African Islamic Culture & Institutions


class SwahiliToken(BaseModel):
    """Represents a tokenized Swahili word with morphological annotations."""

    raw_token: str
    normalized_token: str
    lemma: str
    prefix: str | None = None
    suffix: str | None = None
    is_arabic_loanword: bool = False
    arabic_root: str | None = None
    canonical_term: str | None = None
    dialect_marker: SwahiliDialect | None = None


class IslamicTerm(BaseModel):
    """Structured representation of an Islamic concept in Swahili."""

    id: str
    swahili_term: str
    arabic_original: str
    arabic_transliteration: str
    english_equivalent: str
    category: IslamicDomain
    definition_sw: str
    definition_en: str
    variants_sw: list[str] = Field(default_factory=list)
    dialect_notes: str | None = None
    common_misspellings: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)


class LoanwordMatch(BaseModel):
    """Result of mapping a Swahili token to an Arabic loanword origin."""

    raw_word: str
    matched_term: str
    arabic_original: str
    arabic_transliteration: str
    category: str
    confidence: float
    phonological_rule_applied: str | None = None
    morphological_prefix: str | None = None


class DialectResult(BaseModel):
    """Classification of regional Swahili dialect in a query."""

    primary_dialect: SwahiliDialect
    confidence: float
    is_coastal: bool
    detected_markers: list[str] = Field(default_factory=list)
    normalized_equivalents: dict[str, str] = Field(default_factory=dict)


class CodeSwitchSegment(BaseModel):
    """A segment of text classified by its specific language."""

    text: str
    language: str  # "sw", "ar", "en", "mixed"
    is_islamic_formula: bool = False
    gloss: str | None = None


class CodeSwitchResult(BaseModel):
    """Full analysis of code-switching within a multi-lingual input."""

    dominant_language: str
    switch_type: CodeSwitchType
    segments: list[CodeSwitchSegment] = Field(default_factory=list)
    arabic_phrases: list[str] = Field(default_factory=list)
    contains_quran_or_hadith: bool = False
    contains_dua: bool = False


class CulturalContext(BaseModel):
    """East African Islamic cultural and legal context extracted from query."""

    shafi_madhhab_relevant: bool = False
    local_institutions_mentioned: list[str] = Field(default_factory=list)
    prayer_time_context: str | None = None
    cultural_event_context: str | None = None
    honorifics_guidance: list[str] = Field(default_factory=list)
    recommended_madhhab: str | None = "shafii"


class SwahiliAnalysisResult(BaseModel):
    """Aggregated analysis of a Swahili query across all pipeline stages."""

    original_text: str
    normalized_text: str
    tokens: list[SwahiliToken] = Field(default_factory=list)
    detected_terms: list[IslamicTerm] = Field(default_factory=list)
    loanwords: list[LoanwordMatch] = Field(default_factory=list)
    dialect: DialectResult
    code_switch: CodeSwitchResult
    cultural_context: CulturalContext
    processing_time_ms: float = 0.0


class SwahiliPromptEnhancement(BaseModel):
    """Enhanced prompt payload configured for optimal Swahili generation."""

    system_instructions: str
    contextual_glossary: dict[str, str] = Field(default_factory=dict)
    cultural_notes: list[str] = Field(default_factory=list)
    enhanced_user_prompt: str


class SwahiliAnalyzeRequest(BaseModel):
    """Request schema for /swahili/analyze endpoint."""

    text: str = Field(..., description="Swahili or code-mixed Islamic query to analyze", min_length=1)
    target_dialect: SwahiliDialect | None = None


class SwahiliNormalizeRequest(BaseModel):
    """Request schema for /swahili/normalize endpoint."""

    text: str = Field(..., min_length=1)
    preserve_dialect: bool = False


class SwahiliNormalizeResponse(BaseModel):
    """Response schema for /swahili/normalize endpoint."""

    original_text: str
    normalized_text: str
    replaced_terms: dict[str, str] = Field(default_factory=dict)


class SwahiliCodeSwitchRequest(BaseModel):
    """Request schema for /swahili/code-switch endpoint."""

    text: str = Field(..., min_length=1)


class SwahiliTermsQuery(BaseModel):
    """Query parameters for searching Swahili Islamic terminology."""

    query: str | None = None
    category: IslamicDomain | None = None
    limit: int = 50
