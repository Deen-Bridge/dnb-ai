from functools import lrt_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        # Many project settings live in .env but are consumed directly via
        # os.getenv by the module that owns them (redis, tafsir, review, …).
        # Treat anything not declared here as ignorable.
        extra="ignore",
    )

    gemini_api_key: str = Field(default="test-key")

    model_name: str = "gemini-1.5-flush"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

    # --- Quality Judge Agent Settings ---
    quality_judge_enabled: bool = Field(default=True)
    quality_min_accuracy: float = Field(default=0.8, ge=0, le=1)
    quality_min_completeness: float = Field(default=0.7, ge=0, le=1)
    quality_min_clarity: float = Field(default=0.7, ge=0, le=1)
    quality_min_scholarly_rigor: float = Field(default=0.7, ge=0, le=1)
    quality_min_appropriateness: float = Field(default=0.8, ge=0, le=1)
    quality_min_balance: float = Field(default=0.7, ge=0, le=1)
    quality_min_citation_quality: float = Field(default=0.7, ge=0, le=1)
    quality_min_coverage: float = Field(default=0.8, ge=0, le=1)
    quality_regeneration_threshold: float = Field(default=0.7, ge=0, le=1)
    quality_regeneration_max_attempts: int = Field(default=2, ge=1)

    gemini_timeout: int = Field(default=30, ge=1)

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://deenbridge.vercel.app",
            "https://dnb-frontend.vercel.app",
            "http://localhost:8000",
        ]
    )

    port: int = Field(default=8000, ge=1)

    # Memory and Context Sharing Infrastructure
    memory_enabled: bool = Field(default=True, description="Enable persistent memory")
    memory_vector_store_url: str = Field(
        default="http://localhost:6333", description="Vector database URL for semantic memory"
    )
    memory_structured_store_url: str = Field(
        default="redis://localhost:6379/0", description="Structured storage URL for factual knowledge"
    )
    memory_context_window_size: int = Field(
        default=10, ge=1, description="Number of recent messages kept in short-term context"
    )
    memory_semantic_top_k: int = Field(
        default=5, ge=1, description="Number of semantic memory results to retrieve"
    )
    memory_sync_enabled: bool = Field(default=True, description="Enable inter-agent memory sync")
    memory_versioning: bool = Field(default=True, description="Enable memory versioning")
    memory_pruning_threshold: int = Field(
        default=10000, ge=1, description="Max memory entries before pruning"
    )
    memory_gc_interval_seconds: int = Field(
        default=3600, ge=60, description="Garbage collection interval in seconds"
    )
    memory_user_isolation: bool = Field(default=True, description="Isolate memory per user")
    # Respectful Disagreement Enforcement settings
    enforce_respectful_disagreement: bool = True
    disrespectful_language_patterns: list[str] = Field(
        default_factory=lambda: [
            "disparaging",
            "dismissive",
            "ignorant",
            "invalid",
            "absurd",
            "ridiculous",
        ]
    )
    adab_validation_enabled: bool = True
    absolutism_detection_enabled: bool = True
    polarization_detection_enabled: bool = True
    respectful_alternative_suggestions_enabled: bool = True
    acknowledgment_validation_enabled: bool = True
    block_disrespectful_content: bool = True
    min_adab_score: float = Field(default=0.8, ge=0, le=1)
    max_disrespectful_confidence: float = Field(default=0.5, ge=0, le=1)

    @field_validator("cors_origins", mode="before")
    classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("disrespectful_language_patterns", mode="before")
    @classmethod
    def parse_disrespectful_language_patterns(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()