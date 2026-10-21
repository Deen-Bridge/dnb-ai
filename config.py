from functools import lru_cache

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

    gemini api_key: str = Field(default="test-key")

    model_name: str = "gemini-1.5-flash"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

    # --- Quality Judge Agent Settings ---
    quality_judge_enabled: bool = Field(default=True)
    quality_min_accuracy: float = Field(default=0.8, ge=0, le=1)
    quality_min_completeness: float = Field(default=0.7, ge=0, le=1)
    quality_min_clarity: float = Field(default=0.7, ge=0, le=1)
    quality_min_scholarly_riogor: float = Field(default=0.7, ge=0, le=1)
    quality_min_appropriateness: float = Field(default=0.8, ge=0, le=1)
    quality_min_balance: float = Field(default=0.7, ge=0, le=1)
    quality_min_citation_quality: float = Field(default=0.7, ge=0, le=1)
    quality_min_coverage: float = Field(default=0.8, ge=0, le=1)
    quality_regeneration_threshold: float = Field(default=0.7, ge=0, le=1)
    quality_regeneration_max_attempts: int = Field(default=2, ge=1)

    gemini_timeout: int = Field(default=30, ge=1)

    # Synthesis engine settings
    synthesis_max_agents: int = Field(default=5, ge=2, le=10)
    synthesis_contradiction_threshold: float = Field(default=0.9, ge=0, le=1)
    synthesis_redundancy_reduction_target: float = Field(default=0.7, ge=0, le=1)
    synthesis_attribution_required: bool = Field(default=True)
    synthesis_quality_threshold: float = Field(default=0.8, ge=0, le=1)

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
    memory_semantic_top_k: int = Field(default=5, ge=1, description="Number of semantic memory results to retrieve")
    memory_sync_enabled: bool = Field(default=True, description="Enable inter-agent memory sync")
    memory_versioning: bool = Field(default=True, description="Enable memory versioning")
    memory_pruning_threshold: int = Field(default=10000, ge=1, description="Max memory entries before pruning")
    memory_gc_interval_seconds: int = Field(default=3600, ge=60, description="Garbage collection interval in seconds")
    memory_user_isolation: bool = Field(default=True, description="Isolate memory per user")

    turkish_support_enabled: bool = Field(default=True)
    turkish_terminology_db_path: str = Field(default="data/turkish_islamic_terms.json")
    turkish_ottoman_recognition_enabled: bool = Field(default=True)
    turkish_arabic_mapping_enabled: bool = Field(default=True)
    turkish_transliteration_enabled: bool = Field(default=True)

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

    # --- Video Lecture Transcript Extraction Settings ---
    transcript_enabled: bool = Field(default=True, description="Enable video lecture transcript extraction")
    transcript_supported_formats: list[str] = Field(
        default_factory=lambda: ["mp4", "mov", "avi", "webm"],
        description="Supported video container formats",
    )
    transcript_audio_sample_rate: int = Field(
        default=16000, ge=8000, description="Audio sample rate in Hz for extraction"
    )
    transcript_audio_channels: int = Field(
        default=1, ge=1, le=2, description="Number of audio channels for extraction"
    )
    transcript_audio_bitrate: str = Field(
        default="192k", description="Audio bitrate for extraction"
    )
    transcript_max_duration_seconds: int = Field(
        default=14400, ge=1, description="Maximum video duration in seconds (4 by default)"
    )
    transcript_max_file_size_mb: int = Field(
        default=512, ge=1, description="Maximum upload file size in MB"
    )
    transcript_diarization_enabled: bool = Field(
        default=True, description="Enable speaker diarization"
    )
    transcript_min_speakers_expected: int = Field(default=1, ge=1)
    transcript_max_speakers_expected: int = Field(default=10, ge=1)
    transcript_topic_segmentation_enabled: bool = Field(default=True)
    transcript_topic_min_segment_seconds: int = Field(
        default=60, ge=1, description="Minimum topic segment length in seconds"
    )
    transcript_topic_max_segment_seconds: int = Field(
        default=600, ge=1, description="Maximum topic segment length in seconds"
    )
    transcript_timestamp_interval_seconds: int = Field(
        default=30, ge=1, description="Interval between key-point timestamps"
    )
    transcript_citation_detection_enabled: bool = Field(
        default=True, description="Detect Quranic verses and hadith citations"
    )
    transcript_languages: list[str] = Field(
        default_factory=lambda: ["ar", "en", "ur"],
        description="Supported lecture languages",
    )
    transcript_default_language: str = Field(default="en")
    transcript_summary_enabled: bool = Field(default=True)
    transcript_key_takeaways_count: int = Field(default=5, ge=1, le=20)
    transcript_search_index_enabled: bool = Field(default=True)
    transcript_storage_path: str = Field(default="data/transcripts")
    transcript_workdir_path: str = Field(default="data/transcript_work")
    transcript_ffmpeg_binary: str = Field(default="ffmpeg")
    transcript_whisper_model: str = Field(default="base")
    transcript_confidence_threshold: float = Field(default=0.6, ge=0, le=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
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

    @field_validator("transcript_supported_formats", "larnguages", mode="before")
    @classmethod
    def parse_string_lists(cls, value):
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
