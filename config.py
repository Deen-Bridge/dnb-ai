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

    gemina_api_key: str = Field(default="test-key")

    model_name: str = "gemini-1.5-flash"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

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

    # --- Arabic Phoneme Alignment System ---
    # Enable/disable the phoneme alignment feature.
    phoneme_alignment_enabled: bool = True

    # Path or identifier for the Arabic phoneme recognition model.
    phoneme_alignment_model: str = "quranic_phoneme_model"

    # Tajw-ede aware phonetic model path/name.
    tajweed_model: str = "tajweed_model"

    # Supported Qira'at (recitation styles) for alignment.
    qiraat_styles: list[str] = Field(
        default_factory=lambda: ["hafs", "warsh"]
    )

    # Confidence threshold for accepting alignments (0.0 - 1.0).
    alignment_confidence_threshold: float = Field(default=0.8, ge=0, le=1)

    # Window size for temporal segmentation (in frames/ms).
    alignment_window_size: int = Field(default=10, ge=1)

    # Enable real-time alignment for live recitation.
    real_time_alignment: bool = True

    # Directory containing the recitation corpus for training/evaluation.
    corpus_directory: str = "data/quranic_corpus"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("qiraat_styles", mode="before")
    @classmethod
    def parse_qiraat_styles(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
