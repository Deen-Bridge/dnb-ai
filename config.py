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

    gemini_api_key: str = ""

    model_name: str = "gemini-1.5-flash"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

    gemini_timeout: int = Field(default=30, ge=1)

    # Calligraphy OCR (#234)
    calligraphy_provider: str = "gemini"
    calligraphy_max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    calligraphy_min_confidence: float = Field(default=0.35, ge=0, le=1)
    manuscripts_provider: str = Field(default="gemini")
    manuscripts_max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    manuscripts_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- Hybrid retrieval (#226) ---
    hybrid_enabled: bool = True
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_semantic_weight: float = Field(default=0.5, ge=0, le=1)
    hybrid_keyword_weight: float = Field(default=0.5, ge=0, le=1)
    hybrid_top_k: int = Field(default=5, ge=1, le=50)
    hybrid_enable_semantic_channel: bool = True
    hybrid_enable_keyword_channel: bool = True

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://deenbridge.vercel.app",
            "https://dnb-frontend.vercel.app",
            "http://localhost:8000",
        ]
    )

    port: int = Field(default=8000, ge=1)

    # Narrator biography (rijal) database
    enable_narrator_db: bool = Field(default=True)
    narrator_db_path: str = Field(default="data/narrators.json")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
