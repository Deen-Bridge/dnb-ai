from functools import lru_cache
from typing import List

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

    gemini_api_key: str

    model_name: str = "gemini-1.5-flash"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

    gemini_timeout: int = Field(default=30, ge=1)

    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://deenbridge.vercel.app",
            "https://dnb-frontend.vercel.app",
            "http://localhost:8000",
        ]
    )

    port: int = Field(default=8000, ge=1)

    # --- Hybrid retrieval (#226): vector+keyword fusion ----------------------
    # Master switch; weights seed the balanced-mode mix, and the toggles let a
    # deployment run a single channel while a backend (e.g. pgvector) is rolled out.
    hybrid_enabled: bool = True
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_semantic_weight: float = Field(default=0.5, ge=0, le=1)
    hybrid_keyword_weight: float = Field(default=0.5, ge=0, le=1)
    hybrid_top_k: int = Field(default=5, ge=1, le=50)
    hybrid_enable_semantic_channel: bool = True
    hybrid_enable_keyword_channel: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()