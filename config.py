import enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TajweedLevel(str, enum.Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


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

    # Tajweed detection settings
    enable_tajweed_detection: bool = Field(default=True, description="Enable the tajweed error detection system")
    tajweed_min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum acceptable tajweed score before flagging")
    tajweed_level: TajweedLevel = Field(default=TajweedLevel.advanced, description="Default difficulty level for tajweed analysis")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
