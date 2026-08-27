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

    gemini_api_key: str = Field(default="test-key")

    model_name: str = "gemini-1.5-flash"

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1)

    gemini_timeout: int = Field(default=30, ge=1)

    # Fallback model configuration
    fallback_enabled: bool = Field(default=True, description="Enable automatic failover to fallback models")
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Ordered list of fallback models to use when primary fails",
    )
    fallback_health_check_interval: int = Field(
        default=30, ge=1, description="Interval in seconds for health checks on models"
    )
    fallback_failure_threshold: int = Field(
        default=3, ge=1, description="Number of consecutive failures to trip circuit breaker"
    )
    fallback_circuit_breaker_timeout: int = Field(
        default=60, ge=1, description="Recovery timeout in seconds after circuit breaker trips"
    )
    fallback_quality_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum acceptable quality score for fallback models"
    )
    fallback_restore_interval: int = Field(
        default=120, ge=1, description="Interval in seconds to retry primary model after recovery"
    )
    fallback_alerts_enabled: bool = Field(
        default=True, description="Enable alerts when fallback is activated"
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://deenbridge.vercel.app",
            "https://dnb-frontend.vercel.app",
            "http://localhost:8000",
        ]
    )

    port: int = Field(default=8000, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("fallback_models", mode="before")
    @classmethod
    def parse_fallback_models(cls, value):
        if isinstance(value, str):
            # Support comma-separated or JSON array string (if from env)
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        value = parsed
                except json.JSONDecodeError:
                    pass
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Setting's()
