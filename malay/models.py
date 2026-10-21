from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MalayQueryRequest(BaseModel):
    text: str = Field(..., description="Input text in Malay or Indonesian, Latin or Jawi script")
    dialect: str | None = Field(None, description="Regional dialect: 'malaysia' or 'indonesia'")
    target_script: str | None = Field("latin", description="Target script for output: 'latin' or 'jawi'")
    context: dict[str, Any] | None = Field(default_factory=dict)


class MalayQueryResponse(BaseModel):
    original_text: str
    normalized_text: str
    detected_dialect: str
    detected_script: str
    transliterated_text: str | None = None
    islamic_terms_identified: list[dict[str, Any]] = Field(default_factory=list)
    optimized_response: str
    confidence: float = 1.0
