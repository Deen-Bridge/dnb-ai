from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class ProviderReply:
    text: str
    provider: str


class LLMProvider(Protocol):
    name: str

    async def generate(
        self,
        messages: list[Message],
        *,
        system: str | None,
        config: GenerationConfig,
    ) -> ProviderReply: ...
