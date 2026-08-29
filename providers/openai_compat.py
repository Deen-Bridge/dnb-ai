import httpx

from async_runtime import http_client_pool

from .types import GenerationConfig, Message, ProviderReply


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        model_name: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = client

    async def generate(
        self,
        messages: list[Message],
        *,
        system: str | None,
        config: GenerationConfig,
    ) -> ProviderReply:
        payload_messages = [{"role": "system", "content": system}] if system else []
        payload_messages.extend(
            {"role": "assistant" if message.role == "model" else message.role, "content": message.content}
            for message in messages
        )
        client = self.client or http_client_pool.get()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": payload_messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_output_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return ProviderReply(text=data["choices"][0]["message"]["content"], provider=self.name)
