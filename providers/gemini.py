import google.generativeai as genai

from .types import GenerationConfig, Message, ProviderReply


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        genai.configure(api_key=api_key)

    async def generate(
        self,
        messages: list[Message],
        *,
        system: str | None,
        config: GenerationConfig,
    ) -> ProviderReply:
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
        )
        history = [{"role": message.role, "parts": [{"text": message.content}]} for message in messages[:-1]]
        session = model.start_chat(history=history)
        response = await session.send_message_async(
            messages[-1].content,
            generation_config={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_output_tokens": config.max_output_tokens,
            },
        )
        return ProviderReply(text=response.text, provider=self.name)
