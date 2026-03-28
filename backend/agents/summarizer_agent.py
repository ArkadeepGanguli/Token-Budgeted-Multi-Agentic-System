from __future__ import annotations

from backend.models.ollama_client import ModelResponse, OllamaClient


class SummarizerAgent:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    async def run(
        self,
        content: str,
        model: str,
        tokens_remaining: int,
    ) -> ModelResponse:
        system_prompt = (
            "You are Summarizer Agent. Compress the response while preserving key facts.\n"
            f"You have {tokens_remaining} tokens remaining. Be concise and efficient.\n"
            "Keep output short and information-dense.\n"
            "If the input contains multiple subtask sections, preserve every section in order and do not drop any subtask."
        )
        user_prompt = (
            "Summarize the content below into a compact final response:\n\n"
            f"{content}"
        )
        return await self.client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
