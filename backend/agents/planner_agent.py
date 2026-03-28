from __future__ import annotations

from backend.models.ollama_client import ModelResponse, OllamaClient


class PlannerAgent:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    async def run(
        self,
        task: str,
        classification: str,
        model: str,
        tokens_remaining: int,
        shared_context: str = "",
    ) -> ModelResponse:
        system_prompt = (
            "You are Planner Agent. Create a concise execution plan for the task.\n"
            f"You have {tokens_remaining} tokens remaining. Be concise and efficient.\n"
            "Return 1-3 short bullet points focused only on this subtask."
        )
        user_prompt = (
            f"Task classification: {classification}\n"
            f"Subtask: {task}\n"
            f"Prior agent context:\n{shared_context or 'None'}\n"
            "Output only a minimal actionable plan."
        )
        return await self.client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
