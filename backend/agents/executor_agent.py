from __future__ import annotations

import re
from pathlib import Path

from backend.models.ollama_client import ModelResponse, OllamaClient
from backend.tools.calculator import calculate
from backend.tools.formatter import format_json, format_text
from backend.tools.mock_search import search_local_kb


class ExecutorAgent:
    def __init__(self, client: OllamaClient, kb_path: Path) -> None:
        self.client = client
        self.kb_path = kb_path

    def _extract_expression(self, task: str) -> str | None:
        match = re.search(r"([-+/*()%\d.\s]{3,})", task)
        if match:
            return match.group(1).strip()
        return None

    def _run_tools(self, task: str) -> tuple[list[dict], str]:
        lower = task.lower()
        tool_calls: list[dict] = []
        notes: list[str] = []

        if any(token in lower for token in {"search", "lookup", "find info", "look up"}):
            search_results = search_local_kb(task, self.kb_path)
            tool_calls.append({"tool": "mock_search", "input": task, "output": search_results})
            notes.append("Local search results:\n" + "\n".join(f"- {item}" for item in search_results))

        if any(token in lower for token in {"calculate", "compute", "math"}):
            expression = self._extract_expression(task) or task
            calc_result = calculate(expression)
            tool_calls.append({"tool": "calculator", "input": expression, "output": calc_result})
            notes.append(f"Calculator result: {calc_result}")

        if "format this json" in lower or "prettify json" in lower:
            json_candidate = task.split(":", 1)[-1].strip() if ":" in task else task
            formatted = format_json(json_candidate)
            tool_calls.append({"tool": "string_formatter", "input": json_candidate, "output": formatted})
            notes.append("JSON formatter output:\n" + formatted)
        elif any(token in lower for token in {"format", "reformat", "clean text"}):
            formatted_text = format_text(task)
            tool_calls.append({"tool": "string_formatter", "input": task, "output": formatted_text})
            notes.append("Text formatter output:\n" + formatted_text)

        return tool_calls, "\n\n".join(notes).strip()

    async def run(
        self,
        task: str,
        classification: str,
        plan: str,
        model: str,
        tokens_remaining: int,
        shared_context: str = "",
        agent_messages: list[str] | None = None,
        force_code: bool = False,
    ) -> tuple[ModelResponse, list[dict]]:
        tool_calls, tool_context = self._run_tools(task)
        prior_messages = "\n".join(agent_messages or [])
        target_words = 90 if classification == "simple" else 170 if classification == "moderate" else 260
        code_directive = (
            "You MUST include runnable Python code in a fenced ```python block and include explicit time/space complexity."
            if force_code
            else "If code is requested, include runnable Python code and explicit time/space complexity."
        )
        system_prompt = (
            "You are Executor Agent. Execute the plan and answer the task directly.\n"
            f"You have {tokens_remaining} tokens remaining. Be concise and efficient.\n"
            f"Target output length: under {target_words} words unless code is explicitly requested.\n"
            "When tool context is available, incorporate it accurately.\n"
            "If prior agent outputs exist, build on them instead of repeating work.\n"
            "You MUST fully complete this subtask. If the subtask asks for multiple deliverables, include all deliverables.\n"
            f"{code_directive}"
        )
        user_prompt = (
            f"Task classification: {classification}\n"
            f"Subtask: {task}\n\n"
            f"Planner message:\n{plan}\n\n"
            f"Shared context from prior subtasks:\n{shared_context or 'None'}\n\n"
            f"Agent communication log:\n{prior_messages or 'None'}\n\n"
            f"Tool context:\n{tool_context or 'No tools used.'}\n\n"
            "Return the final answer only."
        )
        response = await self.client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
        return response, tool_calls
