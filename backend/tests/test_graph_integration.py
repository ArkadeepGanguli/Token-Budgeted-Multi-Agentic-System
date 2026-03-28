import asyncio

from backend.config import Settings
from backend.graph import MultiAgentWorkflow
from backend.models.ollama_client import ModelResponse
from backend.schemas import RunEvent


class FakeOllamaClient:
    def __init__(
        self,
        fail_large: bool = False,
        fail_small: bool = False,
        large_model: str = "llama-3.1-8b-instant",
        small_model: str = "qwen2:1.5b",
    ) -> None:
        self.fail_large = fail_large
        self.fail_small = fail_small
        self.large_model = large_model
        self.small_model = small_model
        self.calls: list[dict] = []

    async def chat(self, model: str, system_prompt: str, user_prompt: str) -> ModelResponse:
        self.calls.append({"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.fail_large and model == self.large_model:
            raise RuntimeError("simulated large model failure")
        if self.fail_small and model == self.small_model:
            raise RuntimeError("simulated small model failure")

        if "Planner Agent" in system_prompt:
            text = "- Analyze task\n- Produce concise answer"
        elif "Summarizer Agent" in system_prompt:
            text = "Compressed summary."
        elif "MUST include runnable Python code" in system_prompt:
            text = (
                "```python\n"
                "def dijkstra(graph, start):\n"
                "    return {}, {}\n"
                "```\n"
                "Time complexity: O((V+E)logV). Space complexity: O(V+E)."
            )
        else:
            if model == "qwen2:1.5b" and "compare" in user_prompt.lower():
                text = "It depends."
            else:
                text = "Detailed technical explanation with examples and code blocks."

        return ModelResponse(
            text=text,
            prompt_tokens=15,
            completion_tokens=20,
            total_tokens=35,
            latency_ms=5.0,
            model=model,
            metadata={},
        )

    async def health(self) -> dict:
        return {
            "ok": True,
            "ollama_reachable": True,
            "available_models": ["qwen2:1.5b"],
            "groq_models": ["llama-3.1-8b-instant"],
            "groq_configured": True,
            "groq_reachable": True,
            "large_provider": "groq",
            "missing_models": [],
        }


def test_moderate_escalation_and_event_stream() -> None:
    async def _run() -> tuple[dict, list[RunEvent], FakeOllamaClient]:
        events: list[RunEvent] = []
        settings = Settings()
        client = FakeOllamaClient(fail_large=False, large_model=settings.large_model)
        workflow = MultiAgentWorkflow(settings=settings, client=client)

        async def callback(event: RunEvent) -> None:
            events.append(event)

        state = await workflow.run("run-1", "Compare indexing strategies for databases", callback)
        return state, events, client

    state, events, client = asyncio.run(_run())
    assert state["status"] in {"completed", "budget_exhausted"}
    assert any(event.event_type == "route_decision" for event in events)
    assert any("escalat" in event.message.lower() for event in events if event.event_type == "route_decision")
    assert any(call["model"] == client.large_model for call in client.calls)
    assert state.get("model_budgets")
    assert state.get("orchestration_savings")


def test_large_failure_fallback_to_small() -> None:
    async def _run() -> tuple[dict, list[RunEvent]]:
        events: list[RunEvent] = []
        settings = Settings()
        client = FakeOllamaClient(fail_large=True, large_model=settings.large_model)
        workflow = MultiAgentWorkflow(settings=settings, client=client)

        async def callback(event: RunEvent) -> None:
            events.append(event)

        state = await workflow.run("run-2", "Explain Dijkstra algorithm with code", callback)
        return state, events

    state, events = asyncio.run(_run())
    assert state["selected_model"] == "qwen2:1.5b"
    assert state["status"] in {"completed", "budget_exhausted"}
    assert any("fallback" in event.message.lower() for event in events if event.event_type == "route_decision")


def test_mixed_subtasks_route_to_both_models() -> None:
    async def _run() -> tuple[dict, FakeOllamaClient]:
        settings = Settings()
        client = FakeOllamaClient(fail_large=False, large_model=settings.large_model)
        workflow = MultiAgentWorkflow(settings=settings, client=client)

        async def callback(_: RunEvent) -> None:
            return None

        state = await workflow.run(
            "run-3",
            "Calculate 2+2, then explain Dijkstra algorithm with examples",
            callback,
        )
        return state, client

    state, client = asyncio.run(_run())
    assert len(state.get("subtasks", [])) >= 2
    assert len(state.get("subtask_results", [])) == len(state.get("subtasks", []))
    used_models = {call["model"] for call in client.calls}
    assert "qwen2:1.5b" in used_models
    assert client.large_model in used_models


def test_small_failure_fallback_to_large_for_subtasks() -> None:
    async def _run() -> tuple[dict, list[RunEvent], FakeOllamaClient]:
        events: list[RunEvent] = []
        settings = Settings()
        client = FakeOllamaClient(
            fail_large=False,
            fail_small=True,
            large_model=settings.large_model,
            small_model=settings.small_model,
        )
        workflow = MultiAgentWorkflow(settings=settings, client=client)

        async def callback(event: RunEvent) -> None:
            events.append(event)

        state = await workflow.run("run-4", "Format this JSON: {\"a\":1,\"b\":[3,2,1]}", callback)
        return state, events, client

    state, events, client = asyncio.run(_run())
    assert state["status"] in {"completed", "budget_exhausted"}
    assert any(call["model"] == client.large_model for call in client.calls)
    assert any("fallback" in event.message.lower() for event in events if event.event_type in {"route_decision", "error"})


def test_full_query_keeps_all_subtasks_in_final_output() -> None:
    async def _run() -> dict:
        settings = Settings()
        client = FakeOllamaClient(fail_large=False, large_model=settings.large_model, small_model=settings.small_model)
        workflow = MultiAgentWorkflow(settings=settings, client=client)

        async def callback(_: RunEvent) -> None:
            return None

        return await workflow.run(
            "run-5",
            "Calculate 1250*37/5, then explain Dijkstra algorithm and write optimized Python code with complexity. Show both the outputs.",
            callback,
        )

    state = asyncio.run(_run())
    subtasks = state.get("subtasks", [])
    results = state.get("subtask_results", [])
    assert len(subtasks) == 3
    assert len(results) == len(subtasks)
    assert "Subtask 1" in state.get("executor_output", "")
    assert "Subtask 2" in state.get("executor_output", "")
    assert "Subtask 3" in state.get("executor_output", "")
    assert "```python" in state.get("executor_output", "")
