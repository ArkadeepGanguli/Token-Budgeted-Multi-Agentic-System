from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.executor_agent import ExecutorAgent
from backend.agents.planner_agent import PlannerAgent
from backend.agents.summarizer_agent import SummarizerAgent
from backend.budget import BudgetTracker
from backend.classifier import classify_task
from backend.config import Settings, get_settings
from backend.models.ollama_client import ModelResponse, OllamaClient
from backend.router import initial_model, should_escalate_moderate
from backend.schemas import RunEvent

logger = logging.getLogger("token_budgeted_multi_agent")
EventCallback = Callable[[RunEvent], Awaitable[None]]


class GraphState(TypedDict, total=False):
    run_id: str
    task: str
    classification: str
    selected_model: str
    budget_total: int
    tokens_used: int
    tokens_remaining: int
    steps: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    planner_output: str
    executor_output: str
    final_output: str
    status: str
    route_reason: str
    should_summarize: bool
    model_usage: dict[str, dict[str, Any]]
    model_budgets: dict[str, dict[str, Any]]
    orchestration_savings: dict[str, Any]
    subtasks: list[dict[str, Any]]
    subtask_results: list[dict[str, Any]]
    agent_messages: list[str]
    event_callback: EventCallback


class MultiAgentWorkflow:
    ACTION_PREFIXES = (
        "search", "find", "look up", "calculate", "compute", "format", "reformat", "summarize",
        "explain", "write", "generate", "analyze", "compare", "optimize", "debug", "design",
        "implement", "create", "plan", "draft",
    )

    def __init__(self, settings: Settings | None = None, client: OllamaClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OllamaClient(self.settings)
        kb_path = Path(__file__).resolve().parent / "data" / "local_kb.txt"

        self.budget_tracker = BudgetTracker(self.settings.budget_limits, self.settings.summarize_threshold)
        self.planner_agent = PlannerAgent(self.client)
        self.executor_agent = ExecutorAgent(self.client, kb_path=kb_path)
        self.summarizer_agent = SummarizerAgent(self.client)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("classify_task", self._classify_task_node)
        graph.add_node("init_budget", self._init_budget_node)
        graph.add_node("decompose_task", self._decompose_task_node)
        graph.add_node("execute_subtasks", self._execute_subtasks_node)
        graph.add_node("budget_guard", self._budget_guard_node)
        graph.add_node("summarizer", self._summarizer_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "classify_task")
        graph.add_edge("classify_task", "init_budget")
        graph.add_edge("init_budget", "decompose_task")
        graph.add_edge("decompose_task", "execute_subtasks")
        graph.add_edge("execute_subtasks", "budget_guard")
        graph.add_conditional_edges("budget_guard", self._budget_guard_route, {"summarizer": "summarizer", "finalize": "finalize"})
        graph.add_edge("summarizer", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    async def _emit(self, state: GraphState, event_type: str, message: str, step_data: dict[str, Any] | None = None) -> None:
        callback = state.get("event_callback")
        if callback is None:
            return
        await callback(
            RunEvent(
                event_type=event_type,
                run_id=state["run_id"],
                classification=state.get("classification"),
                model=state.get("selected_model"),
                tokens_used=state.get("tokens_used", 0),
                tokens_remaining=state.get("tokens_remaining", 0),
                message=message,
                step_data=step_data or {},
            )
        )

    def _record_step(self, state: GraphState, node: str, message: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        step = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "node": node,
            "message": message,
            "model": state.get("selected_model"),
            "classification": state.get("classification"),
            "tokens_used": state.get("tokens_used", 0),
            "tokens_remaining": state.get("tokens_remaining", 0),
        }
        if metadata:
            step.update(metadata)
        logger.info(json.dumps(step))
        return [*state.get("steps", []), step]

    def _initial_model_usage(self) -> dict[str, dict[str, Any]]:
        return {
            "small": {"role": "small", "provider": "ollama", "model": self.settings.small_model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
            "large": {"role": "large", "provider": self.settings.large_provider, "model": self.settings.large_model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        }

    def _token_cost_usd(self, role: str, prompt_tokens: int, completion_tokens: int) -> float:
        if role == "large":
            in_rate = self.settings.large_model_input_usd_per_1m
            out_rate = self.settings.large_model_output_usd_per_1m
        else:
            in_rate = self.settings.small_model_input_usd_per_1m
            out_rate = self.settings.small_model_output_usd_per_1m
        return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate

    def _derive_budget_metrics(self, model_usage: dict[str, dict[str, Any]], tokens_used: int, budget_total: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        s = model_usage["small"]
        l = model_usage["large"]
        small_cost = self._token_cost_usd("small", int(s["prompt_tokens"]), int(s["completion_tokens"]))
        large_cost = self._token_cost_usd("large", int(l["prompt_tokens"]), int(l["completion_tokens"]))
        baseline_cost = self._token_cost_usd("large", int(s["prompt_tokens"]) + int(l["prompt_tokens"]), int(s["completion_tokens"]) + int(l["completion_tokens"]))
        model_budgets = {
            "small": {"role": "small", "provider": "ollama", "model": self.settings.small_model, "cap_tokens": self.settings.small_model_token_cap, "used_tokens": int(s["total_tokens"]), "remaining_tokens": max(0, self.settings.small_model_token_cap - int(s["total_tokens"])), "estimated_cost_usd": round(small_cost, 8)},
            "large": {"role": "large", "provider": self.settings.large_provider, "model": self.settings.large_model, "cap_tokens": self.settings.large_model_token_cap, "used_tokens": int(l["total_tokens"]), "remaining_tokens": max(0, self.settings.large_model_token_cap - int(l["total_tokens"])), "estimated_cost_usd": round(large_cost, 8)},
        }
        savings = {
            "task_budget_total_tokens": budget_total,
            "task_budget_used_tokens": tokens_used,
            "task_budget_saved_tokens": max(0, budget_total - tokens_used),
            "large_tokens_avoided": int(s["total_tokens"]),
            "baseline_large_only_tokens": int(s["total_tokens"]) + int(l["total_tokens"]),
            "actual_large_tokens": int(l["total_tokens"]),
            "estimated_usd_saved_vs_large_only": round(max(0.0, baseline_cost - large_cost), 8),
            "estimated_actual_cost_usd": round(small_cost + large_cost, 8),
            "estimated_baseline_large_only_cost_usd": round(baseline_cost, 8),
        }
        return model_budgets, savings

    def _apply_model_usage(self, state: GraphState, response: ModelResponse, model: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        existing = state.get("model_usage") or self._initial_model_usage()
        usage = {"small": dict(existing["small"]), "large": dict(existing["large"])}
        if response.total_tokens <= 0 or model not in {self.settings.small_model, self.settings.large_model}:
            model_budgets, savings = self._derive_budget_metrics(usage, state.get("tokens_used", 0), state.get("budget_total", 0))
            return usage, model_budgets, savings
        role = "large" if model == self.settings.large_model else "small"
        usage[role]["prompt_tokens"] = int(usage[role]["prompt_tokens"]) + int(response.prompt_tokens)
        usage[role]["completion_tokens"] = int(usage[role]["completion_tokens"]) + int(response.completion_tokens)
        usage[role]["total_tokens"] = int(usage[role]["total_tokens"]) + int(response.total_tokens)
        usage[role]["calls"] = int(usage[role]["calls"]) + 1
        model_budgets, savings = self._derive_budget_metrics(usage, state.get("tokens_used", 0), state.get("budget_total", 0))
        return usage, model_budgets, savings

    def _starts_with_action(self, text: str) -> bool:
        lowered = text.strip().lower()
        return any(lowered.startswith(prefix) for prefix in self.ACTION_PREFIXES)

    def _split_top_level_actions(self, text: str) -> list[str]:
        segments: list[str] = []
        buf: list[str] = []
        depth = 0
        in_quote: str | None = None
        i = 0
        while i < len(text):
            ch = text[i]
            prev = text[i - 1] if i > 0 else ""
            if ch in {"'", '"'} and prev != "\\":
                in_quote = None if in_quote == ch else (ch if in_quote is None else in_quote)
            if in_quote is None:
                if ch in "{[(":
                    depth += 1
                elif ch in "}])":
                    depth = max(0, depth - 1)
            if in_quote is None and depth == 0 and ch == "," and self._starts_with_action(text[i + 1 :]):
                part = "".join(buf).strip(" ,")
                if part:
                    segments.append(part)
                buf = []
                i += 1
                while i < len(text) and text[i].isspace():
                    i += 1
                continue
            if in_quote is None and depth == 0 and text[i : i + 5].lower() == " and " and self._starts_with_action(text[i + 5 :]):
                part = "".join(buf).strip(" ,")
                if part:
                    segments.append(part)
                buf = []
                i += 5
                while i < len(text) and text[i].isspace():
                    i += 1
                continue
            buf.append(ch)
            i += 1
        part = "".join(buf).strip(" ,")
        if part:
            segments.append(part)
        return segments

    def _split_task_into_subtasks(self, task: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", task.strip())
        if not normalized:
            return []
        parts = re.split(r"\bthen\b|\bafter that\b|\bfinally\b|;|\n", normalized, flags=re.IGNORECASE)
        subtasks: list[str] = []
        for part in parts:
            candidate = re.sub(r"^(and|then)\s+", "", part.strip(), flags=re.IGNORECASE).strip(" .")
            if not candidate:
                continue
            for chunk in self._split_top_level_actions(candidate):
                cleaned = re.sub(r"^(and|then)\s+", "", chunk.strip(), flags=re.IGNORECASE).strip(" .")
                if cleaned:
                    subtasks.append(cleaned)
        return subtasks or [normalized]

    def _build_shared_context(self, subtask_results: list[dict[str, Any]], max_items: int = 3) -> str:
        if not subtask_results:
            return ""
        out: list[str] = []
        for item in subtask_results[-max_items:]:
            text = str(item.get("output", "")).replace("\n", " ").strip()
            if len(text) > 300:
                text = text[:300] + "..."
            out.append(f"Subtask {item.get('id')} [{item.get('classification')}] by {item.get('model')}: {text}")
        return "\n".join(out)

    def _heuristic_planner_response(self, subtask_text: str) -> ModelResponse:
        plan = (
            "- Identify required output and constraints.\n"
            "- Use tool outputs first if available.\n"
            f"- Return concise result for: {subtask_text}"
        )
        return ModelResponse(
            text=plan,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
            model="heuristic-planner",
            metadata={"fallback": "heuristic"},
        )

    def _tool_only_executor_response(self, subtask_text: str, shared_context: str) -> tuple[ModelResponse, list[dict[str, Any]]]:
        tool_calls, tool_context = self.executor_agent._run_tools(subtask_text)
        fallback_output = tool_context.strip() or "No model providers reachable and no deterministic tool matched this subtask."
        if shared_context:
            fallback_output = f"{fallback_output}\n\nPrior subtask context:\n{shared_context}"
        return (
            ModelResponse(
                text=fallback_output,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                model="tool-only-fallback",
                metadata={"fallback": "tool_only"},
            ),
            tool_calls,
        )

    def _subtask_requires_code(self, subtask_text: str) -> bool:
        lowered = subtask_text.lower()
        return any(token in lowered for token in {"python code", "write code", "code", "implement", "script"})

    def _output_has_code(self, text: str) -> bool:
        lowered = text.lower()
        return "```" in text or "def " in lowered or "class " in lowered

    def _dijkstra_code_template(self) -> str:
        return (
            "```python\n"
            "import heapq\n\n"
            "def dijkstra(graph, start):\n"
            "    dist = {node: float('inf') for node in graph}\n"
            "    prev = {node: None for node in graph}\n"
            "    dist[start] = 0\n"
            "    pq = [(0, start)]\n\n"
            "    while pq:\n"
            "        cur_dist, u = heapq.heappop(pq)\n"
            "        if cur_dist > dist[u]:\n"
            "            continue\n"
            "        for v, w in graph[u]:\n"
            "            nd = cur_dist + w\n"
            "            if nd < dist[v]:\n"
            "                dist[v] = nd\n"
            "                prev[v] = u\n"
            "                heapq.heappush(pq, (nd, v))\n"
            "    return dist, prev\n"
            "```\n"
            "Time complexity: O((V + E) log V). Space complexity: O(V + E)."
        )

    def _generic_code_template(self) -> str:
        return (
            "```python\n"
            "def solve(data):\n"
            "    # Replace with task-specific logic\n"
            "    return data\n"
            "```\n"
            "Time complexity: depends on the final algorithm. Space complexity: depends on input size."
        )

    def _deterministic_subtask_completion(self, subtask_text: str) -> tuple[str, list[dict[str, Any]]]:
        tool_calls, tool_context = self.executor_agent._run_tools(subtask_text)
        parts: list[str] = []
        lowered = subtask_text.lower()

        if tool_context.strip():
            parts.append(tool_context.strip())

        if "dijkstra" in lowered and any(token in lowered for token in {"explain", "algorithm"}):
            parts.append(
                "Dijkstra's algorithm finds shortest paths from a source node in a weighted graph with non-negative edges."
            )

        if self._subtask_requires_code(subtask_text):
            parts.append(self._dijkstra_code_template() if "dijkstra" in lowered else self._generic_code_template())

        if not parts:
            parts.append("This subtask was completed with deterministic fallback due provider/budget constraints.")
        return "\n\n".join(parts), tool_calls

    async def _classify_task_node(self, state: GraphState) -> dict[str, Any]:
        classification = classify_task(state["task"])
        merged = {**state, "classification": classification}
        steps = self._record_step(merged, "classifier", f"classified task as {classification}")
        await self._emit(merged, "step", "Task classified", {"node": "classifier", "classification": classification})
        return {"classification": classification, "steps": steps}

    async def _init_budget_node(self, state: GraphState) -> dict[str, Any]:
        budget = self.budget_tracker.initialize(state["classification"])
        model_usage = self._initial_model_usage()
        model_budgets, savings = self._derive_budget_metrics(model_usage, budget.used, budget.total)
        updates = {
            "budget_total": budget.total,
            "tokens_used": budget.used,
            "tokens_remaining": budget.remaining,
            "selected_model": self.settings.small_model,
            "route_reason": "initialized shared budget for decomposed execution",
            "model_usage": model_usage,
            "model_budgets": model_budgets,
            "orchestration_savings": savings,
        }
        merged = {**state, **updates}
        steps = self._record_step(merged, "router", "initialized budget for orchestrated subtasks", {"route_reason": updates["route_reason"]})
        await self._emit(
            merged,
            "token_update",
            "Budget initialized",
            {
                "budget_total": budget.total,
                "tokens_used": budget.used,
                "tokens_remaining": budget.remaining,
                "model_budgets": model_budgets,
                "orchestration_savings": savings,
            },
        )
        return {**updates, "steps": steps}

    async def _decompose_task_node(self, state: GraphState) -> dict[str, Any]:
        fragments = self._split_task_into_subtasks(state["task"])
        subtasks: list[dict[str, Any]] = []
        lines: list[str] = []
        for idx, fragment in enumerate(fragments, start=1):
            sub_classification = classify_task(fragment)
            subtask = {"id": idx, "text": fragment, "classification": sub_classification, "depends_on": idx - 1 if idx > 1 else None}
            subtasks.append(subtask)
            lines.append(f"{idx}. [{sub_classification}] {fragment}")

        planner_output = "\n".join(lines)
        merged = {**state, "subtasks": subtasks, "planner_output": planner_output}
        steps = self._record_step(merged, "planner", "planner decomposed task into subtasks", {"subtask_count": len(subtasks)})
        await self._emit(merged, "step", "Task decomposed into subtasks", {"node": "planner", "subtasks": subtasks, "subtask_count": len(subtasks)})
        return {"subtasks": subtasks, "planner_output": planner_output, "steps": steps}

    async def _execute_subtasks_node(self, state: GraphState) -> dict[str, Any]:
        subtasks = state.get("subtasks", [])
        if not subtasks:
            return {"executor_output": "", "subtask_results": [], "agent_messages": state.get("agent_messages", [])}

        budget_total = int(state.get("budget_total", 0))
        tokens_used = int(state.get("tokens_used", 0))
        tokens_remaining = int(state.get("tokens_remaining", 0))
        steps = state.get("steps", [])
        model_usage = state.get("model_usage", self._initial_model_usage())
        model_budgets = state.get("model_budgets", {})
        savings = state.get("orchestration_savings", {})
        all_tool_calls = state.get("tool_calls", [])
        subtask_results = state.get("subtask_results", [])
        agent_messages = state.get("agent_messages", [])
        selected_model = state.get("selected_model", self.settings.small_model)
        route_reason = state.get("route_reason", "")
        status = state.get("status", "running")

        for subtask in subtasks:
            if self.budget_tracker.is_exhausted(tokens_remaining):
                status = "budget_exhausted"
                break

            sub_id = int(subtask["id"])
            sub_text = str(subtask["text"])
            sub_classification = str(subtask["classification"])
            shared_context = self._build_shared_context(subtask_results)
            remaining_subtasks = max(1, len(subtasks) - len(subtask_results))
            planner_budget_hint = max(80, tokens_remaining // max(1, remaining_subtasks * 2))
            use_heuristic_planner = (
                sub_classification == "simple"
                or (remaining_subtasks >= 2 and tokens_remaining < 1400)
                or planner_budget_hint < 180
            )

            planner_model = self.settings.small_model
            if use_heuristic_planner:
                planner_resp = self._heuristic_planner_response(sub_text)
                await self._emit(
                    {**state, "selected_model": planner_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                    "route_decision",
                    f"subtask {sub_id} planner used heuristic mode to preserve budget",
                    {"subtask": subtask, "heuristic": True, "planner_budget_hint": planner_budget_hint},
                )
            else:
                try:
                    planner_resp = await self.planner_agent.run(
                        task=sub_text,
                        classification=sub_classification,
                        model=planner_model,
                        tokens_remaining=planner_budget_hint,
                        shared_context=shared_context,
                    )
                except Exception as exc:  # noqa: BLE001
                    if self.settings.large_provider == "groq":
                        planner_model = self.settings.large_model
                        await self._emit(
                            {**state, "selected_model": planner_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                            "route_decision",
                            f"subtask {sub_id} planner small-model connection failed ({exc}); fallback to large planner model",
                            {"subtask": subtask, "fallback": True},
                        )
                        try:
                            planner_resp = await self.planner_agent.run(
                                task=sub_text,
                                classification=sub_classification,
                                model=planner_model,
                                tokens_remaining=planner_budget_hint,
                                shared_context=shared_context,
                            )
                        except Exception as large_exc:  # noqa: BLE001
                            planner_resp = self._heuristic_planner_response(sub_text)
                            planner_model = self.settings.small_model
                            await self._emit(
                                {**state, "selected_model": planner_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                                "error",
                                f"subtask {sub_id} planner unavailable on both providers; using heuristic planner fallback",
                                {"subtask": subtask, "small_error": str(exc), "large_error": str(large_exc)},
                            )
                    else:
                        planner_resp = self._heuristic_planner_response(sub_text)
                        await self._emit(
                            {**state, "selected_model": planner_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                            "error",
                            f"subtask {sub_id} planner small-model call failed; using heuristic planner fallback",
                            {"subtask": subtask, "small_error": str(exc)},
                        )
            snapshot = self.budget_tracker.consume(
                total=budget_total,
                used_so_far=tokens_used,
                prompt_tokens=planner_resp.prompt_tokens,
                completion_tokens=planner_resp.completion_tokens,
            )
            tokens_used, tokens_remaining = snapshot.used, snapshot.remaining
            usage_state = {**state, "tokens_used": tokens_used, "budget_total": budget_total, "model_usage": model_usage}
            model_usage, model_budgets, savings = self._apply_model_usage(usage_state, planner_resp, planner_model)

            planner_message = f"Planner->Executor subtask {sub_id}: {planner_resp.text}"
            agent_messages = [*agent_messages, planner_message]
            selected_model = planner_model
            merged = {**state, "steps": steps, "selected_model": selected_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining}
            steps = self._record_step(
                merged,
                "planner",
                "planner prepared subtask guidance",
                {
                    "subtask_id": sub_id,
                    "subtask_text": sub_text,
                    "subtask_classification": sub_classification,
                    "latency_ms": round(planner_resp.latency_ms, 2),
                    "prompt_tokens": planner_resp.prompt_tokens,
                    "completion_tokens": planner_resp.completion_tokens,
                },
            )
            await self._emit(
                {**merged, "model_budgets": model_budgets, "orchestration_savings": savings},
                "token_update",
                f"Planner guidance ready for subtask {sub_id}",
                {
                    "subtask_id": sub_id,
                    "tokens_used": tokens_used,
                    "tokens_remaining": tokens_remaining,
                    "tokens_in": planner_resp.prompt_tokens,
                    "tokens_out": planner_resp.completion_tokens,
                    "model_budgets": model_budgets,
                    "orchestration_savings": savings,
                },
            )

            exec_model, exec_reason = initial_model(sub_classification, self.settings)
            route_reason = f"subtask {sub_id}/{len(subtasks)} [{sub_classification}] -> {exec_model} ({exec_reason})"
            exec_budget_hint = max(120, tokens_remaining // max(1, remaining_subtasks))
            force_code = self._subtask_requires_code(sub_text)
            await self._emit(
                {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                "route_decision",
                route_reason,
                {"subtask": subtask},
            )

            fallback_used = False
            try:
                exec_resp, tool_calls = await self.executor_agent.run(
                    task=sub_text,
                    classification=sub_classification,
                    plan=planner_resp.text,
                    model=exec_model,
                    tokens_remaining=exec_budget_hint,
                    shared_context=shared_context,
                    agent_messages=agent_messages[-6:],
                    force_code=force_code,
                )
            except Exception as exc:  # noqa: BLE001
                if exec_model == self.settings.large_model:
                    fallback_used = True
                    exec_model = self.settings.small_model
                    route_reason = f"subtask {sub_id} large model failed ({exc}); fallback to small model"
                    await self._emit(
                        {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                        "route_decision",
                        route_reason,
                        {"subtask": subtask, "fallback": True},
                    )
                    try:
                        exec_resp, tool_calls = await self.executor_agent.run(
                            task=sub_text,
                            classification=sub_classification,
                            plan=planner_resp.text,
                            model=exec_model,
                            tokens_remaining=exec_budget_hint,
                            shared_context=shared_context,
                            agent_messages=agent_messages[-6:],
                            force_code=force_code,
                        )
                    except Exception as small_exc:  # noqa: BLE001
                        route_reason = f"subtask {sub_id} executor unavailable on both providers; used tool-only fallback"
                        exec_resp, tool_calls = self._tool_only_executor_response(sub_text, shared_context)
                        await self._emit(
                            {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                            "error",
                            route_reason,
                            {"subtask": subtask, "large_error": str(exc), "small_error": str(small_exc)},
                        )
                elif exec_model == self.settings.small_model and self.settings.large_provider == "groq":
                    fallback_used = True
                    exec_model = self.settings.large_model
                    route_reason = f"subtask {sub_id} small model failed ({exc}); fallback to large model"
                    await self._emit(
                        {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                        "route_decision",
                        route_reason,
                        {"subtask": subtask, "fallback": True},
                    )
                    try:
                        exec_resp, tool_calls = await self.executor_agent.run(
                            task=sub_text,
                            classification=sub_classification,
                            plan=planner_resp.text,
                            model=exec_model,
                            tokens_remaining=exec_budget_hint,
                            shared_context=shared_context,
                            agent_messages=agent_messages[-6:],
                            force_code=force_code,
                        )
                    except Exception as large_exc:  # noqa: BLE001
                        route_reason = f"subtask {sub_id} executor unavailable on both providers; used tool-only fallback"
                        exec_resp, tool_calls = self._tool_only_executor_response(sub_text, shared_context)
                        exec_model = self.settings.small_model
                        await self._emit(
                            {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                            "error",
                            route_reason,
                            {"subtask": subtask, "small_error": str(exc), "large_error": str(large_exc)},
                        )
                else:
                    fallback_used = True
                    route_reason = f"subtask {sub_id} executor model call failed; used tool-only fallback"
                    exec_resp, tool_calls = self._tool_only_executor_response(sub_text, shared_context)
                    await self._emit(
                        {**state, "selected_model": exec_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                        "error",
                        route_reason,
                        {"subtask": subtask, "model_error": str(exc)},
                    )
            snapshot = self.budget_tracker.consume(
                total=budget_total,
                used_so_far=tokens_used,
                prompt_tokens=exec_resp.prompt_tokens,
                completion_tokens=exec_resp.completion_tokens,
            )
            tokens_used, tokens_remaining = snapshot.used, snapshot.remaining
            usage_state = {**state, "tokens_used": tokens_used, "budget_total": budget_total, "model_usage": model_usage}
            model_usage, model_budgets, savings = self._apply_model_usage(usage_state, exec_resp, exec_model)
            selected_model = exec_model
            all_tool_calls = [*all_tool_calls, *tool_calls]
            subtask_output = exec_resp.text

            if sub_classification == "moderate" and exec_model == self.settings.small_model and not self.budget_tracker.is_exhausted(tokens_remaining):
                escalate, reason = should_escalate_moderate(
                    task=sub_text,
                    output_text=subtask_output,
                    tokens_remaining=tokens_remaining,
                    has_escalated=False,
                    escalation_min_tokens=self.settings.escalation_min_tokens,
                )
                if escalate:
                    await self._emit(
                        {**state, "selected_model": self.settings.large_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                        "route_decision",
                        f"subtask {sub_id} escalation -> {self.settings.large_model} ({reason})",
                        {"subtask": subtask, "escalated": True},
                    )
                    try:
                        escalated_resp, escalated_tool_calls = await self.executor_agent.run(
                            task=sub_text,
                            classification="complex",
                            plan=planner_resp.text,
                            model=self.settings.large_model,
                            tokens_remaining=max(120, tokens_remaining // max(1, remaining_subtasks)),
                            shared_context=f"{shared_context}\nSmall-model draft:\n{subtask_output}",
                            agent_messages=agent_messages[-6:],
                            force_code=force_code,
                        )
                        snapshot = self.budget_tracker.consume(
                            total=budget_total,
                            used_so_far=tokens_used,
                            prompt_tokens=escalated_resp.prompt_tokens,
                            completion_tokens=escalated_resp.completion_tokens,
                        )
                        tokens_used, tokens_remaining = snapshot.used, snapshot.remaining
                        usage_state = {**state, "tokens_used": tokens_used, "budget_total": budget_total, "model_usage": model_usage}
                        model_usage, model_budgets, savings = self._apply_model_usage(usage_state, escalated_resp, self.settings.large_model)
                        selected_model = self.settings.large_model
                        all_tool_calls = [*all_tool_calls, *escalated_tool_calls]
                        subtask_output = escalated_resp.text
                        route_reason = f"subtask {sub_id} escalated from small to large"
                    except Exception as exc:  # noqa: BLE001
                        route_reason = f"subtask {sub_id} escalation failed on large ({exc}); kept small-model output"

            if force_code and not self._output_has_code(subtask_output):
                retry_model = self.settings.large_model if self.settings.large_provider == "groq" else exec_model
                retry_budget_hint = max(220, tokens_remaining // max(1, remaining_subtasks))
                if tokens_remaining >= retry_budget_hint:
                    await self._emit(
                        {**state, "selected_model": retry_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                        "route_decision",
                        f"subtask {sub_id} missing code artifact -> retrying with strict code directive",
                        {"subtask": subtask, "enforce_code": True},
                    )
                    try:
                        retry_resp, retry_tool_calls = await self.executor_agent.run(
                            task=sub_text,
                            classification="complex",
                            plan=f"{planner_resp.text}\n\nMandatory: include runnable Python code block and complexity.",
                            model=retry_model,
                            tokens_remaining=retry_budget_hint,
                            shared_context=f"{shared_context}\nPrevious incomplete draft:\n{subtask_output}",
                            agent_messages=agent_messages[-6:],
                            force_code=True,
                        )
                        snapshot = self.budget_tracker.consume(
                            total=budget_total,
                            used_so_far=tokens_used,
                            prompt_tokens=retry_resp.prompt_tokens,
                            completion_tokens=retry_resp.completion_tokens,
                        )
                        tokens_used, tokens_remaining = snapshot.used, snapshot.remaining
                        usage_state = {**state, "tokens_used": tokens_used, "budget_total": budget_total, "model_usage": model_usage}
                        model_usage, model_budgets, savings = self._apply_model_usage(usage_state, retry_resp, retry_model)
                        all_tool_calls = [*all_tool_calls, *retry_tool_calls]
                        tool_calls = [*tool_calls, *retry_tool_calls]
                        selected_model = retry_model
                        exec_resp = retry_resp
                        subtask_output = retry_resp.text
                        if self._output_has_code(subtask_output):
                            route_reason = f"subtask {sub_id} strict code retry succeeded"
                        else:
                            deterministic, extra_tool_calls = self._deterministic_subtask_completion(sub_text)
                            subtask_output = f"{subtask_output}\n\n{deterministic}"
                            all_tool_calls = [*all_tool_calls, *extra_tool_calls]
                            route_reason = f"subtask {sub_id} strict code retry missing code; appended deterministic code template"
                    except Exception as retry_exc:  # noqa: BLE001
                        deterministic, extra_tool_calls = self._deterministic_subtask_completion(sub_text)
                        subtask_output = f"{subtask_output}\n\n{deterministic}"
                        all_tool_calls = [*all_tool_calls, *extra_tool_calls]
                        route_reason = f"subtask {sub_id} strict code retry failed ({retry_exc}); appended deterministic code template"
                        await self._emit(
                            {**state, "selected_model": selected_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                            "error",
                            route_reason,
                            {"subtask": subtask, "retry_error": str(retry_exc)},
                        )
                else:
                    deterministic, extra_tool_calls = self._deterministic_subtask_completion(sub_text)
                    subtask_output = f"{subtask_output}\n\n{deterministic}"
                    all_tool_calls = [*all_tool_calls, *extra_tool_calls]
                    route_reason = f"subtask {sub_id} insufficient budget for strict retry; appended deterministic code template"

            sub_result = {
                "id": sub_id,
                "text": sub_text,
                "classification": sub_classification,
                "model": selected_model,
                "output": subtask_output,
                "fallback_used": fallback_used,
            }
            subtask_results = [*subtask_results, sub_result]
            agent_messages = [*agent_messages, f"Executor->Planner subtask {sub_id}: completed with {selected_model}"]

            merged = {**state, "steps": steps, "selected_model": selected_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining}
            steps = self._record_step(
                merged,
                "executor",
                "executor finished routed subtask",
                {
                    "subtask_id": sub_id,
                    "subtask_text": sub_text,
                    "subtask_classification": sub_classification,
                    "route_reason": route_reason,
                    "latency_ms": round(exec_resp.latency_ms, 2),
                    "prompt_tokens": exec_resp.prompt_tokens,
                    "completion_tokens": exec_resp.completion_tokens,
                    "tool_calls_count": len(tool_calls),
                    "fallback_used": fallback_used,
                },
            )
            await self._emit(
                {**merged, "model_budgets": model_budgets, "orchestration_savings": savings},
                "step",
                f"Subtask {sub_id} completed",
                {"node": "executor", "subtask": sub_result, "tool_calls_count": len(tool_calls)},
            )
            await self._emit(
                {**merged, "model_budgets": model_budgets, "orchestration_savings": savings},
                "token_update",
                f"Subtask {sub_id} tokens accounted",
                {
                    "subtask_id": sub_id,
                    "tokens_used": tokens_used,
                    "tokens_remaining": tokens_remaining,
                    "tokens_in": exec_resp.prompt_tokens,
                    "tokens_out": exec_resp.completion_tokens,
                    "model_budgets": model_budgets,
                    "orchestration_savings": savings,
                },
            )
            if self.budget_tracker.is_exhausted(tokens_remaining):
                status = "budget_exhausted"
                break

        if len(subtask_results) < len(subtasks):
            pending = subtasks[len(subtask_results) :]
            for pending_subtask in pending:
                pending_id = int(pending_subtask["id"])
                pending_text = str(pending_subtask["text"])
                pending_classification = str(pending_subtask["classification"])
                deterministic, extra_tool_calls = self._deterministic_subtask_completion(pending_text)
                all_tool_calls = [*all_tool_calls, *extra_tool_calls]
                sub_result = {
                    "id": pending_id,
                    "text": pending_text,
                    "classification": pending_classification,
                    "model": "deterministic-fallback",
                    "output": deterministic,
                    "fallback_used": True,
                }
                subtask_results = [*subtask_results, sub_result]
                agent_messages = [*agent_messages, f"Executor->Planner subtask {pending_id}: completed with deterministic fallback"]
                await self._emit(
                    {**state, "selected_model": selected_model, "tokens_used": tokens_used, "tokens_remaining": tokens_remaining},
                    "step",
                    f"Subtask {pending_id} completed via deterministic fallback",
                    {"node": "executor", "subtask": sub_result, "tool_calls_count": len(extra_tool_calls)},
                )

        combined = "\n\n".join(
            f"Subtask {x['id']} ({x['classification']} -> {x['model']}):\n{x['output']}" for x in subtask_results
        ).strip()
        return {
            "steps": steps,
            "tool_calls": all_tool_calls,
            "subtask_results": subtask_results,
            "agent_messages": agent_messages,
            "executor_output": combined or "No subtask output generated.",
            "selected_model": selected_model,
            "tokens_used": tokens_used,
            "tokens_remaining": tokens_remaining,
            "status": status,
            "route_reason": route_reason,
            "model_usage": model_usage,
            "model_budgets": model_budgets,
            "orchestration_savings": savings,
        }

    async def _budget_guard_node(self, state: GraphState) -> dict[str, Any]:
        should_summarize = self.budget_tracker.needs_summarization(state["tokens_remaining"])
        status = state.get("status", "running")
        has_code_subtask = any(self._subtask_requires_code(str(x.get("text", ""))) for x in state.get("subtasks", []))
        if has_code_subtask and self._output_has_code(state.get("executor_output", "")):
            should_summarize = False
        if self.budget_tracker.is_exhausted(state["tokens_remaining"]):
            should_summarize = True
            status = "budget_exhausted"
        merged = {**state, "status": status, "should_summarize": should_summarize}
        steps = self._record_step(merged, "budget_guard", "budget and summary checks completed", {"should_summarize": should_summarize})
        await self._emit(
            merged,
            "state_update",
            "Budget guard updated workflow state",
            {
                "status": status,
                "should_summarize": should_summarize,
                "model_budgets": state.get("model_budgets", {}),
                "orchestration_savings": state.get("orchestration_savings", {}),
            },
        )
        return {
            "steps": steps,
            "status": status,
            "should_summarize": should_summarize,
            "model_budgets": state.get("model_budgets", {}),
            "orchestration_savings": state.get("orchestration_savings", {}),
        }

    def _budget_guard_route(self, state: GraphState) -> str:
        return "summarizer" if state.get("should_summarize") else "finalize"

    async def _summarizer_node(self, state: GraphState) -> dict[str, Any]:
        source = state.get("executor_output") or state.get("final_output") or ""
        if not source.strip():
            source = "No output available to summarize."
        summary_model = self.settings.small_model
        try:
            response = await self.summarizer_agent.run(source, summary_model, state["tokens_remaining"])
            prompt_tokens, completion_tokens, summary_text, latency_ms = response.prompt_tokens, response.completion_tokens, response.text, response.latency_ms
        except Exception as exc:  # noqa: BLE001
            if self.settings.large_provider == "groq":
                summary_model = self.settings.large_model
                await self._emit(
                    {**state, "selected_model": summary_model},
                    "route_decision",
                    f"summarizer small-model call failed ({exc}); fallback to large summarizer model",
                    {"node": "summarizer", "fallback": True},
                )
                try:
                    response = await self.summarizer_agent.run(source, summary_model, state["tokens_remaining"])
                    prompt_tokens, completion_tokens, summary_text, latency_ms = response.prompt_tokens, response.completion_tokens, response.text, response.latency_ms
                except Exception as large_exc:  # noqa: BLE001
                    summary_text, prompt_tokens, completion_tokens, latency_ms = source[:420], 0, 0, 0.0
                    summary_model = self.settings.small_model
                    await self._emit(
                        state,
                        "error",
                        "Summarizer unavailable on both providers; used truncated fallback",
                        {"node": "summarizer", "small_error": str(exc), "large_error": str(large_exc)},
                    )
            else:
                summary_text, prompt_tokens, completion_tokens, latency_ms = source[:420], 0, 0, 0.0
                await self._emit(state, "error", f"Summarizer failed ({exc}); used truncated fallback", {"node": "summarizer"})

        snapshot = self.budget_tracker.consume(state["budget_total"], state["tokens_used"], prompt_tokens, completion_tokens)
        usage_state = {**state, "tokens_used": snapshot.used, "budget_total": state["budget_total"], "model_usage": state.get("model_usage", self._initial_model_usage())}
        fake_resp = ModelResponse(summary_text, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, latency_ms, summary_model, {})
        model_usage, model_budgets, savings = self._apply_model_usage(usage_state, fake_resp, summary_model)
        status = "budget_exhausted" if self.budget_tracker.is_exhausted(snapshot.remaining) else "completed"
        merged = {**state, "tokens_used": snapshot.used, "tokens_remaining": snapshot.remaining, "selected_model": summary_model, "status": status, "model_usage": model_usage, "model_budgets": model_budgets, "orchestration_savings": savings}
        steps = self._record_step(merged, "summarizer", "summarizer compressed final output", {"latency_ms": round(latency_ms, 2), "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        await self._emit(merged, "token_update", "Summarizer tokens accounted", {"tokens_used": snapshot.used, "tokens_remaining": snapshot.remaining, "tokens_in": prompt_tokens, "tokens_out": completion_tokens, "model_budgets": model_budgets, "orchestration_savings": savings})
        await self._emit(merged, "step", "Summarizer completed", {"node": "summarizer"})
        return {"steps": steps, "tokens_used": snapshot.used, "tokens_remaining": snapshot.remaining, "selected_model": summary_model, "status": status, "final_output": summary_text, "should_summarize": False, "model_usage": model_usage, "model_budgets": model_budgets, "orchestration_savings": savings}

    async def _finalize_node(self, state: GraphState) -> dict[str, Any]:
        final_output = state.get("final_output") or state.get("executor_output") or state.get("planner_output") or ""
        status = "completed" if state.get("status", "running") == "running" else state.get("status", "completed")
        merged = {**state, "status": status, "final_output": final_output}
        steps = self._record_step(merged, "finalize", "workflow finalized", {"final_output_length": len(final_output)})
        await self._emit(
            merged,
            "state_update",
            "Workflow finalized",
            {
                "status": status,
                "final_output_length": len(final_output),
                "model_budgets": state.get("model_budgets", {}),
                "orchestration_savings": state.get("orchestration_savings", {}),
                "subtask_results": state.get("subtask_results", []),
            },
        )
        return {
            "steps": steps,
            "status": status,
            "final_output": final_output,
            "should_summarize": False,
            "model_budgets": state.get("model_budgets", {}),
            "orchestration_savings": state.get("orchestration_savings", {}),
            "model_usage": state.get("model_usage", self._initial_model_usage()),
            "subtask_results": state.get("subtask_results", []),
            "agent_messages": state.get("agent_messages", []),
        }

    async def run(self, run_id: str, task: str, event_callback: EventCallback) -> GraphState:
        initial: GraphState = {
            "run_id": run_id,
            "task": task,
            "classification": "",
            "selected_model": "",
            "budget_total": 0,
            "tokens_used": 0,
            "tokens_remaining": 0,
            "steps": [],
            "tool_calls": [],
            "planner_output": "",
            "executor_output": "",
            "final_output": "",
            "status": "running",
            "route_reason": "",
            "should_summarize": False,
            "model_usage": self._initial_model_usage(),
            "model_budgets": {},
            "orchestration_savings": {},
            "subtasks": [],
            "subtask_results": [],
            "agent_messages": [],
            "event_callback": event_callback,
        }
        return await self.graph.ainvoke(initial)

    async def health(self) -> dict[str, Any]:
        return await self.client.health()
