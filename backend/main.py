from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.graph import MultiAgentWorkflow
from backend.schemas import HealthResponse, RunEvent, RunRequest, RunResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

settings = get_settings()
workflow = MultiAgentWorkflow(settings=settings)

app = FastAPI(title="Token-Budgeted Local Multi-Agent System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class RunContext:
    queue: asyncio.Queue[str | None]
    status: str = "running"


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunContext] = {}
        self._lock = asyncio.Lock()

    async def create(self, run_id: str) -> None:
        async with self._lock:
            self._runs[run_id] = RunContext(queue=asyncio.Queue())

    async def get(self, run_id: str) -> RunContext | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def publish(self, run_id: str, event: RunEvent) -> None:
        context = await self.get(run_id)
        if context is None:
            return
        payload = event.model_dump_json()
        message = f"event: {event.event_type}\ndata: {payload}\n\n"
        await context.queue.put(message)

    async def finish(self, run_id: str, status: str) -> None:
        context = await self.get(run_id)
        if context is None:
            return
        context.status = status
        await context.queue.put(None)


run_manager = RunManager()


async def _execute_run(run_id: str, task: str) -> None:
    async def callback(event: RunEvent) -> None:
        await run_manager.publish(run_id, event)

    try:
        final_state = await workflow.run(run_id=run_id, task=task, event_callback=callback)
        done_event = RunEvent(
            event_type="done",
            run_id=run_id,
            classification=final_state.get("classification"),
            model=final_state.get("selected_model"),
            tokens_used=final_state.get("tokens_used", 0),
            tokens_remaining=final_state.get("tokens_remaining", 0),
            message="Run completed",
            step_data={
                "status": final_state.get("status"),
                "final_output": final_state.get("final_output", ""),
                "steps": final_state.get("steps", []),
                "tool_calls": final_state.get("tool_calls", []),
                "subtask_results": final_state.get("subtask_results", []),
                "agent_messages": final_state.get("agent_messages", []),
                "model_budgets": final_state.get("model_budgets", {}),
                "orchestration_savings": final_state.get("orchestration_savings", {}),
            },
        )
        await run_manager.publish(run_id, done_event)
        await run_manager.finish(run_id, status=final_state.get("status", "completed"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Run failed", extra={"run_id": run_id})
        event = RunEvent(
            event_type="error",
            run_id=run_id,
            message=f"Run failed: {exc}",
            step_data={"error": str(exc)},
        )
        await run_manager.publish(run_id, event)
        await run_manager.finish(run_id, status="error")


@app.post("/api/run", response_model=RunResponse)
async def start_run(payload: RunRequest) -> RunResponse:
    run_id = str(uuid4())
    await run_manager.create(run_id)
    asyncio.create_task(_execute_run(run_id, payload.task))
    return RunResponse(run_id=run_id)


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    context = await run_manager.get(run_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Run ID not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(context.queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield item

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    health_data = await workflow.health()
    return HealthResponse(
        ok=bool(health_data.get("ok")),
        ollama_reachable=bool(health_data.get("ollama_reachable")),
        groq_configured=bool(health_data.get("groq_configured")),
        groq_reachable=bool(health_data.get("groq_reachable")),
        large_provider=str(health_data.get("large_provider", settings.large_provider)),
        configured_models={"small": settings.small_model, "large": settings.large_model},
        available_models=health_data.get("available_models", []),
        groq_models=health_data.get("groq_models", []),
        missing_models=health_data.get("missing_models", []),
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found: %s", FRONTEND_DIR)
