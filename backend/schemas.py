from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="User task to execute")


class RunResponse(BaseModel):
    run_id: str


class RunEvent(BaseModel):
    event_type: str
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    classification: Optional[str] = None
    model: Optional[str] = None
    tokens_used: int = 0
    tokens_remaining: int = 0
    message: str = ""
    step_data: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    ollama_reachable: bool
    groq_configured: bool = False
    groq_reachable: bool = False
    large_provider: str = "ollama"
    configured_models: Dict[str, str]
    available_models: list[str] = Field(default_factory=list)
    groq_models: list[str] = Field(default_factory=list)
    missing_models: list[str] = Field(default_factory=list)
