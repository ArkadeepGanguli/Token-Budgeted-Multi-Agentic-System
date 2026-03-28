from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any, Dict
from urllib.error import HTTPError, URLError
import urllib.request

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.budget import estimate_tokens
from backend.config import Settings


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    metadata: Dict[str, Any]


class OllamaClient:
    """Hybrid model client: local Ollama for small model, Groq API for large model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm_cache: Dict[str, ChatOllama] = {}

    def _get_ollama_llm(self, model: str) -> ChatOllama:
        if model not in self._llm_cache:
            self._llm_cache[model] = ChatOllama(
                model=model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
                timeout=self.settings.ollama_timeout_seconds,
            )
        return self._llm_cache[model]

    async def _chat_ollama(self, model: str, system_prompt: str, user_prompt: str) -> ModelResponse:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        llm = self._get_ollama_llm(model)
        started = time.perf_counter()
        response = await llm.ainvoke(messages)
        latency_ms = (time.perf_counter() - started) * 1000

        text = str(getattr(response, "content", "")).strip()
        metadata = getattr(response, "response_metadata", {}) or {}
        prompt_tokens = (
            metadata.get("prompt_eval_count")
            or metadata.get("input_tokens")
            or estimate_tokens(system_prompt + "\n" + user_prompt)
        )
        completion_tokens = (
            metadata.get("eval_count")
            or metadata.get("output_tokens")
            or estimate_tokens(text)
        )
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        return ModelResponse(
            text=text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=model,
            metadata=metadata,
        )

    def _sync_groq_chat(self, model: str, system_prompt: str, user_prompt: str) -> ModelResponse:
        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        payload = {
            "model": model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.settings.groq_base_url.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "Accept": "application/json",
                "User-Agent": "token-budget-multi-agent/1.0",
            },
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.settings.groq_timeout_seconds) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise RuntimeError(f"Groq API error {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Groq network error: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choices = raw_payload.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = str(message.get("content", "")).strip()
        usage = raw_payload.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(system_prompt + "\n" + user_prompt))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=model,
            metadata=raw_payload,
        )

    async def _chat_groq(self, model: str, system_prompt: str, user_prompt: str) -> ModelResponse:
        return await asyncio.to_thread(self._sync_groq_chat, model, system_prompt, user_prompt)

    async def chat(self, model: str, system_prompt: str, user_prompt: str) -> ModelResponse:
        use_groq_large = (
            self.settings.large_provider == "groq"
            and model == self.settings.large_model
        )
        if use_groq_large:
            return await self._chat_groq(model, system_prompt, user_prompt)
        return await self._chat_ollama(model, system_prompt, user_prompt)

    async def get_available_models(self) -> list[str]:
        def _fetch_tags() -> list[str]:
            url = f"{self.settings.ollama_base_url.rstrip('/')}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("models", [])
            return [str(item.get("name", "")) for item in models if item.get("name")]

        return await asyncio.to_thread(_fetch_tags)

    def _sync_get_groq_models(self) -> list[str]:
        if not self.settings.groq_api_key:
            return []
        url = f"{self.settings.groq_base_url.rstrip('/')}/models"
        request = urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "Accept": "application/json",
                "User-Agent": "token-budget-multi-agent/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=self.settings.groq_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data", [])
        return [str(item.get("id", "")) for item in data if item.get("id")]

    async def get_groq_models(self) -> list[str]:
        return await asyncio.to_thread(self._sync_get_groq_models)

    async def health(self) -> dict[str, Any]:
        ollama_reachable = False
        available_ollama: list[str] = []
        missing: list[str] = []
        try:
            available_ollama = await self.get_available_models()
            ollama_reachable = True
            if self.settings.small_model not in available_ollama:
                missing.append(self.settings.small_model)
        except Exception:  # noqa: BLE001
            ollama_reachable = False
            missing.append(self.settings.small_model)

        groq_configured = bool(self.settings.groq_api_key)
        groq_reachable = False
        groq_models: list[str] = []
        if self.settings.large_provider == "groq":
            if not groq_configured:
                missing.append(self.settings.large_model)
            else:
                try:
                    groq_models = await self.get_groq_models()
                    groq_reachable = True
                    if self.settings.large_model not in groq_models:
                        missing.append(self.settings.large_model)
                except Exception:  # noqa: BLE001
                    groq_reachable = False
                    missing.append(self.settings.large_model)
        else:
            if self.settings.large_model not in available_ollama:
                missing.append(self.settings.large_model)

        return {
            "ok": ollama_reachable and (groq_reachable if self.settings.large_provider == "groq" else True),
            "ollama_reachable": ollama_reachable,
            "available_models": available_ollama,
            "groq_models": groq_models,
            "groq_configured": groq_configured,
            "groq_reachable": groq_reachable,
            "large_provider": self.settings.large_provider,
            "missing_models": sorted(set(missing)),
        }
