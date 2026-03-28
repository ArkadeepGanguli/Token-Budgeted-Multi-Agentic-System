from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict


@dataclass(frozen=True)
class Settings:
    small_model: str = os.getenv("SMALL_MODEL", "qwen2:1.5b")
    large_model: str = os.getenv("LARGE_MODEL", "llama-3.1-8b-instant")
    large_provider: str = os.getenv("LARGE_PROVIDER", "groq").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_timeout_seconds: int = int(os.getenv("GROQ_TIMEOUT_SECONDS", "90"))
    summarize_threshold: int = int(os.getenv("SUMMARIZE_THRESHOLD", "80"))
    escalation_min_tokens: int = int(os.getenv("ESCALATION_MIN_TOKENS", "280"))
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    small_model_token_cap: int = int(os.getenv("SMALL_MODEL_TOKEN_CAP", "4000"))
    large_model_token_cap: int = int(os.getenv("LARGE_MODEL_TOKEN_CAP", "2500"))
    # Optional pricing knobs. Keep local model at zero by default.
    # Default rates are configurable proxies for cost tracking:
    # - large: Groq-style 8B pricing tier
    # - small: low-cost small-model hosting tier baseline
    small_model_input_usd_per_1m: float = float(os.getenv("SMALL_MODEL_INPUT_USD_PER_1M", "0.10"))
    small_model_output_usd_per_1m: float = float(os.getenv("SMALL_MODEL_OUTPUT_USD_PER_1M", "0.10"))
    large_model_input_usd_per_1m: float = float(os.getenv("LARGE_MODEL_INPUT_USD_PER_1M", "0.05"))
    large_model_output_usd_per_1m: float = float(os.getenv("LARGE_MODEL_OUTPUT_USD_PER_1M", "0.08"))
    budget_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "simple": int(os.getenv("BUDGET_SIMPLE", "200")),
            "moderate": int(os.getenv("BUDGET_MODERATE", "800")),
            "complex": int(os.getenv("BUDGET_COMPLEX", "3000")),
        }
    )


def get_settings() -> Settings:
    return Settings()
