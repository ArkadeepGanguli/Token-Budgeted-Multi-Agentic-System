from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def estimate_tokens(text: str) -> int:
    # Rough approximation for local budget control when provider metadata is missing.
    return max(1, (len(text) + 3) // 4)


@dataclass
class BudgetSnapshot:
    total: int
    used: int
    remaining: int


class BudgetTracker:
    def __init__(self, budget_limits: Dict[str, int], summarize_threshold: int = 80) -> None:
        self.budget_limits = budget_limits
        self.summarize_threshold = summarize_threshold

    def get_budget(self, classification: str) -> int:
        return self.budget_limits.get(classification, self.budget_limits["moderate"])

    def initialize(self, classification: str) -> BudgetSnapshot:
        total = self.get_budget(classification)
        return BudgetSnapshot(total=total, used=0, remaining=total)

    def consume(
        self,
        total: int,
        used_so_far: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> BudgetSnapshot:
        used = max(0, used_so_far + max(prompt_tokens, 0) + max(completion_tokens, 0))
        remaining = max(0, total - used)
        return BudgetSnapshot(total=total, used=used, remaining=remaining)

    def is_exhausted(self, remaining: int) -> bool:
        return remaining <= 0

    def needs_summarization(self, remaining: int) -> bool:
        return remaining <= self.summarize_threshold

