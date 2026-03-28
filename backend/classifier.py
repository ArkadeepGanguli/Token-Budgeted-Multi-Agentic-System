from __future__ import annotations

from typing import Dict


SIMPLE_KEYWORDS = {
    "summarize",
    "summary",
    "format",
    "reformat",
    "calculate",
    "compute",
    "math",
    "search",
    "lookup",
    "fix grammar",
    "clean up",
    "shorten",
    "json format",
}

MODERATE_KEYWORDS = {
    "compare",
    "outline",
    "plan",
    "draft",
    "evaluate",
    "tradeoff",
    "pros and cons",
}

COMPLEX_KEYWORDS = {
    "algorithm",
    "optimize",
    "optimization",
    "dijkstra",
    "reason",
    "proof",
    "architecture",
    "write code",
    "python code",
    "debug",
    "design system",
}


def _keyword_score(text: str, keywords: set[str], weight: int = 2) -> int:
    return sum(weight for keyword in keywords if keyword in text)


def classify_task(task: str) -> str:
    text = task.strip().lower()
    if not text:
        return "simple"

    scores: Dict[str, int] = {"simple": 0, "moderate": 0, "complex": 0}
    scores["simple"] += _keyword_score(text, SIMPLE_KEYWORDS, weight=3)
    scores["moderate"] += _keyword_score(text, MODERATE_KEYWORDS, weight=2)
    scores["complex"] += _keyword_score(text, COMPLEX_KEYWORDS, weight=3)

    word_count = len(text.split())
    if word_count > 30:
        scores["complex"] += 2
    elif word_count > 15:
        scores["moderate"] += 1

    if any(token in text for token in {"explain", "why", "how"}):
        scores["moderate"] += 1
    if any(token in text for token in {"calculate", "compute", "format this json"}):
        scores["simple"] += 2
    if any(token in text for token in {"implement", "code", "optimize"}):
        scores["complex"] += 2
    if any(token in text for token in {"format", "summarize"}):
        scores["simple"] += 2

    # Favor moderate for near ties to avoid overusing large models.
    ranking = sorted(scores.items(), key=lambda item: (item[1], item[0] == "moderate"))
    winner = ranking[-1][0]
    return winner
