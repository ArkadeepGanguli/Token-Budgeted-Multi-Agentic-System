from __future__ import annotations

from typing import Tuple

from backend.config import Settings


LOW_CONFIDENCE_MARKERS = {
    "i'm not sure",
    "not sure",
    "might be",
    "cannot guarantee",
    "insufficient context",
    "it depends",
}


def initial_model(classification: str, settings: Settings) -> Tuple[str, str]:
    if classification == "simple":
        return settings.small_model, "simple task -> small model"
    if classification == "moderate":
        return settings.small_model, "moderate task -> start with small model"
    return settings.large_model, "complex task -> large model"


def should_escalate_moderate(
    task: str,
    output_text: str,
    tokens_remaining: int,
    has_escalated: bool,
    escalation_min_tokens: int,
) -> Tuple[bool, str]:
    if has_escalated:
        return False, "already escalated once"
    if tokens_remaining < escalation_min_tokens:
        return False, "insufficient remaining budget for escalation"

    normalized_output = output_text.strip().lower()
    output_word_count = len(normalized_output.split())
    task_word_count = max(1, len(task.split()))
    too_short = output_word_count < max(70, task_word_count * 3)
    low_confidence = any(marker in normalized_output for marker in LOW_CONFIDENCE_MARKERS)
    code_requested = "code" in task.lower() or "algorithm" in task.lower()
    missing_code_structure = code_requested and "```" not in output_text and output_word_count < 140

    if low_confidence:
        return True, "moderate task -> low-confidence response, escalating"
    if too_short:
        return True, "moderate task -> shallow response, escalating"
    if missing_code_structure:
        return True, "moderate task -> missing technical depth, escalating"
    return False, "moderate task resolved by small model"

