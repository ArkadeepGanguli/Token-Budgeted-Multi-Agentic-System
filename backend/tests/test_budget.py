from backend.budget import BudgetTracker, estimate_tokens


def test_estimate_tokens_non_zero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_budget_tracker_consume_and_exhaustion() -> None:
    tracker = BudgetTracker({"simple": 200, "moderate": 800, "complex": 2000}, summarize_threshold=80)
    initial = tracker.initialize("simple")
    assert initial.total == 200
    assert initial.remaining == 200

    after = tracker.consume(total=200, used_so_far=0, prompt_tokens=90, completion_tokens=120)
    assert after.used == 210
    assert after.remaining == 0
    assert tracker.is_exhausted(after.remaining) is True
    assert tracker.needs_summarization(after.remaining) is True

