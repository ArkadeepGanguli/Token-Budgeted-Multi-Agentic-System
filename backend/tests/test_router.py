from backend.config import Settings
from backend.router import initial_model, should_escalate_moderate


def test_initial_model_rules() -> None:
    settings = Settings()
    model_simple, _ = initial_model("simple", settings)
    model_moderate, _ = initial_model("moderate", settings)
    model_complex, _ = initial_model("complex", settings)

    assert model_simple == settings.small_model
    assert model_moderate == settings.small_model
    assert model_complex == settings.large_model


def test_escalation_rule_for_shallow_moderate_answer() -> None:
    should_escalate, reason = should_escalate_moderate(
        task="Compare two database indexing strategies with examples.",
        output_text="It depends. Consider context.",
        tokens_remaining=600,
        has_escalated=False,
        escalation_min_tokens=280,
    )
    assert should_escalate is True
    assert "escalating" in reason


def test_no_escalation_when_budget_too_low() -> None:
    should_escalate, reason = should_escalate_moderate(
        task="Compare two database indexing strategies with examples.",
        output_text="short",
        tokens_remaining=100,
        has_escalated=False,
        escalation_min_tokens=280,
    )
    assert should_escalate is False
    assert "insufficient remaining budget" in reason

