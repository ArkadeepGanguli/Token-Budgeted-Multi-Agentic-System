from backend.classifier import classify_task


def test_required_scenarios_classification() -> None:
    assert classify_task("Summarize this paragraph") == "simple"
    assert classify_task("Explain Dijkstra algorithm") == "complex"
    assert classify_task("Format this JSON: {\"a\":1}") == "simple"
    assert classify_task("Write optimized Python code for sorting") == "complex"


def test_moderate_intent_classification() -> None:
    assert classify_task("Compare tradeoffs between two caching strategies") == "moderate"


def test_calculation_routes_simple() -> None:
    assert classify_task("Calculate 1250*37/5") == "simple"
