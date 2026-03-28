from pathlib import Path

from backend.tools.calculator import calculate
from backend.tools.formatter import format_json, format_text
from backend.tools.mock_search import search_local_kb


def test_calculator() -> None:
    assert calculate("2 + 3 * 4") == "14"
    assert "error" in calculate("import os").lower()


def test_formatter() -> None:
    assert '"a": 1' in format_json('{"a":1}')
    assert "not valid json" in format_json("oops").lower()
    assert "\n" in format_text("a b c d e f g h i j k", width=10)


def test_mock_search(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb.txt"
    kb_path.write_text("alpha token budget\nbeta routing\n", encoding="utf-8")
    results = search_local_kb("routing token", kb_path)
    assert results
    assert "token" in " ".join(results).lower() or "routing" in " ".join(results).lower()

