from __future__ import annotations

import json
import textwrap


def format_json(raw_text: str) -> str:
    try:
        parsed = json.loads(raw_text)
        return json.dumps(parsed, indent=2, ensure_ascii=True, sort_keys=True)
    except json.JSONDecodeError:
        return "Input is not valid JSON."


def format_text(raw_text: str, width: int = 88) -> str:
    normalized = " ".join(raw_text.split())
    return textwrap.fill(normalized, width=width)

