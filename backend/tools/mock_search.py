from __future__ import annotations

from pathlib import Path


def search_local_kb(query: str, kb_path: Path, top_k: int = 3) -> list[str]:
    if not kb_path.exists():
        return ["Local KB file missing."]

    terms = [term for term in query.lower().split() if len(term) > 2]
    if not terms:
        return ["No meaningful terms to search."]

    matches: list[tuple[int, str]] = []
    for line in kb_path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        score = sum(1 for term in terms if term in normalized.lower())
        if score:
            matches.append((score, normalized))

    if not matches:
        return ["No matching results found in local KB."]
    ranked = sorted(matches, key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked[:top_k]]

