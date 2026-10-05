from __future__ import annotations

import json
from collections import Counter
from typing import Iterator


def _tokens(text: str) -> Iterator[str]:
    current: list[str] = []
    for character in text:
        if character.isalnum():
            current.append(character)
        elif current:
            yield "".join(current)
            current.clear()
    if current:
        yield "".join(current)


def count_terms(text: str) -> dict[str, int]:


    counts = Counter(token.casefold() for token in _tokens(text))
    return dict(sorted(counts.items()))


def tally_utf8(data: bytes) -> bytes:


    counts = count_terms(data.decode("utf-8", errors="strict"))
    report = {
        "counts": counts,
        "total_terms": sum(counts.values()),
        "unique_terms": len(counts),
        "version": 1,
    }
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + b"\n"
