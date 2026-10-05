from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from token_prediction.contracts import CanonicalEvent


def write_canonical_jsonl(path: str | Path, events: Iterable[CanonicalEvent]) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite canonical trajectory: {destination}")
    materialized = tuple(events)
    if not materialized:
        raise ValueError("cannot write an empty canonical trajectory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
            for event in materialized
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
