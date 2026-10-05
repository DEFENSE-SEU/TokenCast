from __future__ import annotations

import json
from typing import Any

__all__ = ["canonical_json_bytes", "strict_json_object"]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:


    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output
