from __future__ import annotations

from typing import Any


def merge_values(earlier: dict[str, Any], later: dict[str, Any]) -> dict[str, Any]:


    merged = dict(earlier)
    for key, value in later.items():
        current = merged.get(key)
        if key in merged and isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_values(current, value)
        else:
            merged[key] = value
    return merged
