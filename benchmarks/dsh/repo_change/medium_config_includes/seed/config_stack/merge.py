from __future__ import annotations

from typing import Any


def merge_values(earlier: dict[str, Any], later: dict[str, Any]) -> dict[str, Any]:


    merged = dict(earlier)
    merged.update(later)
    return merged
