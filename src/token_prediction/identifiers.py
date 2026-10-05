from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import quote


def identifier(namespace: str, *parts: object) -> str:


    values = (namespace, *(str(part) for part in parts))
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError("identifier components must be non-empty")
    return ":".join(quote(value, safe="-._~=@+/") for value in normalized)


def assignment_identifier(
    namespace: str,
    *,
    policy_id: str,
    seed: int | None,
    assignments: Iterable[tuple[object, object]],
) -> str:


    seed_part = "explicit" if seed is None else f"seed={seed}"
    mapping = "+".join(f"{left}={right}" for left, right in assignments)
    return identifier(namespace, policy_id, seed_part, mapping or "empty")


__all__ = ["assignment_identifier", "identifier"]
