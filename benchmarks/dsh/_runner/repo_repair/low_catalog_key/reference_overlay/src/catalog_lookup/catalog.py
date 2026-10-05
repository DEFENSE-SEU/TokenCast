from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class DuplicateItemError(ValueError):


    def __init__(self, canonical_key: str, first_key: str, duplicate_key: str) -> None:
        self.canonical_key = canonical_key
        self.first_key = first_key
        self.duplicate_key = duplicate_key
        super().__init__(
            f"duplicate canonical key {canonical_key!r}: {first_key!r} and {duplicate_key!r}"
        )


def _canonical_key(text: str) -> str:
    return text.strip().casefold()


class Catalog:


    def __init__(self, items: Iterable[tuple[str, Any]]) -> None:
        self._items: dict[str, Any] = {}
        source_keys: dict[str, str] = {}
        for key, value in items:
            canonical_key = _canonical_key(key)
            if canonical_key in self._items:
                raise DuplicateItemError(canonical_key, source_keys[canonical_key], key)
            self._items[canonical_key] = value
            source_keys[canonical_key] = key

    def lookup(self, query: str) -> Any:


        return self._items[_canonical_key(query)]
