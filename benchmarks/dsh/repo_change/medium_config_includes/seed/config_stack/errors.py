from __future__ import annotations

from pathlib import Path
from typing import Iterable


class ConfigFormatError(ValueError):


    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"invalid configuration {self.path}: {reason}")


class IncludeCycleError(ValueError):


    def __init__(self, chain: Iterable[str]) -> None:
        self.chain = tuple(chain)
        super().__init__("include cycle: " + " -> ".join(self.chain))


class IncludeOutsideRootError(ValueError):


    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"configuration path is outside root: {self.path}")
