from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PlanningInputError(ValueError):
    pass


def plan_document(document: object) -> dict[str, Any]:


    raise NotImplementedError("capacity planning is not implemented")


def plan_file(path: str | Path) -> dict[str, Any]:


    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return plan_document(document)


def canonical_bytes(report: dict[str, Any]) -> bytes:


    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
