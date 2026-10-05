from __future__ import annotations

from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


def analyze_evidence(evidence_dir: str | Path) -> dict[str, Any]:


    raise NotImplementedError("cache-drift reconstruction remains to implement")


def incident_bytes(incident: dict[str, Any]) -> bytes:


    raise NotImplementedError("incident serialization remains to implement")


def reconciliation_bytes(rows: list[dict[str, Any]]) -> bytes:


    raise NotImplementedError("reconciliation serialization remains to implement")


def write_artifacts(
    evidence_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:


    raise NotImplementedError("artifact publication remains to implement")
