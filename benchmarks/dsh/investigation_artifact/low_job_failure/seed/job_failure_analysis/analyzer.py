from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


def analyze(evidence_dir: str | Path) -> dict[str, Any]:


    raise NotImplementedError("causal evidence analysis is not implemented")


def _canonical_bytes(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_report(evidence_dir: str | Path, output_path: str | Path) -> dict[str, Any]:


    report = analyze(evidence_dir)
    Path(output_path).write_bytes(_canonical_bytes(report))
    return report
