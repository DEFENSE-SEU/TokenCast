from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Analysis:
    report: dict[str, Any]
    exceptions: tuple[dict[str, str], ...]


def analyze(evidence_directory: str | Path) -> Analysis:


    del evidence_directory
    return Analysis(
        report={
            "orders": [],
            "schema_version": "order-reconciliation-report-v1",
        },
        exceptions=(),
    )


def write_artifacts(analysis: Analysis, output_directory: str | Path) -> tuple[Path, Path]:


    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    exceptions_path = output / "exceptions.csv"
    report_path.write_bytes(
        (
            json.dumps(
                analysis.report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    exceptions_path.write_bytes(b"exception_id,code,order_id,record_id,message,source,line\n")
    return report_path, exceptions_path
