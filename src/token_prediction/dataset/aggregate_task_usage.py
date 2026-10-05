from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .schema import (
    DatasetRow,
    LabelStatus,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SupervisedDataset,
)


AGGREGATE_TASK_USAGE_SCHEMA_VERSION = 1
SWE_BENCH_METADATA_PROJECTION_SCHEMA_VERSION = 1
SWE_BENCH_METADATA_READER_POLICY_ID = "hyparquet_swebench_metadata_projection_v1"


@dataclass(frozen=True)
class SweBenchTaskMetadata:
    instance_id: str
    repo: str
    problem_statement: str
    difficulty: str | None = None


@dataclass(frozen=True)
class AggregateTaskUsageImport:
    dataset: SupervisedDataset
    source_id: str
    model_id: str
    condition_id: str
    aggregation_id: str
    task_count: int
    target_definition: str = "rounded_mean_input_plus_mean_output_tokens"


@dataclass(frozen=True)
class AggregateColumnMapping:
    task_id: str
    measured_input_tokens: str
    measured_output_tokens: str
    estimated_input_tokens: str | None = None
    estimated_output_tokens: str | None = None

    def __post_init__(self) -> None:
        required = (
            "task_id",
            "measured_input_tokens",
            "measured_output_tokens",
        )
        for field_name in required:
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} is required")
            object.__setattr__(self, field_name, value)
        optional = ("estimated_input_tokens", "estimated_output_tokens")
        for field_name in optional:
            value = getattr(self, field_name)
            if value is not None:
                normalized = str(value).strip()
                if not normalized:
                    raise ValueError(f"{field_name} must be non-empty when provided")
                object.__setattr__(self, field_name, normalized)
        if (self.estimated_input_tokens is None) != (self.estimated_output_tokens is None):
            raise ValueError("estimated input and output columns must be provided together")


def load_swebench_verified_metadata(
    parquet_path: str | Path,
) -> dict[str, SweBenchTaskMetadata]:


    try:
        import pyarrow.parquet as pq
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise RuntimeError(
            "SWE-bench parquet ingestion requires pyarrow; install token-prediction[data]"
        ) from exc
    source = Path(parquet_path).resolve()
    table = pq.read_table(
        source,
        columns=["instance_id", "repo", "problem_statement", "difficulty"],
    )
    result: dict[str, SweBenchTaskMetadata] = {}
    for row in table.to_pylist():
        item = SweBenchTaskMetadata(
            instance_id=str(row.get("instance_id") or "").strip(),
            repo=str(row.get("repo") or "").strip(),
            problem_statement=str(row.get("problem_statement") or ""),
            difficulty=(
                str(row.get("difficulty")).strip() if row.get("difficulty") is not None else None
            ),
        )
        if not item.instance_id or not item.repo or not item.problem_statement:
            raise ValueError("SWE-bench metadata contains an incomplete task")
        if item.instance_id in result:
            raise ValueError(f"duplicate SWE-bench instance_id {item.instance_id!r}")
        result[item.instance_id] = item
    if not result:
        raise ValueError("SWE-bench metadata is empty")
    return result


def load_swebench_verified_metadata_json(
    projection_path: str | Path,
) -> dict[str, SweBenchTaskMetadata]:


    source = Path(projection_path).resolve()
    payload = source.read_bytes()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata projection is not UTF-8 JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "reader",
        "source",
        "task_count",
        "tasks",
    }:
        raise ValueError("metadata projection keys do not match the schema")
    if document["schema_version"] != SWE_BENCH_METADATA_PROJECTION_SCHEMA_VERSION:
        raise ValueError("unsupported metadata projection schema")
    reader = document["reader"]
    if not isinstance(reader, dict) or reader != {
        "policy_id": SWE_BENCH_METADATA_READER_POLICY_ID,
        "package": "hyparquet",
        "version": "1.26.2",
        "columns": ["instance_id", "repo", "problem_statement", "difficulty"],
    }:
        raise ValueError("metadata projection reader identity is invalid")
    if not isinstance(document["source"], Mapping):
        raise ValueError("metadata projection source must be an object")
    raw_tasks = document["tasks"]
    if not isinstance(raw_tasks, list) or document["task_count"] != len(raw_tasks):
        raise ValueError("metadata projection task count does not close")
    result: dict[str, SweBenchTaskMetadata] = {}
    previous = ""
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict) or set(raw) != {
            "instance_id",
            "repo",
            "problem_statement",
            "difficulty",
        }:
            raise ValueError(f"metadata projection task {index} is invalid")
        item = SweBenchTaskMetadata(
            instance_id=str(raw["instance_id"] or "").strip(),
            repo=str(raw["repo"] or "").strip(),
            problem_statement=str(raw["problem_statement"] or ""),
            difficulty=(str(raw["difficulty"]).strip() if raw["difficulty"] is not None else None),
        )
        if not item.instance_id or not item.repo or not item.problem_statement:
            raise ValueError("metadata projection contains an incomplete task")
        if item.instance_id <= previous or item.instance_id in result:
            raise ValueError("metadata projection tasks are not uniquely sorted")
        previous = item.instance_id
        result[item.instance_id] = item
    if not result:
        raise ValueError("metadata projection is empty")
    return result


def build_aggregate_task_usage_dataset(
    aggregate_csv_path: str | Path,
    task_metadata: Mapping[str, SweBenchTaskMetadata],
    *,
    columns: AggregateColumnMapping,
    source_id: str,
    benchmark_id: str,
    model_id: str,
    agent_id: str,
    condition_id: str,
    aggregation_id: str,
) -> AggregateTaskUsageImport:


    declared = {
        "source_id": source_id,
        "benchmark_id": benchmark_id,
        "model_id": model_id,
        "agent_id": agent_id,
        "condition_id": condition_id,
        "aggregation_id": aggregation_id,
    }
    for name, value in declared.items():
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{name} is required")
        declared[name] = normalized

    source = Path(aggregate_csv_path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            columns.task_id,
            columns.measured_input_tokens,
            columns.measured_output_tokens,
        }
        if columns.estimated_input_tokens is not None:
            required.add(columns.estimated_input_tokens)
            required.add(str(columns.estimated_output_tokens))
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"aggregate CSV is missing columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("aggregate CSV is empty")

    metadata = dict(task_metadata)
    ids = [str(row.get(columns.task_id) or "").strip() for row in raw_rows]
    if any(not value for value in ids):
        raise ValueError("aggregate CSV contains an empty task identifier")
    if len(ids) != len(set(ids)):
        raise ValueError("aggregate CSV contains duplicate task identifiers")
    missing_metadata = sorted(set(ids) - set(metadata))
    if missing_metadata:
        raise ValueError(
            f"missing SWE-bench metadata for {len(missing_metadata)} task(s): "
            f"{missing_metadata[:3]}"
        )

    rows: list[DatasetRow] = []
    for raw in raw_rows:
        problem_id = str(raw[columns.task_id]).strip()
        task = metadata[problem_id]
        mean_input = _finite_non_negative_float(
            raw[columns.measured_input_tokens], columns.measured_input_tokens
        )
        mean_output = _finite_non_negative_float(
            raw[columns.measured_output_tokens], columns.measured_output_tokens
        )
        label = int(round(mean_input + mean_output))
        text = task.problem_statement
        task_id = f"{declared['benchmark_id']}:{problem_id}"
        run_id = f"{declared['condition_id']}:{declared['aggregation_id']}"
        trajectory_id = f"{declared['source_id']}:{run_id}:{problem_id}"
        event_id = f"{trajectory_id}:aggregate"
        features: dict[str, Any] = {
            "task_char_count": len(text),
            "task_word_count": len(re.findall(r"\S+", text)),
            "task_line_count": max(1, len(text.splitlines())),
            "task_code_fence_count": text.count("```") // 2,
            "repo_id": task.repo,
            "model_id": declared["model_id"],
            "agent_id": declared["agent_id"],
        }
        if columns.estimated_input_tokens is not None:
            estimated_input = _finite_non_negative_float(
                raw[columns.estimated_input_tokens], columns.estimated_input_tokens
            )
            estimated_output_name = str(columns.estimated_output_tokens)
            estimated_output = _finite_non_negative_float(
                raw[estimated_output_name], estimated_output_name
            )
            features["llm_self_estimated_total_tokens"] = estimated_input + estimated_output
        point = PredictionPoint(
            point_id=(
                f"{event_id}:{PredictionPosition.TASK_LAUNCH.value}:"
                f"{PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS.value}"
            ),
            source_event_id=event_id,
            task_id=task_id,
            trajectory_id=trajectory_id,
            run_id=run_id,
            prediction_context_id=f"{task_id}:initial",
            condition_id=declared["condition_id"],
            logical_call_id=None,
            attempt_id=None,
            cutoff_event_seq=0,
            position=PredictionPosition.TASK_LAUNCH,
            target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS,
            features=features,
            known_offset_tokens=0,
        )
        rows.append(DatasetRow(point=point, label=label, status=LabelStatus.OBSERVED))

    dataset_id = f"{declared['source_id']}:{declared['condition_id']}:{declared['aggregation_id']}"
    dataset = SupervisedDataset(
        dataset_id=dataset_id,
        rows=tuple(sorted(rows, key=lambda item: item.point.point_id)),
        schema_version=AGGREGATE_TASK_USAGE_SCHEMA_VERSION,
    )
    return AggregateTaskUsageImport(
        dataset=dataset,
        source_id=declared["source_id"],
        model_id=declared["model_id"],
        condition_id=declared["condition_id"],
        aggregation_id=declared["aggregation_id"],
        task_count=len(rows),
    )


def aggregate_task_usage_import_to_dict(value: AggregateTaskUsageImport) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "dataset_id": value.dataset.dataset_id,
            "source_id": value.source_id,
            "model_id": value.model_id,
            "condition_id": value.condition_id,
            "aggregation_id": value.aggregation_id,
            "task_count": value.task_count,
            "target_definition": value.target_definition,
        }
    )


def _finite_non_negative_float(value: Any, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} contains a non-numeric value") from exc
    if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"column {column!r} must be finite and non-negative")
    return parsed
