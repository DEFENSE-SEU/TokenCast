from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from token_prediction.dataset import (
    TaskFamilyRegistry,
    assign_family_folds,
    assign_permanent_task_holdout,
)
from token_prediction.dataset.development import OUTER_FOLDS, STAGE_SPLIT_SEEDS


ALPHA = 0.10
BOOTSTRAP_ITERATIONS = 10_000
RIDGE_ALPHA = 1.0
SOURCE_MODEL_ID = "external-difficulty-text-channel-ridge-v4"
CANDIDATE_ID = "workload_mapping"
REFERENCE_ID = "empirical_quantile"
PROBLEM_TEXT_ID = "problem_text_workload_mapping"
REPO_TEXT_ID = "repo_text_workload_mapping"
DIFFICULTY_TEXT_ID = "difficulty_text_workload_mapping"
DIFFICULTY_SCALED_ID = "difficulty_workload_scaled_interval"
STRUCTURED_TEXT_ID = "task_structure_workload_mapping"
PROFILE_RESIDUAL_ID = "difficulty_profile_residual_mapping"
PROFILE_SCALED_INTERVAL_ID = "difficulty_profile_scaled_interval"
INPUT_WORKLOAD_ID = "input_workload_mapping"
OUTPUT_WORKLOAD_ID = "output_workload_mapping"
CHANNEL_FUSION_ID = "input_output_workload_fusion"
CHANNEL_FUSION_SCALED_ID = "input_output_scaled_interval"
SELECTED_CANDIDATE_ID = CHANNEL_FUSION_SCALED_ID
IN_FOLD_TEXT_ID = "in_fold_text_ridge"
IN_FOLD_DOCUMENT_VARIANT = "difficulty_problem"
SEED_ENSEMBLE_ID = "input_output_seed_ensemble"
CROSS_MODEL_ID = "cross_model_workload_mapping"
SELF_PREDICTION_ID = "self_prediction_gpt52_self_prediction_mapping"
SCOUTED_FUSION_ID = "workload_self_prediction_fusion"
PROFILE_RESIDUAL_MAPPING_ID = "profile_residual"
PROFILE_RESIDUAL_ALPHA = 30.0
MAPPING_IDS = (
    "horizon_factor",
    "direct_scale",
    "mae_scale",
    "log_linear",
    PROFILE_RESIDUAL_MAPPING_ID,
    "call_scale",
    "profile_factor",
)
SCALAR_MAPPING_IDS = ("direct_scale", "mae_scale", "log_linear")
SELF_PREDICTION_MODEL_PREFIXES = (
    "claude37sonnet",
    "gemini3",
    "gpt5",
    "gpt52",
    "kimik2",
    "qwen3coder",
    "sonnet45",
    "sonnet4base",
)
DOCUMENT_VARIANTS = (
    "problem",
    "repo_problem",
    "difficulty_problem",
    "repo_difficulty_problem",
    "structure_difficulty_problem",
)
DOCUMENT_CANDIDATES = (
    (PROBLEM_TEXT_ID, "problem"),
    (REPO_TEXT_ID, "repo_problem"),
    (DIFFICULTY_TEXT_ID, "difficulty_problem"),
    (STRUCTURED_TEXT_ID, "structure_difficulty_problem"),
)
TOKEN_FIELDS = (
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
)
INTERVAL_MODES = ("additive", "scaled")


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    source_task_id: str
    total_tokens: float
    call_count: float


@dataclass(frozen=True)
class Forecast:
    lower: float
    point: float
    upper: float
    raw_lower: float
    raw_upper: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Development-only Task-initialization workload transfer"
    )
    parser.add_argument(
        "--captures-root",
        type=Path,
        default=Path(
            "workspace/datasets/swebench-dual-model-v1/"
            "recollection-gpt-oauth-v18/runs/runs"
        ),
    )
    parser.add_argument(
        "--source-calls",
        type=Path,
        default=Path("workspace/experiments/reference_calls/calls.jsonl"),
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=Path(
            "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
        ),
    )
    parser.add_argument(
        "--self-predictions",
        type=Path,
        default=Path(
            "workspace/external/self_prediction_cohort/"
            "all_models_averaged_predictions_new.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/task_initialization_transfer"),
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _canonical_target_task_id(source_task_id: str) -> str:
    task_id = str(source_task_id).strip()
    if not task_id:
        raise ValueError("source task id is empty")
    return f"swebench-{task_id.replace('__', '-').replace('_', '-')}"


def _capture_roster(captures_root: Path) -> dict[str, list[Path]]:

    captures: dict[str, list[Path]] = {}
    for path in sorted(captures_root.rglob("capture.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        task = document.get("task")
        if not isinstance(task, Mapping):
            raise ValueError(f"capture has no task object: {path}")
        task_id = str(task.get("task_id") or "")
        if not task_id:
            raise ValueError(f"capture task identity is empty: {path}")
        captures.setdefault(task_id, []).append(path)
    if not captures:
        raise ValueError("capture roster is empty")
    return captures


def _development_tasks(task_ids: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    assignment = assign_permanent_task_holdout(
        task_ids,
        minimum_development_tasks=15,
    )
    return assignment.development_tasks, assignment.final_holdout_tasks


def _usage_total(usage: object, *, attempt_id: str) -> int:
    if not isinstance(usage, Mapping):
        raise ValueError(f"Attempt {attempt_id} has no usage object")
    total = 0
    for name in TOKEN_FIELDS:
        value = usage.get(name)
        if value is None:
            value = 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Attempt {attempt_id} has invalid {name}")
        total += value
    return total


def _load_development_outcomes(
    captures: Mapping[str, list[Path]],
    development_tasks: frozenset[str],
    target_to_source: Mapping[str, str],
) -> dict[str, TaskOutcome]:

    outcomes: dict[str, TaskOutcome] = {}
    for task_id in sorted(development_tasks):
        totals, counts = [], []
        for capture_path in captures[task_id]:
            completed = _capture_totals(capture_path, task_id=task_id)
            if completed is not None:
                totals.append(completed[0])
                counts.append(completed[1])
        if not totals:
            raise ValueError(f"development capture is not complete: {task_id}")
        outcomes[task_id] = TaskOutcome(
            task_id=task_id,
            source_task_id=target_to_source[task_id],
            total_tokens=float(np.mean(totals)),
            call_count=float(np.mean(counts)),
        )
    return outcomes


def _capture_totals(capture_path: Path, *, task_id: str) -> tuple[int, int] | None:

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    terminal = capture.get("terminal")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("outcome") != "completed"
        or terminal.get("attempt_ledger_complete") is not True
    ):
        return None
    observer_path = capture_path.with_name("observer.jsonl")
    attempt_ids: set[str] = set()
    logical_call_ids: set[str] = set()
    total_tokens = 0
    with observer_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "attempt/end":
                continue
            attempt_id = str(record.get("attempt_id") or "")
            logical_call_id = str(record.get("logical_call_id") or "")
            if not attempt_id or attempt_id in attempt_ids or not logical_call_id:
                raise ValueError(
                    f"observer Attempt identity is invalid at {observer_path}:{line_number}"
                )
            attempt_ids.add(attempt_id)
            logical_call_ids.add(logical_call_id)
            total_tokens += _usage_total(record.get("usage"), attempt_id=attempt_id)
    if not attempt_ids or not logical_call_ids or total_tokens <= 0:
        raise ValueError(f"development observer ledger is empty: {task_id}")
    return total_tokens, len(logical_call_ids)


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    table = pq.read_table(
        path,
        columns=["instance_id", "repo", "problem_statement", "difficulty"],
    )
    metadata: dict[str, dict[str, str]] = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        if task_id in metadata:
            raise ValueError(f"duplicate task metadata: {task_id}")
        metadata[task_id] = {
            "repo": str(row["repo"] or ""),
            "problem_statement": str(row["problem_statement"] or ""),
            "difficulty": str(row["difficulty"] or ""),
        }
    if not metadata:
        raise ValueError("task metadata is empty")
    return metadata


def _csv_float(row: Mapping[str, str], name: str, *, task_id: str) -> float:
    raw = row.get(name)
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError as exc:
        raise ValueError(f"self-prediction cohort row {task_id} has invalid {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"self-prediction cohort row {task_id} has invalid {name}")
    return value


def _load_self_prediction_profiles(
    path: Path,
    *,
    ground_truth_tasks: frozenset[str],
    self_prediction_tasks: frozenset[str],
) -> tuple[dict[str, tuple[float, ...]], dict[str, float]]:
    ground_truth: dict[str, tuple[float, ...]] = {}
    self_predictions: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "problem_id" not in reader.fieldnames:
            raise ValueError("self-prediction table has no problem_id column")
        for row in reader:
            task_id = str(row.get("problem_id") or "").strip()
            if not task_id:
                raise ValueError("self-prediction table contains an empty problem_id")
            if task_id in ground_truth_tasks:
                ground_truth[task_id] = tuple(
                    _csv_float(row, f"{prefix}_gt_input_token_avg", task_id=task_id)
                    + _csv_float(row, f"{prefix}_gt_output_token_avg", task_id=task_id)
                    for prefix in SELF_PREDICTION_MODEL_PREFIXES
                )
            if task_id in self_prediction_tasks:
                self_predictions[task_id] = _csv_float(
                    row,
                    "gpt52_predicted_avg_input",
                    task_id=task_id,
                ) + _csv_float(
                    row,
                    "gpt52_predicted_avg_output",
                    task_id=task_id,
                )
    if set(ground_truth) != set(ground_truth_tasks):
        missing = sorted(set(ground_truth_tasks) - set(ground_truth))
        raise ValueError(f"reference ground truth is missing tasks: {missing[:3]}")
    if set(self_predictions) != set(self_prediction_tasks):
        missing = sorted(set(self_prediction_tasks) - set(self_predictions))
        raise ValueError(f"published self-prediction is missing tasks: {missing[:3]}")
    return ground_truth, self_predictions


def _cross_model_profiles(
    ground_truth: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    if not ground_truth:
        raise ValueError("cross-model profile input is empty")
    matrix = np.asarray(list(ground_truth.values()), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(SELF_PREDICTION_MODEL_PREFIXES):
        raise ValueError("cross-model profile width is invalid")
    log_totals = np.log1p(matrix)
    model_centers = np.median(log_totals, axis=0)
    task_effects = np.median(log_totals - model_centers, axis=1)
    return {
        task_id: float(np.exp(effect))
        for task_id, effect in zip(ground_truth, task_effects)
    }


def _text_vectorizer() -> FeatureUnion:

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=12_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=12_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _fit_external_text_score(
    metadata: Mapping[str, Mapping[str, str]],
    targets: Mapping[str, float],
    *,
    query_by_target: Mapping[str, str],
    document_variant: str = "repo_difficulty_problem",
) -> tuple[dict[str, float], dict[str, int]]:
    fit_tasks = tuple(sorted(set(metadata) & set(targets)))
    if len(fit_tasks) < 100:
        raise ValueError("external text fitting requires at least 100 tasks")
    query_tasks = tuple(query_by_target[task_id] for task_id in sorted(query_by_target))
    if any(task_id not in metadata for task_id in query_tasks):
        raise ValueError("target task metadata is missing")
    vectorizer = _text_vectorizer()
    fit_matrix = vectorizer.fit_transform(
        [_task_document(metadata[task_id], document_variant) for task_id in fit_tasks]
    )
    query_matrix = vectorizer.transform(
        [_task_document(metadata[task_id], document_variant) for task_id in query_tasks]
    )
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    model.fit(fit_matrix, np.log1p([targets[task_id] for task_id in fit_tasks]))
    values = np.maximum(1e-6, np.expm1(model.predict(query_matrix)))
    return (
        {
            task_id: float(value)
            for task_id, value in zip(sorted(query_by_target), values)
        },
        {
            "fit_tasks": len(fit_tasks),
            "document_variant": document_variant,
            "word_features": len(vectorizer.transformer_list[0][1].vocabulary_),
            "char_features": len(vectorizer.transformer_list[1][1].vocabulary_),
        },
    )


def _load_source_profiles(
    path: Path,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    totals_by_task: dict[str, list[float]] = {}
    calls_by_task: dict[str, list[float]] = {}
    tokens_per_call_by_task: dict[str, list[float]] = {}
    input_by_task: dict[str, list[float]] = {}
    output_by_task: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id") or "")
            calls = record.get("calls")
            if not task_id or not isinstance(calls, list) or not calls:
                raise ValueError(f"invalid source record at line {line_number}")
            total = 0.0
            input_total = 0.0
            output_total = 0.0
            for call in calls:
                if not isinstance(call, Mapping):
                    raise ValueError(f"invalid source Call at line {line_number}")
                value = call.get("total_tokens")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"invalid source total_tokens at line {line_number}")
                total += float(value)
                for name in ("input_tokens", "output_tokens"):
                    channel_value = call.get(name)
                    if (
                        isinstance(channel_value, bool)
                        or not isinstance(channel_value, (int, float))
                        or channel_value < 0
                    ):
                        raise ValueError(
                            f"invalid source {name} at line {line_number}"
                        )
                input_total += float(call["input_tokens"])
                output_total += float(call["output_tokens"])
            totals_by_task.setdefault(task_id, []).append(total)
            call_count = float(len(calls))
            calls_by_task.setdefault(task_id, []).append(call_count)
            tokens_per_call_by_task.setdefault(task_id, []).append(total / call_count)
            input_by_task.setdefault(task_id, []).append(input_total)
            output_by_task.setdefault(task_id, []).append(output_total)
    if not totals_by_task:
        raise ValueError("source Call archive is empty")
    return (
        {
            task_id: float(np.median(values))
            for task_id, values in totals_by_task.items()
        },
        {
            task_id: float(np.median(values))
            for task_id, values in calls_by_task.items()
        },
        {
            task_id: float(np.median(values))
            for task_id, values in tokens_per_call_by_task.items()
        },
        {
            task_id: float(np.median(values))
            for task_id, values in input_by_task.items()
        },
        {
            task_id: float(np.median(values))
            for task_id, values in output_by_task.items()
        },
    )


def _task_document(
    metadata: Mapping[str, str],
    variant: str = "repo_difficulty_problem",
) -> str:
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"unknown task document variant: {variant}")
    repo = metadata["repo"].replace("/", "_").replace("-", "_")
    difficulty = metadata["difficulty"].replace(" ", "_")
    problem = metadata["problem_statement"]
    prefixes: list[str] = []
    if variant in {"repo_problem", "repo_difficulty_problem"}:
        prefixes.append(f"repo_{repo}")
    if variant in {
        "difficulty_problem",
        "repo_difficulty_problem",
        "structure_difficulty_problem",
    }:
        prefixes.append(f"difficulty_{difficulty}")
    if variant == "structure_difficulty_problem":
        lines = problem.splitlines()
        feature_counts = {
            "words": len(re.findall(r"\b\w+\b", problem)),
            "lines": len(lines),
            "code_blocks": problem.count("```") // 2,
            "list_items": sum(
                bool(re.match(r"\s*(?:[-*]|\d+[.)])\s", line))
                for line in lines
            ),
            "file_refs": len(
                re.findall(
                    r"(?<!\w)[\w./-]+\."
                    r"(?:py|js|ts|java|cpp|c|h|rs|go|rb|md|rst|toml|yml|yaml)"
                    r"(?::\d+)?",
                    problem,
                    re.IGNORECASE,
                )
            ),
            "test_error_refs": len(
                re.findall(
                    r"\b(?:test\w*|pytest|assert\w*|error|exception|traceback|"
                    r"fail(?:ed|ure|ing)?)\b",
                    problem,
                    re.IGNORECASE,
                )
            ),
            "expected_behavior": len(
                re.findall(
                    r"\b(?:expected|actual|should|instead)\b",
                    problem,
                    re.IGNORECASE,
                )
            ),
        }
        prefixes.extend(
            f"{name}_{int(math.log2(count + 1))}"
            for name, count in feature_counts.items()
        )
    prefixes.append(problem)
    return " ".join(prefixes)


def _fit_source_scores(
    metadata: Mapping[str, Mapping[str, str]],
    source_totals: Mapping[str, float],
    source_calls: Mapping[str, float],
    source_tokens_per_call: Mapping[str, float],
    *,
    excluded_source_tasks: frozenset[str],
    query_by_target: Mapping[str, str],
    document_variant: str = "repo_difficulty_problem",
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, int],
]:
    fit_tasks = tuple(
        sorted(
            (
                set(metadata)
                & set(source_totals)
                & set(source_calls)
                & set(source_tokens_per_call)
            )
            - excluded_source_tasks
        )
    )
    if len(fit_tasks) < 100:
        raise ValueError("external workload fitting requires at least 100 tasks")
    query_tasks = tuple(query_by_target[task_id] for task_id in sorted(query_by_target))
    if any(task_id not in metadata for task_id in query_tasks):
        raise ValueError("target task metadata is missing")
    vectorizer = _text_vectorizer()
    fit_matrix = vectorizer.fit_transform(
        [_task_document(metadata[task_id], document_variant) for task_id in fit_tasks]
    )
    query_matrix = vectorizer.transform(
        [_task_document(metadata[task_id], document_variant) for task_id in query_tasks]
    )
    target_ids = tuple(sorted(query_by_target))
    scores: list[dict[str, float]] = []
    for targets in (source_totals, source_calls, source_tokens_per_call):
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
        model.fit(
            fit_matrix,
            np.log1p([targets[task_id] for task_id in fit_tasks]),
        )
        values = np.maximum(1.0, np.expm1(model.predict(query_matrix)))
        scores.append(
            {task_id: float(value) for task_id, value in zip(target_ids, values)}
        )
    return (
        scores[0],
        scores[1],
        scores[2],
        {
            "fit_tasks": len(fit_tasks),
            "excluded_tasks": len(excluded_source_tasks),
            "document_variant": document_variant,
            "word_features": len(vectorizer.transformer_list[0][1].vocabulary_),
            "char_features": len(vectorizer.transformer_list[1][1].vocabulary_),
        },
    )


def _weighted_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile values are empty")
    ordered = sorted(float(value) for value in values)
    threshold = quantile * len(ordered)
    for index, value in enumerate(ordered, start=1):
        if index >= threshold:
            return value
    return ordered[-1]


def _interval(values: Sequence[float]) -> tuple[float, float, float]:
    lower = _weighted_quantile(values, ALPHA / 2)
    point = _weighted_quantile(values, 0.5)
    upper = _weighted_quantile(values, 1 - ALPHA / 2)
    return min(lower, point), point, max(upper, point)


def _conformal_expansion(
    raw_forecasts: Mapping[str, tuple[float, float, float]],
    outcomes: Mapping[str, TaskOutcome],
) -> float:
    scores = sorted(
        max(lower - outcomes[task_id].total_tokens, outcomes[task_id].total_tokens - upper, 0.0)
        for task_id, (lower, _point, upper) in raw_forecasts.items()
    )
    return _split_conformal_quantile(scores)


def _conformal_factor(
    raw_forecasts: Mapping[str, tuple[float, float, float]],
    outcomes: Mapping[str, TaskOutcome],
) -> float:
    scores = sorted(
        max(
            lower / outcomes[task_id].total_tokens,
            outcomes[task_id].total_tokens / upper,
            1.0,
        )
        for task_id, (lower, _point, upper) in raw_forecasts.items()
    )
    return _split_conformal_quantile(scores)


def _repair_forecast(
    raw: tuple[float, float, float], expansion: float
) -> Forecast:
    raw_lower, point, raw_upper = raw
    point = max(0.0, point)
    raw_lower = min(max(0.0, raw_lower), point)
    raw_upper = max(max(0.0, raw_upper), point)
    return Forecast(
        lower=max(0.0, raw_lower - expansion),
        point=point,
        upper=max(point, raw_upper + expansion),
        raw_lower=raw_lower,
        raw_upper=raw_upper,
    )


def _repair_scaled_forecast(
    raw: tuple[float, float, float], factor: float
) -> Forecast:
    if factor < 1.0:
        raise ValueError("scaled conformal factor must be at least one")
    raw_lower, point, raw_upper = raw
    point = max(0.0, point)
    raw_lower = min(max(0.0, raw_lower), point)
    raw_upper = max(max(0.0, raw_upper), point)
    return Forecast(
        lower=raw_lower / factor,
        point=point,
        upper=raw_upper * factor,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
    )


def _split_conformal_quantile(scores: Iterable[float]) -> float:

    ordered = sorted(float(score) for score in scores)
    if not ordered:
        raise ValueError("calibration scores are empty")
    rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * (1 - ALPHA))))
    return ordered[rank - 1]


def _in_fold_text_forecasts(
    fit_documents: Sequence[str],
    fit_totals: Sequence[float],
    calibration_documents: Sequence[str],
    calibration_totals: Sequence[float],
    query_documents: Sequence[str],
    *,
    alpha: float = RIDGE_ALPHA,
    dense: Sequence[np.ndarray] | None = None,
) -> list[Forecast]:

    vectorizer = _text_vectorizer()
    matrices = [
        vectorizer.fit_transform(fit_documents),
        vectorizer.transform(calibration_documents),
        vectorizer.transform(query_documents),
    ]
    if dense is not None:
        matrices = [sparse.hstack([m, sparse.csr_matrix(d)], format="csr") for m, d in zip(matrices, dense)]
    fit_matrix, calibration_matrix, query_matrix = matrices
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(fit_matrix, np.log1p(np.asarray(fit_totals, dtype=np.float64)))
    calibration_logs = model.predict(calibration_matrix)
    radius = _split_conformal_quantile(
        np.abs(np.log1p(np.asarray(calibration_totals, dtype=np.float64)) - calibration_logs)
    )
    forecasts = []
    for log_point in model.predict(query_matrix):
        point = max(0.0, float(np.expm1(log_point)))
        forecasts.append(
            Forecast(
                lower=min(point, max(0.0, float(np.expm1(log_point - radius)))),
                point=point,
                upper=max(point, float(np.expm1(log_point + radius))),
                raw_lower=point,
                raw_upper=point,
            )
        )
    return forecasts


def _history_median_forecast(
    fit_totals: Sequence[float], calibration_totals: Sequence[float]
) -> Forecast:

    raw = _interval(fit_totals)
    expansion = _split_conformal_quantile(
        max(raw[0] - total, total - raw[2], 0.0) for total in calibration_totals
    )
    return _repair_forecast(raw, expansion)


def _fold_roles(
    fold_by_task: Mapping[str, int], tasks: Iterable[str], test_fold: int, folds: int
) -> dict[str, list[str]]:

    named = {
        test_fold: "test",
        (test_fold + 1) % folds: "calibration",
        (test_fold + 2) % folds: "validation",
    }
    roles: dict[str, list[str]] = {"train": [], "validation": [], "calibration": [], "test": []}
    for task_id in sorted(tasks):
        roles[named.get(fold_by_task[task_id], "train")].append(task_id)
    if not all(roles.values()):
        raise ValueError(f"fold {test_fold} has an empty partition")
    return roles


def _mae_scale(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("weighted median inputs must have equal positive length")
    if any(weight < 0 for weight in weights):
        raise ValueError("weighted median weights cannot be negative")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total_weight = sum(weight for _value, weight in ordered)
    if total_weight <= 0:
        raise ValueError("weighted median requires positive total weight")
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total_weight / 2:
            return float(value)
    return float(ordered[-1][0])


def _mapping_point(
    source_score: float,
    parameters: tuple[str, float, float, float, float, float],
    source_call_score: float | None = None,
    source_tokens_per_call_score: float | None = None,
) -> float:
    (
        mapping_id,
        first,
        second,
        third,
        _residual_lower,
        _residual_upper,
    ) = parameters
    if mapping_id == "log_linear":
        return float(max(1.0, np.expm1(first + second * np.log1p(source_score))))
    if mapping_id == PROFILE_RESIDUAL_MAPPING_ID:
        call_score = source_score if source_call_score is None else source_call_score
        tokens_per_call_score = (
            source_score
            if source_tokens_per_call_score is None
            else source_tokens_per_call_score
        )
        source_log = np.log1p(source_score)
        adjustment = (
            first
            + second * (np.log1p(call_score) - source_log)
            + third * (np.log1p(tokens_per_call_score) - source_log)
        )
        return float(max(1.0, np.expm1(source_log + adjustment)))
    if mapping_id == "call_scale":
        call_score = source_score if source_call_score is None else source_call_score
        return float(call_score * first)
    if mapping_id == "profile_factor":
        call_score = source_score if source_call_score is None else source_call_score
        tokens_per_call_score = (
            source_score
            if source_tokens_per_call_score is None
            else source_tokens_per_call_score
        )
        return float(call_score * tokens_per_call_score * first * second)
    return float(source_score * first)


def _mapping_parameters(
    mapping_id: str,
    train_tasks: Sequence[str],
    outcomes: Mapping[str, TaskOutcome],
    source_scores: Mapping[str, float],
    source_call_scores: Mapping[str, float] | None = None,
    source_tokens_per_call_scores: Mapping[str, float] | None = None,
) -> tuple[str, float, float, float, float, float]:
    if mapping_id not in MAPPING_IDS:
        raise ValueError(f"unknown workload mapping: {mapping_id}")
    ratios = [
        outcomes[task_id].total_tokens / source_scores[task_id]
        for task_id in train_tasks
    ]
    call_scores = source_call_scores or source_scores
    tokens_per_call_scores = source_tokens_per_call_scores or source_scores
    if mapping_id == "horizon_factor":
        workload_to_calls = _weighted_quantile(
            [
                outcomes[task_id].call_count / source_scores[task_id]
                for task_id in train_tasks
            ],
            0.5,
        )
        tokens_per_call = _weighted_quantile(
            [
                outcomes[task_id].total_tokens / outcomes[task_id].call_count
                for task_id in train_tasks
            ],
            0.5,
        )
        first, second, third = workload_to_calls * tokens_per_call, 0.0, 0.0
    elif mapping_id == "direct_scale":
        first, second, third = _weighted_quantile(ratios, 0.5), 0.0, 0.0
    elif mapping_id == "mae_scale":
        first, second, third = _mae_scale(
            ratios,
            [source_scores[task_id] for task_id in train_tasks],
        ), 0.0, 0.0
    elif mapping_id == PROFILE_RESIDUAL_MAPPING_ID:
        source_logs = np.log1p(
            [source_scores[task_id] for task_id in train_tasks]
        )
        design = np.column_stack(
            (
                np.log1p([call_scores[task_id] for task_id in train_tasks])
                - source_logs,
                np.log1p(
                    [tokens_per_call_scores[task_id] for task_id in train_tasks]
                )
                - source_logs,
            )
        )
        model = Ridge(alpha=PROFILE_RESIDUAL_ALPHA)
        model.fit(
            design,
            np.log1p(
                [outcomes[task_id].total_tokens for task_id in train_tasks]
            )
            - source_logs,
        )
        first = float(model.intercept_)
        second, third = (float(value) for value in model.coef_)
    elif mapping_id == "call_scale":
        first, second, third = _weighted_quantile(
            [
                outcomes[task_id].total_tokens / call_scores[task_id]
                for task_id in train_tasks
            ],
            0.5,
        ), 0.0, 0.0
    elif mapping_id == "profile_factor":
        first = _weighted_quantile(
            [
                outcomes[task_id].call_count / call_scores[task_id]
                for task_id in train_tasks
            ],
            0.5,
        )
        second = _weighted_quantile(
            [
                (outcomes[task_id].total_tokens / outcomes[task_id].call_count)
                / tokens_per_call_scores[task_id]
                for task_id in train_tasks
            ],
            0.5,
        )
        third = 0.0
    else:
        design = np.column_stack(
            (
                np.ones(len(train_tasks)),
                np.log1p([source_scores[task_id] for task_id in train_tasks]),
            )
        )
        targets = np.log1p(
            [outcomes[task_id].total_tokens for task_id in train_tasks]
        )
        first, second = (
            float(value) for value in np.linalg.lstsq(design, targets, rcond=None)[0]
        )
        third = 0.0
    provisional = (mapping_id, first, second, third, 0.0, 0.0)
    train_predictions = {
        task_id: _mapping_point(
            source_scores[task_id],
            provisional,
            call_scores[task_id],
            tokens_per_call_scores[task_id],
        )
        for task_id in train_tasks
    }
    residuals = [
        outcomes[task_id].total_tokens - train_predictions[task_id]
        for task_id in train_tasks
    ]
    residual_lower = _weighted_quantile(residuals, ALPHA / 2)
    residual_upper = _weighted_quantile(residuals, 1 - ALPHA / 2)
    return mapping_id, first, second, third, residual_lower, residual_upper


def _mapping_raw(
    task_id: str,
    source_scores: Mapping[str, float],
    parameters: tuple[str, float, float, float, float, float],
    source_call_scores: Mapping[str, float] | None = None,
    source_tokens_per_call_scores: Mapping[str, float] | None = None,
) -> tuple[float, float, float]:
    (
        _mapping_id,
        _first,
        _second,
        _third,
        residual_lower,
        residual_upper,
    ) = parameters
    call_scores = source_call_scores or source_scores
    tokens_per_call_scores = source_tokens_per_call_scores or source_scores
    point = _mapping_point(
        source_scores[task_id],
        parameters,
        call_scores[task_id],
        tokens_per_call_scores[task_id],
    )
    return point + residual_lower, point, point + residual_upper


def _evaluate_seed(
    seed: int,
    registry: TaskFamilyRegistry,
    development_tasks: frozenset[str],
    outcomes: Mapping[str, TaskOutcome],
    source_scores: Mapping[str, float],
    source_call_scores: Mapping[str, float] | None = None,
    source_tokens_per_call_scores: Mapping[str, float] | None = None,
    *,
    candidate_id: str = CANDIDATE_ID,
    mapping_ids: Sequence[str] = MAPPING_IDS,
    interval_mode: str = "additive",
    fold_by_task: Mapping[str, int] | None = None,
    folds: int = OUTER_FOLDS,
) -> tuple[dict[str, dict[str, Forecast]], list[dict[str, object]]]:
    if not candidate_id or candidate_id == REFERENCE_ID:
        raise ValueError("candidate id must be distinct from the reference")
    if not mapping_ids or any(mapping_id not in MAPPING_IDS for mapping_id in mapping_ids):
        raise ValueError("candidate mapping ids are empty or unknown")
    if interval_mode not in INTERVAL_MODES:
        raise ValueError(f"unknown interval mode: {interval_mode}")
    if folds < 4:
        raise ValueError("need at least four folds for train, validation, calibration and test")
    if fold_by_task is None:
        fold_by_task = assign_family_folds(
            registry,
            folds=folds,
            seed=seed,
            family_ids=development_tasks,
        ).task_to_fold
    call_scores = source_call_scores or source_scores
    tokens_per_call_scores = source_tokens_per_call_scores or source_scores
    predictions = {REFERENCE_ID: {}, candidate_id: {}}
    fold_reports: list[dict[str, object]] = []
    for test_fold in range(folds):
        calibration_fold = (test_fold + 1) % folds
        validation_fold = (test_fold + 2) % folds
        train_tasks = tuple(
            sorted(
                task_id
                for task_id in development_tasks
                if fold_by_task[task_id]
                not in {test_fold, calibration_fold, validation_fold}
            )
        )
        calibration_tasks = tuple(
            sorted(
                task_id
                for task_id in development_tasks
                if fold_by_task[task_id] == calibration_fold
            )
        )
        validation_tasks = tuple(
            sorted(
                task_id
                for task_id in development_tasks
                if fold_by_task[task_id] == validation_fold
            )
        )
        test_tasks = tuple(
            sorted(
                task_id
                for task_id in development_tasks
                if fold_by_task[task_id] == test_fold
            )
        )
        if (
            not train_tasks
            or not calibration_tasks
            or not validation_tasks
            or not test_tasks
        ):
            raise ValueError(f"seed {seed} fold {test_fold} has an empty partition")

        refit_tasks = tuple(sorted((*train_tasks, *validation_tasks)))
        reference_raw_value = _interval(
            [outcomes[task_id].total_tokens for task_id in refit_tasks]
        )
        reference_calibration_raw = {
            task_id: reference_raw_value for task_id in calibration_tasks
        }
        reference_expansion = _conformal_expansion(
            reference_calibration_raw, outcomes
        )

        mapping_fits = {
            mapping_id: _mapping_parameters(
                mapping_id,
                train_tasks,
                outcomes,
                source_scores,
                source_call_scores,
                source_tokens_per_call_scores,
            )
            for mapping_id in mapping_ids
        }
        validation_mae: dict[str, float] = {}
        for mapping_id in mapping_ids:
            parameters = mapping_fits[mapping_id]
            points = [
                _mapping_point(
                    source_scores[task_id],
                    parameters,
                    call_scores[task_id],
                    tokens_per_call_scores[task_id],
                )
                for task_id in validation_tasks
            ]
            validation_mae[mapping_id] = float(
                np.mean(
                    [
                        abs(point - outcomes[task_id].total_tokens)
                        for point, task_id in zip(points, validation_tasks)
                    ]
                )
            )
        selected_mapping_id = min(
            mapping_ids,
            key=lambda mapping_id: (validation_mae[mapping_id], mapping_id),
        )
        parameters = _mapping_parameters(
            selected_mapping_id,
            refit_tasks,
            outcomes,
            source_scores,
            source_call_scores,
            source_tokens_per_call_scores,
        )

        if interval_mode == "scaled":
            refit_ratios = [
                outcomes[task_id].total_tokens
                / _mapping_point(
                    source_scores[task_id],
                    parameters,
                    call_scores[task_id],
                    tokens_per_call_scores[task_id],
                )
                for task_id in refit_tasks
            ]
            ratio_lower = min(1.0, _weighted_quantile(refit_ratios, ALPHA / 2))
            ratio_upper = max(
                1.0,
                _weighted_quantile(refit_ratios, 1 - ALPHA / 2),
            )

            def selected_raw(task_id: str) -> tuple[float, float, float]:
                point = _mapping_point(
                    source_scores[task_id],
                    parameters,
                    call_scores[task_id],
                    tokens_per_call_scores[task_id],
                )
                return point * ratio_lower, point, point * ratio_upper

        else:

            def selected_raw(task_id: str) -> tuple[float, float, float]:
                return _mapping_raw(
                    task_id,
                    source_scores,
                    parameters,
                    source_call_scores,
                    source_tokens_per_call_scores,
                )

        candidate_calibration_raw = {
            task_id: selected_raw(task_id)
            for task_id in calibration_tasks
        }
        if interval_mode == "scaled":
            candidate_adjustment = _conformal_factor(
                candidate_calibration_raw, outcomes
            )
        else:
            candidate_adjustment = _conformal_expansion(
                candidate_calibration_raw, outcomes
            )
        for task_id in test_tasks:
            predictions[REFERENCE_ID][task_id] = _repair_forecast(
                reference_raw_value, reference_expansion
            )
            if interval_mode == "scaled":
                predictions[candidate_id][task_id] = _repair_scaled_forecast(
                    selected_raw(task_id),
                    candidate_adjustment,
                )
            else:
                predictions[candidate_id][task_id] = _repair_forecast(
                    selected_raw(task_id),
                    candidate_adjustment,
                )
        fold_reports.append(
            {
                "test_fold": test_fold,
                "train_task_count": len(train_tasks),
                "validation_task_count": len(validation_tasks),
                "refit_task_count": len(refit_tasks),
                "calibration_task_count": len(calibration_tasks),
                "test_task_count": len(test_tasks),
                "selected_mapping_id": selected_mapping_id,
                "validation_mae": validation_mae,
                "mapping_parameters": {
                    "first": parameters[1],
                    "second": parameters[2],
                    "third": parameters[3],
                },
                "candidate_interval_mode": interval_mode,
                "candidate_conformal_adjustment": candidate_adjustment,
                "reference_conformal_expansion": reference_expansion,
            }
        )
    expected = set(development_tasks)
    if any(set(values) != expected for values in predictions.values()):
        raise ValueError("cross-validation predictions do not cover development tasks")
    return predictions, fold_reports


def _evaluate_in_fold_seed(
    seed: int,
    registry: TaskFamilyRegistry,
    development_tasks: frozenset[str],
    outcomes: Mapping[str, TaskOutcome],
    documents: Mapping[str, str],
    *,
    fold_by_task: Mapping[str, int] | None = None,
    folds: int = OUTER_FOLDS,
) -> dict[str, dict[str, Forecast]]:

    if folds < 4:
        raise ValueError("need at least four folds for train, validation, calibration and test")
    if fold_by_task is None:
        fold_by_task = assign_family_folds(
            registry, folds=folds, seed=seed, family_ids=development_tasks
        ).task_to_fold
    predictions: dict[str, dict[str, Forecast]] = {REFERENCE_ID: {}, IN_FOLD_TEXT_ID: {}}
    for test_fold in range(folds):
        roles = _fold_roles(fold_by_task, development_tasks, test_fold, folds)
        fit_tasks = roles["train"] + roles["validation"]
        fit_totals = [outcomes[task_id].total_tokens for task_id in fit_tasks]
        calibration_totals = [
            outcomes[task_id].total_tokens for task_id in roles["calibration"]
        ]
        text = _in_fold_text_forecasts(
            [documents[task_id] for task_id in fit_tasks],
            fit_totals,
            [documents[task_id] for task_id in roles["calibration"]],
            calibration_totals,
            [documents[task_id] for task_id in roles["test"]],
        )
        reference = _history_median_forecast(fit_totals, calibration_totals)
        for task_id, forecast in zip(roles["test"], text):
            predictions[IN_FOLD_TEXT_ID][task_id] = forecast
            predictions[REFERENCE_ID][task_id] = reference
    return predictions


def _metrics(
    forecasts: Mapping[str, Forecast], outcomes: Mapping[str, TaskOutcome]
) -> dict[str, float | int]:
    task_ids = sorted(forecasts)
    truths = np.asarray([outcomes[task_id].total_tokens for task_id in task_ids])
    points = np.asarray([forecasts[task_id].point for task_id in task_ids])
    lowers = np.asarray([forecasts[task_id].lower for task_id in task_ids])
    uppers = np.asarray([forecasts[task_id].upper for task_id in task_ids])
    absolute = np.abs(points - truths)
    widths = uppers - lowers
    interval_scores = widths.copy()
    interval_scores += (2 / ALPHA) * np.maximum(lowers - truths, 0.0)
    interval_scores += (2 / ALPHA) * np.maximum(truths - uppers, 0.0)
    return {
        "n_tasks": len(task_ids),
        "mae": float(np.mean(absolute)),
        "wape": float(np.sum(absolute) / np.sum(np.abs(truths))),
        "mean_bias": float(np.mean(points - truths)),
        "coverage": float(np.mean((lowers <= truths) & (truths <= uppers))),
        "interval_score": float(np.mean(interval_scores)),
        "normalized_interval_width": float(
            np.mean(widths / np.maximum(np.abs(truths), 1.0))
        ),
    }


def _paired_bootstrap_from_errors(
    candidate_errors: Mapping[str, float],
    reference_errors: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    task_ids = sorted(candidate_errors)
    if set(task_ids) != set(reference_errors):
        raise ValueError("paired bootstrap task sets differ")
    deltas = np.asarray(
        [candidate_errors[task_id] - reference_errors[task_id] for task_id in task_ids]
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(task_ids), size=(iterations, len(task_ids)))
    samples = deltas[indices].mean(axis=1)
    candidate_mae = float(np.mean(list(candidate_errors.values())))
    reference_mae = float(np.mean(list(reference_errors.values())))
    delta = candidate_mae - reference_mae
    return {
        "n_tasks": len(task_ids),
        "iterations": iterations,
        "candidate_mae": candidate_mae,
        "reference_mae": reference_mae,
        "mae_delta": delta,
        "relative_mae_change_pct": 100.0 * delta / reference_mae,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "win_probability": float(np.mean(samples < 0.0)),
    }


def _paired_bootstrap(
    predictions: Mapping[str, Mapping[str, Forecast]],
    outcomes: Mapping[str, TaskOutcome],
    *,
    iterations: int,
    seed: int,
    candidate_id: str = CANDIDATE_ID,
    reference_id: str = REFERENCE_ID,
) -> dict[str, float | int]:
    candidate_errors = {
        task_id: abs(forecast.point - outcomes[task_id].total_tokens)
        for task_id, forecast in predictions[candidate_id].items()
    }
    reference_errors = {
        task_id: abs(forecast.point - outcomes[task_id].total_tokens)
        for task_id, forecast in predictions[reference_id].items()
    }
    return _paired_bootstrap_from_errors(
        candidate_errors,
        reference_errors,
        iterations=iterations,
        seed=seed,
    )


def _mean_metrics(
    reports: Mapping[int, Mapping[str, Mapping[str, float | int]]],
    candidate_id: str,
) -> dict[str, float]:
    names = tuple(
        name
        for name in next(iter(reports.values()))[candidate_id]
        if name != "n_tasks"
    )
    return {
        name: float(
            np.mean([float(report[candidate_id][name]) for report in reports.values()])
        )
        for name in names
    }


def in_fold_inputs(
    captures_root: Path, metadata_path: Path
) -> tuple[
    TaskFamilyRegistry,
    frozenset[str],
    dict[str, TaskOutcome],
    dict[str, str],
    dict[str, list[Path]],
]:

    captures = _capture_roster(captures_root)
    registry = TaskFamilyRegistry.from_mapping({task_id: task_id for task_id in captures})
    development_tasks, _ = _development_tasks(captures)
    metadata = _load_metadata(metadata_path)
    source_by_target = {
        _canonical_target_task_id(source_task_id): source_task_id
        for source_task_id in metadata
    }
    missing = sorted(development_tasks - set(source_by_target))
    if missing:
        raise ValueError(f"development tasks missing from task metadata: {missing[:3]}")
    outcomes = _load_development_outcomes(captures, development_tasks, source_by_target)
    documents = {
        task_id: _task_document(
            metadata[source_by_target[task_id]], IN_FOLD_DOCUMENT_VARIANT
        )
        for task_id in development_tasks
    }
    return registry, development_tasks, outcomes, documents, captures


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    captures = _capture_roster(args.captures_root)
    registry = TaskFamilyRegistry.from_mapping(
        {task_id: task_id for task_id in captures}
    )
    development_tasks, final_holdout_tasks = _development_tasks(captures)

    metadata = _load_metadata(args.task_metadata)
    (
        source_totals,
        source_calls,
        source_tokens_per_call,
        source_input_totals,
        source_output_totals,
    ) = _load_source_profiles(args.source_calls)
    target_by_source: dict[str, str] = {}
    for source_task_id in metadata:
        target_task_id = _canonical_target_task_id(source_task_id)
        if target_task_id in target_by_source:
            raise ValueError(f"canonical task id collision: {target_task_id}")
        target_by_source[target_task_id] = source_task_id
    if not set(captures) <= set(target_by_source):
        raise ValueError("capture roster contains tasks missing from source metadata")
    source_by_target = {
        task_id: target_by_source[task_id] for task_id in captures
    }
    excluded_source_tasks = frozenset(source_by_target.values())
    outcomes = _load_development_outcomes(
        captures,
        development_tasks,
        source_by_target,
    )
    development_query = {
        task_id: source_by_target[task_id] for task_id in development_tasks
    }
    (
        source_scores,
        source_call_scores,
        source_tokens_per_call_scores,
        source_fit,
    ) = _fit_source_scores(
        metadata,
        source_totals,
        source_calls,
        source_tokens_per_call,
        excluded_source_tasks=excluded_source_tasks,
        query_by_target=development_query,
    )
    document_scores = {
        CANDIDATE_ID: (
            source_scores,
            source_call_scores,
            source_tokens_per_call_scores,
        )
    }
    document_fits = {CANDIDATE_ID: source_fit}
    for candidate_id, document_variant in DOCUMENT_CANDIDATES:
        total_score, call_score, tokens_per_call_score, fit = _fit_source_scores(
            metadata,
            source_totals,
            source_calls,
            source_tokens_per_call,
            excluded_source_tasks=excluded_source_tasks,
            query_by_target=development_query,
            document_variant=document_variant,
        )
        document_scores[candidate_id] = (
            total_score,
            call_score,
            tokens_per_call_score,
        )
        document_fits[candidate_id] = fit

    external_fit_tasks = frozenset(
        (
            set(metadata)
            & set(source_totals)
            & set(source_calls)
            & set(source_tokens_per_call)
            & set(source_input_totals)
            & set(source_output_totals)
        )
        - excluded_source_tasks
    )
    input_workload_scores, input_workload_fit = _fit_external_text_score(
        metadata,
        {
            task_id: source_input_totals[task_id]
            for task_id in external_fit_tasks
        },
        query_by_target=development_query,
        document_variant="difficulty_problem",
    )
    output_workload_scores, output_workload_fit = _fit_external_text_score(
        metadata,
        {
            task_id: source_output_totals[task_id]
            for task_id in external_fit_tasks
        },
        query_by_target=development_query,
        document_variant="difficulty_problem",
    )
    channel_fusion_scores = {
        task_id: math.sqrt(
            input_workload_scores[task_id] * output_workload_scores[task_id]
        )
        for task_id in development_tasks
    }
    development_source_tasks = frozenset(development_query.values())
    reference_ground_truth, self_prediction_by_source = _load_self_prediction_profiles(
        args.self_predictions,
        ground_truth_tasks=external_fit_tasks,
        self_prediction_tasks=development_source_tasks,
    )
    cross_model_targets = _cross_model_profiles(reference_ground_truth)
    cross_model_scores, cross_model_fit = _fit_external_text_score(
        metadata,
        cross_model_targets,
        query_by_target=development_query,
    )
    self_prediction_scores = {
        task_id: self_prediction_by_source[source_task_id]
        for task_id, source_task_id in development_query.items()
    }
    scouted_fusion_scores = {
        task_id: float(
            np.exp(
                (
                    math.log(source_scores[task_id])
                    + math.log(cross_model_scores[task_id])
                    + math.log(self_prediction_scores[task_id])
                )
                / 3.0
            )
        )
        for task_id in development_tasks
    }

    candidate_specs = {
        **{
            candidate_id: (*scores, MAPPING_IDS, "additive")
            for candidate_id, scores in document_scores.items()
        },
        PROFILE_RESIDUAL_ID: (
            *document_scores[DIFFICULTY_TEXT_ID],
            (PROFILE_RESIDUAL_MAPPING_ID,),
            "additive",
        ),
        DIFFICULTY_SCALED_ID: (
            *document_scores[DIFFICULTY_TEXT_ID],
            MAPPING_IDS,
            "scaled",
        ),
        PROFILE_SCALED_INTERVAL_ID: (
            *document_scores[DIFFICULTY_TEXT_ID],
            (PROFILE_RESIDUAL_MAPPING_ID,),
            "scaled",
        ),
        INPUT_WORKLOAD_ID: (
            input_workload_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
        OUTPUT_WORKLOAD_ID: (
            output_workload_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
        CHANNEL_FUSION_ID: (
            channel_fusion_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
        CHANNEL_FUSION_SCALED_ID: (
            channel_fusion_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "scaled",
        ),
        CROSS_MODEL_ID: (
            cross_model_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
        SELF_PREDICTION_ID: (
            self_prediction_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
        SCOUTED_FUSION_ID: (
            scouted_fusion_scores,
            None,
            None,
            SCALAR_MAPPING_IDS,
            "additive",
        ),
    }
    comparison_pairs = tuple(
        (candidate_id, REFERENCE_ID) for candidate_id in candidate_specs
    ) + (
        *((candidate_id, CANDIDATE_ID) for candidate_id, _ in DOCUMENT_CANDIDATES),
        (INPUT_WORKLOAD_ID, SELECTED_CANDIDATE_ID),
        (OUTPUT_WORKLOAD_ID, SELECTED_CANDIDATE_ID),
        (CHANNEL_FUSION_ID, SELECTED_CANDIDATE_ID),
        (CHANNEL_FUSION_SCALED_ID, CHANNEL_FUSION_ID),
        (CHANNEL_FUSION_SCALED_ID, PROFILE_SCALED_INTERVAL_ID),
        (PROFILE_RESIDUAL_ID, SELECTED_CANDIDATE_ID),
        (DIFFICULTY_SCALED_ID, SELECTED_CANDIDATE_ID),
        (DIFFICULTY_SCALED_ID, PROFILE_SCALED_INTERVAL_ID),
        (PROFILE_SCALED_INTERVAL_ID, PROFILE_RESIDUAL_ID),
        (CROSS_MODEL_ID, CANDIDATE_ID),
        (SELF_PREDICTION_ID, CANDIDATE_ID),
        (SCOUTED_FUSION_ID, CANDIDATE_ID),
        (SCOUTED_FUSION_ID, SELF_PREDICTION_ID),
    )
    predictions_by_seed: dict[int, dict[str, dict[str, Forecast]]] = {}
    metrics_by_seed: dict[int, dict[str, dict[str, float | int]]] = {}
    comparisons_by_seed: dict[int, dict[str, float | int]] = {}
    all_comparisons_by_seed: dict[
        int, dict[str, dict[str, float | int]]
    ] = {}
    folds_by_seed: dict[int, dict[str, list[dict[str, object]]]] = {}
    prediction_rows: list[dict[str, object]] = []
    for seed in STAGE_SPLIT_SEEDS:
        predictions: dict[str, dict[str, Forecast]] = {}
        fold_reports_by_candidate: dict[str, list[dict[str, object]]] = {}
        for candidate_id, (
            scores,
            call_scores,
            tokens_per_call_scores,
            mapping_ids,
            interval_mode,
        ) in candidate_specs.items():
            candidate_predictions, fold_reports = _evaluate_seed(
                seed,
                registry,
                development_tasks,
                outcomes,
                scores,
                call_scores,
                tokens_per_call_scores,
                candidate_id=candidate_id,
                mapping_ids=mapping_ids,
                interval_mode=interval_mode,
            )
            if REFERENCE_ID not in predictions:
                predictions[REFERENCE_ID] = candidate_predictions[REFERENCE_ID]
            elif predictions[REFERENCE_ID] != candidate_predictions[REFERENCE_ID]:
                raise ValueError("reference forecasts changed across candidates")
            predictions[candidate_id] = candidate_predictions[candidate_id]
            fold_reports_by_candidate[candidate_id] = fold_reports
        predictions_by_seed[seed] = predictions
        metrics_by_seed[seed] = {
            candidate_id: _metrics(values, outcomes)
            for candidate_id, values in predictions.items()
        }
        comparisons_by_seed[seed] = _paired_bootstrap(
            predictions,
            outcomes,
            iterations=args.bootstrap_iterations,
            seed=seed,
            candidate_id=SELECTED_CANDIDATE_ID,
        )
        all_comparisons_by_seed[seed] = {
            f"{candidate_id}_vs_{reference_id}": _paired_bootstrap(
                predictions,
                outcomes,
                iterations=args.bootstrap_iterations,
                seed=seed,
                candidate_id=candidate_id,
                reference_id=reference_id,
            )
            for candidate_id, reference_id in comparison_pairs
        }
        folds_by_seed[seed] = fold_reports_by_candidate
        fold_assignment = assign_family_folds(
            registry,
            folds=OUTER_FOLDS,
            seed=seed,
            family_ids=development_tasks,
        ).task_to_fold
        for candidate_id, values in predictions.items():
            for task_id, forecast in sorted(values.items()):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_assignment[task_id],
                        "candidate_id": candidate_id,
                        "task_id": task_id,
                        "source_task_id": outcomes[task_id].source_task_id,
                        "target_tokens": outcomes[task_id].total_tokens,
                        "target_call_count": outcomes[task_id].call_count,
                        "source_workload_score": source_scores[task_id],
                        "source_call_score": source_call_scores[task_id],
                        "source_tokens_per_call_score": source_tokens_per_call_scores[
                            task_id
                        ],
                        "document_variant_scores": {
                            variant_candidate_id: scores[0][task_id]
                            for variant_candidate_id, scores in document_scores.items()
                        },
                        "input_workload_score": input_workload_scores[task_id],
                        "output_workload_score": output_workload_scores[task_id],
                        "channel_fusion_score": channel_fusion_scores[task_id],
                        "cross_model_workload_score": cross_model_scores[task_id],
                        "self_prediction_gpt52_self_prediction": self_prediction_scores[task_id],
                        "scouted_fusion_score": scouted_fusion_scores[task_id],
                        **asdict(forecast),
                    }
                )

    pooled_errors = {
        candidate_id: {
            task_id: float(
                np.mean(
                    [
                        abs(
                            predictions_by_seed[seed][candidate_id][task_id].point
                            - outcomes[task_id].total_tokens
                        )
                        for seed in STAGE_SPLIT_SEEDS
                    ]
                )
            )
            for task_id in sorted(development_tasks)
        }
        for candidate_id in (REFERENCE_ID, *candidate_specs)
    }
    pooled_comparisons = {
        f"{candidate_id}_vs_{reference_id}": _paired_bootstrap_from_errors(
            pooled_errors[candidate_id],
            pooled_errors[reference_id],
            iterations=args.bootstrap_iterations,
            seed=STAGE_SPLIT_SEEDS[0],
        )
        for candidate_id, reference_id in comparison_pairs
    }
    seed_ensemble_points = {
        candidate_id: {
            task_id: float(
                np.exp(
                    np.mean(
                        [
                            math.log(
                                max(
                                    1e-9,
                                    predictions_by_seed[seed][candidate_id][task_id].point,
                                )
                            )
                            for seed in STAGE_SPLIT_SEEDS
                        ]
                    )
                )
            )
            for task_id in sorted(development_tasks)
        }
        for candidate_id in (REFERENCE_ID, SELECTED_CANDIDATE_ID)
    }
    seed_ensemble_errors = {
        candidate_id: {
            task_id: abs(point - outcomes[task_id].total_tokens)
            for task_id, point in points.items()
        }
        for candidate_id, points in seed_ensemble_points.items()
    }
    seed_ensemble_summary = {
        "candidate_id": SEED_ENSEMBLE_ID,
        "component_candidate_id": SELECTED_CANDIDATE_ID,
        "split_seed_fusion": "geometric_mean",
        "split_seeds": list(STAGE_SPLIT_SEEDS),
        "mae": float(
            np.mean(list(seed_ensemble_errors[SELECTED_CANDIDATE_ID].values()))
        ),
        "wape": float(
            sum(seed_ensemble_errors[SELECTED_CANDIDATE_ID].values())
            / sum(outcome.total_tokens for outcome in outcomes.values())
        ),
        "paired_task_bootstrap": _paired_bootstrap_from_errors(
            seed_ensemble_errors[SELECTED_CANDIDATE_ID],
            seed_ensemble_errors[REFERENCE_ID],
            iterations=args.bootstrap_iterations,
            seed=STAGE_SPLIT_SEEDS[0],
        ),
    }

    development_observers = [
        path.with_name("observer.jsonl")
        for task_id in sorted(development_tasks)
        for path in captures[task_id]
    ]
    result: dict[str, object] = {
        "schema_version": 3,
        "experiment_id": "task-initialization-static-transfer-v6",
        "status": "development_selection_evidence",
        "protocol": {
            "target": "task_total_accounted_tokens",
            "position": "task_launch",
            "outer_folds": OUTER_FOLDS,
            "split_seeds": list(STAGE_SPLIT_SEEDS),
            "development_task_count": len(development_tasks),
            "final_holdout_task_count": len(final_holdout_tasks),
            "final_holdout_evaluated": False,
            "final_holdout_observers_read": False,
            "source_exact_target_tasks_excluded": len(excluded_source_tasks),
            "family_policy": "declared exact task-family registry",
            "point_selection_partition": "outer validation only",
            "point_refit_partition": "outer train plus validation",
            "interval_calibration_partition": "outer calibration only",
            "core_information_budget": "static task-start context only",
            "scouted_information_budget": (
                "published pre-execution Agent with repository and tool access"
            ),
        },
        "candidate": {
            "candidate_id": SELECTED_CANDIDATE_ID,
            "source_model_id": SOURCE_MODEL_ID,
            "source_target": "task input/output Token totals",
            "document_variant": "difficulty_problem",
            "ridge_alpha": RIDGE_ALPHA,
            "channel_fusion": "geometric_mean",
            "mapping_candidates": list(SCALAR_MAPPING_IDS),
            "interval_mode": "scaled",
            "source_fit": {
                "input": input_workload_fit,
                "output": output_workload_fit,
            },
        },
        "point_ensemble": seed_ensemble_summary,
        "candidate_catalog": {
            CANDIDATE_ID: {
                "setting": "core_task_initialization",
                "source": "GPT-5.2 execution profiles",
                "task_specific_pre_execution_calls": False,
            },
            **{
                candidate_id: {
                    "setting": "core_task_initialization",
                    "source": "GPT-5.2 execution profiles",
                    "document_variant": document_variant,
                    "task_specific_pre_execution_calls": False,
                    "source_fit": document_fits[candidate_id],
                }
                for candidate_id, document_variant in DOCUMENT_CANDIDATES
            },
            PROFILE_RESIDUAL_ID: {
                "setting": "core_task_initialization",
                "source": (
                    "GPT-5.2 total-workload, Call-count, and per-Call profiles"
                ),
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "mapping_candidates": [PROFILE_RESIDUAL_MAPPING_ID],
                "ridge_alpha": PROFILE_RESIDUAL_ALPHA,
                "source_fit": document_fits[DIFFICULTY_TEXT_ID],
            },
            DIFFICULTY_SCALED_ID: {
                "setting": "core_task_initialization",
                "source": "GPT-5.2 execution profiles",
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "mapping_candidates": list(MAPPING_IDS),
                "interval_mode": "scaled",
                "source_fit": document_fits[DIFFICULTY_TEXT_ID],
            },
            PROFILE_SCALED_INTERVAL_ID: {
                "setting": "core_task_initialization",
                "source": (
                    "GPT-5.2 total-workload, Call-count, and per-Call profiles"
                ),
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "mapping_candidates": [PROFILE_RESIDUAL_MAPPING_ID],
                "interval_mode": "scaled",
                "ridge_alpha": PROFILE_RESIDUAL_ALPHA,
                "source_fit": document_fits[DIFFICULTY_TEXT_ID],
            },
            INPUT_WORKLOAD_ID: {
                "setting": "core_task_initialization",
                "source": "GPT-5.2 input-Token profiles",
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "source_fit": input_workload_fit,
            },
            OUTPUT_WORKLOAD_ID: {
                "setting": "core_task_initialization",
                "source": "GPT-5.2 output-Token profiles",
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "source_fit": output_workload_fit,
            },
            CHANNEL_FUSION_ID: {
                "setting": "core_task_initialization",
                "source": "fixed geometric input/output workload fusion",
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "mapping_candidates": list(SCALAR_MAPPING_IDS),
            },
            CHANNEL_FUSION_SCALED_ID: {
                "setting": "core_task_initialization",
                "source": "fixed geometric input/output workload fusion",
                "document_variant": "difficulty_problem",
                "task_specific_pre_execution_calls": False,
                "mapping_candidates": list(SCALAR_MAPPING_IDS),
                "interval_mode": "scaled",
            },
            CROSS_MODEL_ID: {
                "setting": "core_task_initialization",
                "source": "eight-model normalized execution profiles",
                "task_specific_pre_execution_calls": False,
                "source_fit": cross_model_fit,
            },
            SELF_PREDICTION_ID: {
                "setting": "scouted_pre_execution",
                "source": "published GPT-5.2 self-prediction",
                "task_specific_pre_execution_calls": True,
                "mapping_candidates": list(SCALAR_MAPPING_IDS),
            },
            SCOUTED_FUSION_ID: {
                "setting": "scouted_pre_execution",
                "source": (
                    "geometric fusion of GPT-5.2 profile, cross-model profile, "
                    "and published GPT-5.2 self-prediction"
                ),
                "task_specific_pre_execution_calls": True,
                "mapping_candidates": list(SCALAR_MAPPING_IDS),
            },
        },
        "reference": {"candidate_id": REFERENCE_ID},
        "metrics_by_seed": metrics_by_seed,
        "comparisons_by_seed": comparisons_by_seed,
        "all_comparisons_by_seed": all_comparisons_by_seed,
        "fold_fit_reports": folds_by_seed,
        "stability_summary": {
            **{
                candidate_id: _mean_metrics(metrics_by_seed, candidate_id)
                for candidate_id in (REFERENCE_ID, *candidate_specs)
            },
            "paired_task_bootstrap_on_seed_mean_errors": pooled_comparisons[
                f"{SELECTED_CANDIDATE_ID}_vs_{REFERENCE_ID}"
            ],
            "paired_comparisons": pooled_comparisons,
        },
        "provenance": {
            "source_calls": str(args.source_calls.resolve()),
            "source_calls_sha256": _sha256(args.source_calls),
            "task_metadata": str(args.task_metadata.resolve()),
            "task_metadata_sha256": _sha256(args.task_metadata),
            "self_predictions": str(args.self_predictions.resolve()),
            "self_predictions_sha256": _sha256(args.self_predictions),
            "captures_root": str(args.captures_root.resolve()),
            "capture_count": len(captures),
            "development_observer_count": len(development_observers),
            "development_observers_combined_sha256": _combined_digest(
                development_observers
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    summary = result["stability_summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
