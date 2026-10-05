from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    fit_calibrator,
    paired_task_bootstrap,
)
from token_prediction.experiments.candidates import PredictionRecord
from token_prediction.config import project_root_for


CALL_START = "call_start"
TASK_UPDATE = "task_update"
CONDITION_ID = "public-trajectory-count-only-v1"
CAUSAL_STATE_FEATURES = (
    "log_completed_calls",
    "has_history",
    "log_cumulative_output_tokens",
    "log_last_output_tokens",
    "log_mean_output_tokens",
    "log_ema_output_tokens",
    "log_max_output_tokens",
    "output_log_delta",
    "log_cumulative_new_input_tokens",
    "log_last_new_input_tokens",
    "log_mean_new_input_tokens",
    "new_input_log_delta",
    "log_last_input_tokens",
    "last_input_tokens_missing",
)
POSITION_TARGETS = {
    CALL_START: PredictionTarget.CALL_BILLABLE_OUTPUT_TOKENS,
    TASK_UPDATE: PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
}
PRIMARY_REFERENCES = {
    CALL_START: "ema",
    TASK_UPDATE: "call_clock",
}
CANDIDATE_ID = "causal_state_lightgbm"
DEFAULT_GRU_CONFIG = {
    "hidden_dim": 32,
    "learning_rate": 0.01,
    "max_epochs": 200,
    "patience": 20,
    "threads": 4,
}
QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class TraceTurn:
    output_tokens: int
    new_input_tokens: int
    input_tokens: int | None


@dataclass(frozen=True)
class TraceTask:
    task_id: str
    dataset: str
    model: str
    turns: tuple[TraceTurn, ...]


@dataclass(frozen=True)
class TaskPartition:
    pilot_tasks: tuple[str, ...]
    development_reserve_tasks: tuple[str, ...]
    final_holdout_tasks: tuple[str, ...]


@dataclass(frozen=True)
class PilotPoint:
    point_id: str
    task_id: str
    position: str
    call_index: int
    features: Mapping[str, float]
    target_value: float
    sample_weight: float
    fold: int
    model: str


class RawPredictor:
    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        raise NotImplementedError

    def report(self) -> Mapping[str, object]:
        return {}


@dataclass(frozen=True)
class ConstantPredictor(RawPredictor):
    interval: tuple[float, float, float]

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        return np.repeat(np.asarray([self.interval], dtype=np.float64), len(points), axis=0)


@dataclass(frozen=True)
class LogResidualPredictor(RawPredictor):
    fallback: float
    residual_interval: tuple[float, float, float]
    center_feature: str = "log_ema_output_tokens"

    def _center(self, point: PilotPoint) -> float:
        if point.features["has_history"] > 0.0:
            return math.expm1(point.features[self.center_feature])
        return self.fallback

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        rows = []
        for point in points:
            center_log = math.log1p(self._center(point))
            rows.append(
                [
                    max(0.0, math.expm1(max(0.0, center_log + residual)))
                    for residual in self.residual_interval
                ]
            )
        return _repair_predictions(np.asarray(rows, dtype=np.float64))


@dataclass(frozen=True)
class CallClockPredictor(RawPredictor):
    intervals: Mapping[int, tuple[float, float, float]]
    max_bucket: int

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        available = tuple(sorted(self.intervals))
        rows = []
        for point in points:
            bucket = min(int(point.features["completed_calls"]), self.max_bucket)
            if bucket not in self.intervals:
                bucket = min(available, key=lambda value: abs(value - bucket))
            rows.append(self.intervals[bucket])
        return np.asarray(rows, dtype=np.float64)

    def report(self) -> Mapping[str, object]:
        return {
            "max_bucket": self.max_bucket,
            "buckets": list(sorted(self.intervals)),
        }


@dataclass(frozen=True)
class LinearExtrapolationPredictor(RawPredictor):


    total_calls: tuple[int, ...]

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        totals = np.asarray(self.total_calls, dtype=np.float64)
        rows = []
        for point in points:
            completed = int(point.features["completed_calls"])
            cumulative = math.expm1(point.features["log_cumulative_output_tokens"])
            rate = cumulative / max(completed, 1)
            eligible = totals[np.searchsorted(totals, completed, side="right") :]
            if eligible.size:
                remaining_calls = np.quantile(eligible, QUANTILES) - completed
            else:
                remaining_calls = np.zeros(len(QUANTILES))
            rows.append(rate * remaining_calls)
        return _repair_predictions(np.asarray(rows, dtype=np.float64))

    def report(self) -> Mapping[str, object]:
        return {
            "train_task_count": len(self.total_calls),
            "max_total_calls": max(self.total_calls),
        }


class HistoryGruPredictor(RawPredictor):


    def __init__(
        self,
        module: object,
        sequences: Mapping[str, tuple[TraceTurn, ...]],
        train_report: Mapping[str, object],
    ) -> None:
        self._module = module
        self._sequences = sequences
        self._train_report = dict(train_report)

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        import torch

        self._module.eval()
        rows = np.zeros((len(points), len(QUANTILES)), dtype=np.float64)
        by_task: dict[str, list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            by_task[point.task_id].append(index)
        with torch.no_grad():
            for task_id, indexes in by_task.items():
                tensor = _turn_tensor(torch, self._sequences[task_id]).unsqueeze(0)
                log_quantiles = self._module(tensor)[0].numpy()
                for index in indexes:
                    cutoff = points[index].call_index
                    rows[index] = np.expm1(np.maximum(log_quantiles[cutoff], 0.0))
        return _repair_predictions(rows)

    def report(self) -> Mapping[str, object]:
        return self._train_report

    def with_sequences(
        self, sequences: Mapping[str, tuple[TraceTurn, ...]]
    ) -> "HistoryGruPredictor":

        return HistoryGruPredictor(self._module, sequences, self._train_report)


def _turn_tensor(torch: object, turns: Sequence[TraceTurn]) -> object:

    rows = [[0.0, 0.0, 0.0, 0.0]]
    for turn in turns:
        rows.append(
            [
                math.log1p(turn.output_tokens),
                math.log1p(turn.new_input_tokens),
                math.log1p(turn.input_tokens or 0),
                float(turn.input_tokens is None),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class LightGbmQuantilePredictor(RawPredictor):
    feature_names: tuple[str, ...]
    models: tuple[lgb.Booster, lgb.Booster, lgb.Booster]

    def predict(self, points: Sequence[PilotPoint]) -> np.ndarray:
        matrix = _matrix(points, self.feature_names)
        values = np.column_stack(
            [np.expm1(np.maximum(model.predict(matrix), 0.0)) for model in self.models]
        )
        return _repair_predictions(values)

    def report(self) -> Mapping[str, object]:
        median_model = self.models[1]
        gain = median_model.feature_importance(importance_type="gain").astype(np.float64)
        total = float(np.sum(gain))
        normalized = gain / total if total > 0.0 else gain
        return {
            "best_iterations": [model.best_iteration for model in self.models],
            "median_gain_importance": {
                name: float(value)
                for name, value in sorted(
                    zip(self.feature_names, normalized),
                    key=lambda item: (-item[1], item[0]),
                )
            },
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_validation_v1.toml"),
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold-seed", type=int,
                        help="repeat the study under another fold partition of the same pilot cohort; "
                             "the sealed holdout is tied to split_seed and never moves")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_task_ids(path: Path) -> tuple[str, ...]:
    values = pq.read_table(path, columns=["instance_id"]).column("instance_id").to_pylist()
    task_ids = tuple(str(value) for value in values)
    if not task_ids or len(task_ids) != len(set(task_ids)) or any(not value for value in task_ids):
        raise ValueError("source instance_id values must be non-empty and unique")
    return task_ids


def _partition_task_ids(
    task_ids: Sequence[str],
    *,
    seed: int,
    final_holdout_fraction: float,
    pilot_tasks: int,
) -> TaskPartition:
    if not 0.0 < final_holdout_fraction < 1.0:
        raise ValueError("final_holdout_fraction must be in (0, 1)")
    ordered = sorted(task_ids)
    random.Random(seed).shuffle(ordered)
    holdout_count = round(len(ordered) * final_holdout_fraction)
    if pilot_tasks <= 0 or holdout_count + pilot_tasks > len(ordered):
        raise ValueError("pilot task count does not fit outside the final holdout")
    return TaskPartition(
        final_holdout_tasks=tuple(sorted(ordered[:holdout_count])),
        pilot_tasks=tuple(sorted(ordered[holdout_count : holdout_count + pilot_tasks])),
        development_reserve_tasks=tuple(sorted(ordered[holdout_count + pilot_tasks :])),
    )


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _parse_turn(value: object) -> TraceTurn:
    if not isinstance(value, str):
        raise ValueError("pinned source turns must contain JSON strings")
    document = json.loads(value)
    if not isinstance(document, dict):
        raise ValueError("turn JSON must be an object")
    input_tokens = document.get("isl")
    return TraceTurn(
        output_tokens=_integer(document.get("osl"), name="osl", positive=True),
        new_input_tokens=_integer(document.get("isl_new"), name="isl_new"),
        input_tokens=(None if input_tokens is None else _integer(input_tokens, name="isl")),
    )


def _load_selected_tasks(path: Path, selected_task_ids: frozenset[str]) -> tuple[TraceTask, ...]:
    if not selected_task_ids:
        raise ValueError("selected task ids are empty")
    table = pq.read_table(
        path,
        columns=["instance_id", "dataset", "model", "turns"],
        filters=[("instance_id", "in", sorted(selected_task_ids))],
    )
    rows = table.to_pylist()
    observed_ids = {str(row["instance_id"]) for row in rows}
    if observed_ids != selected_task_ids:
        missing = sorted(selected_task_ids - observed_ids)
        raise ValueError(f"selected source tasks are missing: {missing[:3]}")
    tasks = []
    for row in rows:
        turns = tuple(_parse_turn(value) for value in row["turns"])
        if not turns:
            raise ValueError("source trajectory contains no turns")
        tasks.append(
            TraceTask(
                task_id=str(row["instance_id"]),
                dataset=str(row["dataset"]),
                model=str(row["model"]),
                turns=turns,
            )
        )
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _ema(values: Sequence[int], decay: float = 0.5) -> float:
    if not values:
        return 0.0
    result = float(values[0])
    for value in values[1:]:
        result = decay * float(value) + (1.0 - decay) * result
    return result


def _prefix_features(turns: Sequence[TraceTurn]) -> dict[str, float]:
    outputs = [turn.output_tokens for turn in turns]
    new_inputs = [turn.new_input_tokens for turn in turns]
    previous_output = outputs[-2] if len(outputs) > 1 else outputs[-1] if outputs else 0
    previous_new_input = (
        new_inputs[-2] if len(new_inputs) > 1 else new_inputs[-1] if new_inputs else 0
    )
    last_input = turns[-1].input_tokens if turns else None
    return {
        "completed_calls": float(len(turns)),
        "log_completed_calls": math.log1p(len(turns)),
        "has_history": float(bool(turns)),
        "log_cumulative_output_tokens": math.log1p(sum(outputs)),
        "log_last_output_tokens": math.log1p(outputs[-1] if outputs else 0),
        "log_mean_output_tokens": math.log1p(sum(outputs) / len(outputs) if outputs else 0),
        "log_ema_output_tokens": math.log1p(_ema(outputs)),
        "log_max_output_tokens": math.log1p(max(outputs, default=0)),
        "output_log_delta": (
            math.log1p(outputs[-1]) - math.log1p(previous_output) if outputs else 0.0
        ),
        "log_cumulative_new_input_tokens": math.log1p(sum(new_inputs)),
        "log_last_new_input_tokens": math.log1p(new_inputs[-1] if new_inputs else 0),
        "log_mean_new_input_tokens": math.log1p(
            sum(new_inputs) / len(new_inputs) if new_inputs else 0
        ),
        "new_input_log_delta": (
            math.log1p(new_inputs[-1]) - math.log1p(previous_new_input) if new_inputs else 0.0
        ),
        "log_last_input_tokens": math.log1p(last_input or 0),
        "last_input_tokens_missing": float(last_input is None),
    }


def _assign_folds(task_ids: Sequence[str], *, folds: int, seed: int) -> dict[str, int]:
    if folds < 5:
        raise ValueError("the train/validation/calibration/test protocol requires at least 5 folds")
    ordered = sorted(task_ids)
    random.Random(seed + 1).shuffle(ordered)
    return {task_id: index % folds for index, task_id in enumerate(ordered)}


def _build_points(
    tasks: Sequence[TraceTask], fold_by_task: Mapping[str, int]
) -> tuple[PilotPoint, ...]:
    points = []
    for task in tasks:
        fold = fold_by_task[task.task_id]
        suffix_outputs = [0] * len(task.turns)
        running = 0
        for index in range(len(task.turns) - 1, -1, -1):
            running += task.turns[index].output_tokens
            suffix_outputs[index] = running
        for index, turn in enumerate(task.turns):
            features = _prefix_features(task.turns[:index])
            points.append(
                PilotPoint(
                    point_id=f"{task.task_id}:call-start:{index}",
                    task_id=task.task_id,
                    position=CALL_START,
                    call_index=index,
                    features=features,
                    target_value=float(turn.output_tokens),
                    sample_weight=1.0 / len(task.turns),
                    fold=fold,
                    model=task.model,
                )
            )
            if index > 0:
                points.append(
                    PilotPoint(
                        point_id=f"{task.task_id}:task-update:{index}",
                        task_id=task.task_id,
                        position=TASK_UPDATE,
                        call_index=index,
                        features=features,
                        target_value=float(suffix_outputs[index]),
                        sample_weight=1.0 / (len(task.turns) - 1),
                        fold=fold,
                        model=task.model,
                    )
                )
    return tuple(sorted(points, key=lambda point: point.point_id))


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = probability * sum(weight for _value, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _interval(points: Sequence[PilotPoint]) -> tuple[float, float, float]:
    values = [point.target_value for point in points]
    weights = [point.sample_weight for point in points]
    return tuple(_weighted_quantile(values, weights, q) for q in (0.05, 0.5, 0.95))


def _fit_empirical(points: Sequence[PilotPoint]) -> ConstantPredictor:
    return ConstantPredictor(_interval(points))


def _fit_log_residual(points: Sequence[PilotPoint], *, center_feature: str) -> LogResidualPredictor:
    fallback = _weighted_quantile(
        [point.target_value for point in points],
        [point.sample_weight for point in points],
        0.5,
    )
    centers = [
        (
            math.expm1(point.features[center_feature])
            if point.features["has_history"] > 0.0
            else fallback
        )
        for point in points
    ]
    residuals = [
        math.log1p(point.target_value) - math.log1p(center)
        for point, center in zip(points, centers)
    ]
    weights = [point.sample_weight for point in points]
    residual_interval = tuple(_weighted_quantile(residuals, weights, q) for q in QUANTILES)
    return LogResidualPredictor(fallback, residual_interval, center_feature)


def _fit_ema(points: Sequence[PilotPoint]) -> LogResidualPredictor:
    return _fit_log_residual(points, center_feature="log_ema_output_tokens")


def _fit_last_value(points: Sequence[PilotPoint]) -> LogResidualPredictor:
    return _fit_log_residual(points, center_feature="log_last_output_tokens")


def _fit_linear_extrapolation(points: Sequence[PilotPoint]) -> LinearExtrapolationPredictor:
    totals: dict[str, int] = {}
    for point in points:
        completed = int(point.features["completed_calls"])
        totals[point.task_id] = max(totals.get(point.task_id, 0), completed + 1)
    return LinearExtrapolationPredictor(total_calls=tuple(sorted(totals.values())))


def _gru_training_batch(
    torch: object,
    points: Sequence[PilotPoint],
    sequences: Mapping[str, tuple[TraceTurn, ...]],
) -> tuple[object, object, object]:
    task_ids = sorted({point.task_id for point in points})
    steps = max(len(sequences[task_id]) for task_id in task_ids) + 1
    inputs = torch.zeros((len(task_ids), steps, 4), dtype=torch.float32)
    targets = torch.zeros((len(task_ids), steps), dtype=torch.float32)
    weights = torch.zeros((len(task_ids), steps), dtype=torch.float32)
    row_by_task = {task_id: row for row, task_id in enumerate(task_ids)}
    for task_id in task_ids:
        tensor = _turn_tensor(torch, sequences[task_id])
        inputs[row_by_task[task_id], : tensor.shape[0]] = tensor
    for point in points:
        row = row_by_task[point.task_id]
        targets[row, point.call_index] = math.log1p(point.target_value)
        weights[row, point.call_index] = point.sample_weight
    return inputs, targets, weights


def _gru_loss(torch: object, predictions: object, targets: object, weights: object) -> object:
    total = torch.zeros((), dtype=torch.float32)
    for column, quantile in enumerate(QUANTILES):
        error = targets - predictions[:, :, column]
        pinball = torch.maximum(quantile * error, (quantile - 1.0) * error)
        total = total + (weights * pinball).sum()
    return total / (weights.sum() * len(QUANTILES))


def _fit_history_gru(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    *,
    sequences: Mapping[str, tuple[TraceTurn, ...]],
    gru_config: Mapping[str, object],
    seed: int,
) -> HistoryGruPredictor:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(int(gru_config["threads"]))
    hidden_dim = int(gru_config["hidden_dim"])

    class QuantileGru(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(4, hidden_dim, batch_first=True)
            self.head = torch.nn.Linear(hidden_dim, len(QUANTILES))

        def forward(self, batch: object) -> object:
            outputs, _state = self.gru(batch)
            return self.head(outputs)

    module = QuantileGru()
    optimizer = torch.optim.Adam(module.parameters(), lr=float(gru_config["learning_rate"]))
    train_batch = _gru_training_batch(torch, train, sequences)
    validation_batch = _gru_training_batch(torch, validation, sequences)
    best_state = {key: value.clone() for key, value in module.state_dict().items()}
    best_loss = math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(int(gru_config["max_epochs"])):
        module.train()
        optimizer.zero_grad()
        loss = _gru_loss(torch, module(train_batch[0]), train_batch[1], train_batch[2])
        loss.backward()
        optimizer.step()
        module.eval()
        with torch.no_grad():
            validation_loss = float(
                _gru_loss(
                    torch,
                    module(validation_batch[0]),
                    validation_batch[1],
                    validation_batch[2],
                )
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(gru_config["patience"]):
                break
    module.load_state_dict(best_state)
    module.eval()
    return HistoryGruPredictor(
        module,
        sequences,
        {
            "hidden_dim": hidden_dim,
            "best_epoch": best_epoch,
            "best_validation_pinball": best_loss,
        },
    )


def _fit_call_clock(points: Sequence[PilotPoint], *, min_bucket_tasks: int) -> CallClockPredictor:
    counts = Counter(int(point.features["completed_calls"]) for point in points)
    supported = [value for value, count in counts.items() if count >= min_bucket_tasks]
    max_bucket = max(supported) if supported else max(counts)
    grouped: dict[int, list[PilotPoint]] = defaultdict(list)
    for point in points:
        grouped[min(int(point.features["completed_calls"]), max_bucket)].append(point)
    return CallClockPredictor(
        intervals={bucket: _interval(rows) for bucket, rows in grouped.items()},
        max_bucket=max_bucket,
    )


def _matrix(points: Sequence[PilotPoint], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[point.features[name] for name in feature_names] for point in points],
        dtype=np.float64,
    )


def _fit_lightgbm(
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    *,
    feature_names: tuple[str, ...],
    model_config: Mapping[str, object],
    seed: int,
) -> LightGbmQuantilePredictor:
    train_matrix = _matrix(train, feature_names)
    validation_matrix = _matrix(validation, feature_names)
    train_labels = np.log1p([point.target_value for point in train])
    validation_labels = np.log1p([point.target_value for point in validation])
    train_weights = np.asarray([point.sample_weight for point in train], dtype=np.float64)
    validation_weights = np.asarray([point.sample_weight for point in validation], dtype=np.float64)
    models = []
    for quantile in (0.05, 0.5, 0.95):
        params = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": quantile,
            "learning_rate": float(model_config["learning_rate"]),
            "num_leaves": int(model_config["num_leaves"]),
            "min_data_in_leaf": int(model_config["min_data_in_leaf"]),
            "max_depth": int(model_config["max_depth"]),
            "lambda_l2": float(model_config["lambda_l2"]),
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": int(model_config["threads"]),
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
        train_data = lgb.Dataset(
            train_matrix,
            label=train_labels,
            weight=train_weights,
            feature_name=list(feature_names),
            free_raw_data=False,
        )
        validation_data = lgb.Dataset(
            validation_matrix,
            label=validation_labels,
            weight=validation_weights,
            feature_name=list(feature_names),
            reference=train_data,
            free_raw_data=False,
        )
        models.append(
            lgb.train(
                params,
                train_data,
                num_boost_round=int(model_config["rounds"]),
                valid_sets=[validation_data],
                callbacks=[
                    lgb.early_stopping(int(model_config["early_stopping_rounds"]), verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        )
    return LightGbmQuantilePredictor(feature_names, tuple(models))


def _repair_predictions(values: np.ndarray) -> np.ndarray:
    repaired = np.maximum(values, 0.0)
    repaired[:, 0] = np.minimum(repaired[:, 0], repaired[:, 1])
    repaired[:, 2] = np.maximum(repaired[:, 2], repaired[:, 1])
    return repaired


def _raw_forecasts(
    predictor: RawPredictor,
    points: Sequence[PilotPoint],
    target: PredictionTarget,
) -> tuple[dict[str, TokenForecast], float]:
    started = time.perf_counter()
    values = predictor.predict(points)
    latency_ms = 1000.0 * (time.perf_counter() - started) / max(len(points), 1)
    forecasts = {
        point.point_id: TokenForecast(
            point_id=point.point_id,
            target=target,
            lower=float(row[0]),
            point=float(row[1]),
            upper=float(row[2]),
            latency_ms=latency_ms,
        )
        for point, row in zip(points, values)
    }
    return forecasts, latency_ms


def _fold_partition(
    points: Sequence[PilotPoint], fold: int, folds: int
) -> tuple[list[PilotPoint], list[PilotPoint], list[PilotPoint], list[PilotPoint]]:
    calibration_fold = (fold + 1) % folds
    validation_fold = (fold + 2) % folds
    test = [point for point in points if point.fold == fold]
    calibration = [point for point in points if point.fold == calibration_fold]
    validation = [point for point in points if point.fold == validation_fold]
    train = [
        point for point in points if point.fold not in {fold, calibration_fold, validation_fold}
    ]
    task_sets = [
        {point.task_id for point in partition}
        for partition in (train, validation, calibration, test)
    ]
    if any(not partition for partition in (train, validation, calibration, test)):
        raise ValueError("fold partition contains no points")
    if any(task_sets[left] & task_sets[right] for left in range(4) for right in range(left)):
        raise ValueError("task leaked across a fold partition")
    return train, validation, calibration, test


def _fit_predictors(
    position: str,
    train: Sequence[PilotPoint],
    validation: Sequence[PilotPoint],
    *,
    protocol: Mapping[str, object],
    model_config: Mapping[str, object],
    sequences: Mapping[str, tuple[TraceTurn, ...]],
    gru_config: Mapping[str, object],
    seed: int,
) -> dict[str, RawPredictor]:
    predictors: dict[str, RawPredictor] = {
        "empirical_quantile": _fit_empirical(train),
        CANDIDATE_ID: _fit_lightgbm(
            train,
            validation,
            feature_names=CAUSAL_STATE_FEATURES,
            model_config=model_config,
            seed=seed,
        ),
        "history_gru": _fit_history_gru(
            train,
            validation,
            sequences=sequences,
            gru_config=gru_config,
            seed=seed,
        ),
    }
    if position == CALL_START:
        predictors["ema"] = _fit_ema(train)
        predictors["last_value"] = _fit_last_value(train)
    else:
        predictors["call_clock"] = _fit_call_clock(
            train,
            min_bucket_tasks=int(protocol["min_call_clock_tasks"]),
        )
        predictors["linear_extrapolation"] = _fit_linear_extrapolation(train)
    return predictors


def _run_position(
    position: str,
    points: Sequence[PilotPoint],
    *,
    folds: int,
    alpha: float,
    protocol: Mapping[str, object],
    model_config: Mapping[str, object],
    sequences: Mapping[str, tuple[TraceTurn, ...]],
    gru_config: Mapping[str, object],
    split_seed: int,
) -> tuple[
    dict[str, list[PredictionRecord]],
    dict[str, object],
    list[dict[str, object]],
]:
    target = POSITION_TARGETS[position]
    records: dict[str, list[PredictionRecord]] = defaultdict(list)
    scored: dict[str, list[ScoredForecast]] = defaultdict(list)
    fold_reports = []
    for fold in range(folds):
        train, validation, calibration, test = _fold_partition(points, fold, folds)
        predictors = _fit_predictors(
            position,
            train,
            validation,
            protocol=protocol,
            model_config=model_config,
            sequences=sequences,
            gru_config=gru_config,
            seed=split_seed + fold,
        )
        arm_reports = {}
        for candidate_id, predictor in predictors.items():
            calibration_forecasts, _ = _raw_forecasts(predictor, calibration, target)
            calibrator = fit_calibrator(
                "task_max_log_conformal",
                alpha=alpha,
                examples=[
                    CalibrationExample(
                        task_id=point.task_id,
                        forecast=calibration_forecasts[point.point_id],
                        target_value=point.target_value,
                    )
                    for point in calibration
                ],
            )
            test_forecasts, latency_ms = _raw_forecasts(predictor, test, target)
            for point in test:
                forecast = calibrator.transform(test_forecasts[point.point_id])
                record = PredictionRecord(
                    candidate_id=candidate_id,
                    point_id=point.point_id,
                    task_id=point.task_id,
                    trajectory_id=point.task_id,
                    condition_id=CONDITION_ID,
                    fold=fold,
                    target=target,
                    forecast=forecast,
                    sample_weight=point.sample_weight,
                    token_unit_contract_id="provider-output-token-v1",
                    task_family_id=point.task_id,
                    split_group_id=point.task_id,
                    actor_model_id=point.model,
                    model_family_id=point.model,
                )
                records[candidate_id].append(record)
                scored[candidate_id].append(
                    ScoredForecast(
                        task_id=point.task_id,
                        trajectory_id=point.task_id,
                        forecast=forecast,
                        target_value=point.target_value,
                        sample_weight=point.sample_weight,
                        task_family_id=point.task_id,
                        split_group_id=point.task_id,
                        actor_model_id=point.model,
                        model_family_id=point.model,
                    )
                )
            arm_reports[candidate_id] = {
                "predictor": predictor.report(),
                "calibrator": calibrator.to_dict(),
                "prediction_latency_ms_per_point": latency_ms,
            }
        fold_reports.append(
            {
                "fold": fold,
                "task_counts": {
                    name: len({point.task_id for point in values})
                    for name, values in (
                        ("train", train),
                        ("validation", validation),
                        ("calibration", calibration),
                        ("test", test),
                    )
                },
                "point_counts": {
                    "train": len(train),
                    "validation": len(validation),
                    "calibration": len(calibration),
                    "test": len(test),
                },
                "arms": arm_reports,
            }
        )
    truth = {point.point_id: point.target_value for point in points}
    metrics = {
        candidate_id: evaluate_forecasts(values, alpha=alpha)
        for candidate_id, values in scored.items()
    }
    primary_reference = PRIMARY_REFERENCES[position]
    reference_order = [primary_reference, "empirical_quantile"] + sorted(
        arm_id
        for arm_id in records
        if arm_id not in {CANDIDATE_ID, primary_reference, "empirical_quantile"}
    )
    comparisons = {
        f"{CANDIDATE_ID}_vs_{reference_id}": asdict(
            paired_task_bootstrap(
                records[CANDIDATE_ID],
                records[reference_id],
                truth,
                iterations=int(protocol["bootstrap_iterations"]),
                seed=int(protocol["bootstrap_seed"]),
            )
        )
        for reference_id in reference_order
    }
    return records, {"metrics": metrics, "comparisons": comparisons}, fold_reports


def _percentile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _cohort_audit(tasks: Sequence[TraceTask], points: Sequence[PilotPoint]) -> dict[str, object]:
    call_counts = [len(task.turns) for task in tasks]
    task_outputs = [sum(turn.output_tokens for turn in task.turns) for task in tasks]
    return {
        "tasks": len(tasks),
        "calls": sum(call_counts),
        "datasets": sorted({task.dataset for task in tasks}),
        "models": sorted({task.model for task in tasks}),
        "call_count": {
            "min": min(call_counts),
            "median": _percentile(call_counts, 0.5),
            "p90": _percentile(call_counts, 0.9),
            "max": max(call_counts),
        },
        "task_output_tokens": {
            "median": _percentile(task_outputs, 0.5),
            "p90": _percentile(task_outputs, 0.9),
            "max": max(task_outputs),
        },
        "points": {
            position: len([point for point in points if point.position == position])
            for position in (CALL_START, TASK_UPDATE)
        },
        "tasks_by_position": {
            position: len({point.task_id for point in points if point.position == position})
            for position in (CALL_START, TASK_UPDATE)
        },
    }


def _prediction_document(record: PredictionRecord) -> dict[str, object]:
    return {
        "candidate_id": record.candidate_id,
        "point_id": record.point_id,
        "task_id": record.task_id,
        "fold": record.fold,
        "target": record.target.value,
        "sample_weight": record.sample_weight,
        "token_unit_contract_id": record.token_unit_contract_id,
        "forecast": {
            "lower": record.forecast.lower,
            "point": record.forecast.point,
            "upper": record.forecast.upper,
            "raw_lower": record.forecast.raw_lower,
            "raw_point": record.forecast.raw_point,
            "raw_upper": record.forecast.raw_upper,
        },
    }


def run(
    config_path: Path,
    *,
    data_override: Path | None = None,
    output_override: Path | None = None,
    fold_seed_override: int | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    source = config["source"]
    protocol = config["protocol"]
    model_config = config["model"]
    data_path = (data_override or root / source["local_path"]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"source parquet is missing at {data_path}; download {source['url']}"
        )
    actual_sha256 = _sha256(data_path)
    if actual_sha256.lower() != str(source["sha256"]).lower():
        raise ValueError("source parquet SHA-256 does not match the frozen config")

    all_task_ids = _read_task_ids(data_path)


    fold_seed = int(
        fold_seed_override
        if fold_seed_override is not None
        else protocol.get("fold_seed", protocol["split_seed"])
    )
    partition = _partition_task_ids(
        all_task_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    selected = frozenset(partition.pilot_tasks)
    tasks = _load_selected_tasks(data_path, selected)
    if {task.task_id for task in tasks} & set(partition.final_holdout_tasks):
        raise ValueError("final holdout entered the decoded pilot cohort")
    fold_by_task = _assign_folds(
        partition.pilot_tasks,
        folds=int(protocol["folds"]),
        seed=fold_seed,
    )
    points = _build_points(tasks, fold_by_task)
    sequences = {task.task_id: task.turns for task in tasks}
    gru_config = {**DEFAULT_GRU_CONFIG, **config.get("gru", {})}

    position_results = {}
    all_records = []
    fold_reports = {}
    for position in (CALL_START, TASK_UPDATE):
        position_points = [point for point in points if point.position == position]
        records, result, reports = _run_position(
            position,
            position_points,
            folds=int(protocol["folds"]),
            alpha=float(protocol["alpha"]),
            protocol=protocol,
            model_config=model_config,
            sequences=sequences,
            gru_config=gru_config,
            split_seed=fold_seed,
        )
        position_results[position] = result
        fold_reports[position] = reports
        all_records.extend(record for values in records.values() for record in values)

    result = {
        "config_id": config["config_id"],
        "source": {
            **source,
            "local_path": str(data_path),
            "actual_sha256": actual_sha256,
            "source_rows": len(all_task_ids),
        },
        "protocol": {
            **protocol,
            "fold_seed": fold_seed,
            "split_unit": "task_id",
            "task_family_unit": "instance_id",
            "final_holdout_task_count": len(partition.final_holdout_tasks),
            "final_holdout_evaluated": False,
            "final_holdout_turn_payloads_selected": False,
            "pilot_task_count": len(partition.pilot_tasks),
            "development_reserve_task_count": len(partition.development_reserve_tasks),
            "current_request_usage_features_included": False,
            "final_holdout_task_ids": list(partition.final_holdout_tasks),
            "pilot_task_ids": list(partition.pilot_tasks),
            "development_reserve_task_ids": list(partition.development_reserve_tasks),
        },
        "method": {
            "candidate_id": CANDIDATE_ID,
            "causal_state_features": list(CAUSAL_STATE_FEATURES),
            "call_start_reference": "ema",
            "task_update_reference": "call_clock",
            "call_start_additional_references": ["last_value", "history_gru"],
            "task_update_additional_references": ["linear_extrapolation", "history_gru"],
            "linear_extrapolation": (
                "Extrapolated remaining budget: observed per-call output rate times "
                "training-side total-call quantiles conditioned on totals above the cutoff"
            ),
            "history_gru": {**gru_config, "inputs": "per-call log1p osl/isl_new/isl + isl-missing flag"},
            "interval_calibrator": "task_max_log_conformal",
            "target_unit": "provider output Tokens",
            "scope": "count-only causal history; no current request content or checkpoints",
        },
        "cohort": _cohort_audit(tasks, points),
        "positions": position_results,
        "fold_reports": fold_reports,
        "dependencies": {
            "python_model": "LightGBM quantile",
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
    }

    output_dir = (output_override or root / config["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(
            all_records,
            key=lambda item: (item.target.value, item.candidate_id, item.point_id),
        ):
            handle.write(json.dumps(_prediction_document(record), ensure_ascii=False) + "\n")
    return result


def main() -> int:
    args = _parser().parse_args()
    result = run(
        args.config,
        data_override=args.data,
        output_override=args.output_dir,
        fold_seed_override=args.fold_seed,
    )
    summary = {
        position: {
            candidate: {
                "mae": metrics["mae"],
                "wape": metrics["wape"],
                "coverage": metrics["coverage"],
                "normalized_interval_width": metrics["normalized_interval_width"],
            }
            for candidate, metrics in values["metrics"].items()
        }
        for position, values in result["positions"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
