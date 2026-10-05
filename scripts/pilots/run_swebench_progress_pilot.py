from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from scripts.pilots.run_endpoint_evidence_pilot import _paired_bootstrap

FOLDS = 5
TOOL_CALL = re.compile(r'<tool_call name="bash">(.*?)</tool_call>', re.DOTALL)
RETURN_CODE = re.compile(r"<returncode>(-?\d+)</returncode>")
OUTPUT_BLOCK = re.compile(r"<output>(.*?)</output>", re.DOTALL)
EDIT_PATTERN = re.compile(
    r"str_replace|apply_patch|\bsed -i\b|>\s*/|\btee\b|<<\s*'?EOF|\bpatch\b|"
    r"\bcat\s*>|\bpython\s+-c\b.*\bwrite\b"
)
TEST_PATTERN = re.compile(r"\bpytest\b|python -m unittest|\btox\b|\bnosetests\b|runtests")
PATH_PATTERN = re.compile(r"(/[\w./-]+\.\w+)")
PR_DESCRIPTION = re.compile(r"<pr_description>(.*?)</pr_description>", re.DOTALL)
CODE_FENCE = re.compile(r"```")

HISTORY_FEATURE_NAMES = (
    "turn_index",
    "log_cumulative_output_tokens",
    "log_cumulative_input_tokens",
    "log_last_output_tokens",
    "log_mean_output_tokens",
    "output_token_trend",
    "cumulative_toolcalls",
    "log_last_input_tokens",
)
TASK_FEATURE_NAMES = (
    "log_problem_statement_chars",
    "log_problem_statement_words",
    "problem_code_fence_count",
    "problem_path_mention_count",
    "problem_traceback_count",
    "log_first_request_chars",
)
PROGRESS_FEATURE_NAMES = (
    "distinct_files_touched",
    "edit_command_fraction",
    "explore_command_fraction",
    "run_command_count",
    "git_diff_count",
    "zero_returncode_fraction",
    "recent_zero_returncode_fraction",
    "recent_command_novelty",
    "exact_repeat_command_count",
    "log_cumulative_tool_output_bytes",
    "turns_since_last_edit",
    "turns_since_last_zero_returncode",
    "distinct_command_head_count",


    "error_signal_recent",
    "error_signal_rate",
    "traceback_count",
    "success_signal_recent",
    "log_last_git_diff_bytes",
    "log_max_output_bytes",


    "explore_word_rate_recent",
    "edit_word_rate_recent",
    "verify_word_rate_recent",
    "complete_word_rate_recent",
    "complete_word_rate_last",
    "submit_mentioned_recent",
    "explore_to_complete_ratio",
    "log_prose_length_recent",
)

EXPLORE_WORDS = frozenset(
    """examine locate search understand relevant reproduce files find look
    investigate explore inspect identify review read""".split()
)
EDIT_WORDS = frozenset(
    """create modify change edit fix update add replace implement write insert
    apply remove delete""".split()
)
VERIFY_WORDS = frozenset(
    """verify test check confirm run validate ensure rerun reproduce""".split()
)
COMPLETE_WORDS = frozenset(
    """submit final complete completed done finished patch summary ready
    conclusion resolved""".split()
)
ERROR_SIGNAL = re.compile(
    r"Traceback|SyntaxError|\bError\b|error:|Exception|command not found|"
    r"No such file"
)
SUCCESS_SIGNAL = re.compile(r"\bpassed\b|\bOK\b|\bSUCCESS\b|\ball tests pass")
WORD = re.compile(r"[a-z]{3,}")
EXPLORE_HEADS = frozenset("grep cat find ls head tail wc awk".split())
EDIT_HEADS = frozenset("cp echo mkdir mv rm touch".split())
RUN_HEADS = frozenset("python python3 pytest".split())


@dataclass(frozen=True)
class SwePoint:
    point_id: str
    task_id: str
    trajectory_id: str
    fold: int
    update_index: int
    truth: float
    sample_weight: float
    horizon: float
    past_rate: float
    task_features: tuple[float, ...]
    future_costs: tuple[float, ...]
    history_features: tuple[float, ...]
    progress_features: tuple[float, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SWE-bench causal progress pilot for Task-update"
    )
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser


def _commands(raw_response: str) -> list[str]:
    found: list[str] = []
    for blob in TOOL_CALL.findall(raw_response or ""):
        try:
            payload = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, Mapping):
            command = payload.get("command")
            if isinstance(command, str) and command.strip():
                found.append(command.strip())
    return found


def _log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _progress_block(
    turn_commands: Sequence[str],
    return_codes: Sequence[int | None],
    outputs: Mapping[str, np.ndarray],
    prose: Mapping[str, np.ndarray],
) -> tuple[float, ...]:

    cutoff = len(turn_commands)
    files: set[str] = set()
    heads: set[str] = set()
    first_seen: dict[str, int] = {}
    repeats = 0
    edit_turns: list[int] = []
    explore_count = 0
    run_count = 0
    git_diff_turns: list[int] = []
    for turn, command in enumerate(turn_commands):
        if not command:
            continue
        files.update(PATH_PATTERN.findall(command))
        tokens = command.split()
        head = tokens[0] if tokens else ""
        heads.add(head)
        if command in first_seen:
            repeats += 1
        else:
            first_seen[command] = turn
        if EDIT_PATTERN.search(command) or head in EDIT_HEADS:
            edit_turns.append(turn)
        if head in EXPLORE_HEADS or " -n " in command:
            explore_count += 1
        if head in RUN_HEADS:
            run_count += 1
        if "git diff" in command:
            git_diff_turns.append(turn)

    zero_flags = [1.0 if code == 0 else 0.0 for code in return_codes if code is not None]
    recent_zero = zero_flags[-5:]
    window = [
        (turn, command)
        for turn, command in enumerate(turn_commands)
        if command and turn >= cutoff - 5
    ]
    novel = sum(1.0 for turn, command in window if first_seen.get(command) == turn)
    zero_turns = [turn for turn, code in enumerate(return_codes) if code == 0]
    issued = sum(1 for command in turn_commands if command)


    error_recent = float(outputs["error"][-3:].sum())
    error_total = float(outputs["error"].sum())
    traceback_total = float(outputs["traceback"].sum())
    success_recent = float(outputs["success"][-3:].sum())
    last_diff_bytes = (
        float(outputs["bytes"][git_diff_turns[-1]]) if git_diff_turns else 0.0
    )
    total_output_bytes = float(outputs["bytes"].sum())
    max_output_len = float(outputs["length"].max()) if cutoff else 0.0

    recent_words = float(prose["words"][-3:].sum())
    last_words = float(prose["words"][-1]) if cutoff else 0.0
    explore_total = float(prose["explore"].sum())
    complete_total = float(prose["complete"].sum())

    return (
        float(len(files)),
        float(len(edit_turns)) / max(issued, 1),
        float(explore_count) / max(issued, 1),
        float(run_count),
        float(len(git_diff_turns)),
        float(np.mean(zero_flags)) if zero_flags else 0.0,
        float(np.mean(recent_zero)) if recent_zero else 0.0,
        novel / max(len(window), 1),
        float(repeats),
        _log1p(total_output_bytes),
        float(cutoff - 1 - edit_turns[-1]) if edit_turns else float(cutoff),
        float(cutoff - 1 - zero_turns[-1]) if zero_turns else float(cutoff),
        float(len(heads)),
        error_recent,
        error_total / max(cutoff, 1),
        traceback_total,
        success_recent,
        _log1p(last_diff_bytes),
        _log1p(max_output_len),
        float(prose["explore"][-3:].sum()) / max(recent_words, 1.0),
        float(prose["edit"][-3:].sum()) / max(recent_words, 1.0),
        float(prose["verify"][-3:].sum()) / max(recent_words, 1.0),
        float(prose["complete"][-3:].sum()) / max(recent_words, 1.0),
        float(prose["complete"][-1]) / max(last_words, 1.0) if cutoff else 0.0,
        1.0 if float(prose["submit"][-3:].sum()) > 0.0 else 0.0,
        explore_total / max(complete_total, 1.0),
        _log1p(recent_words),
    )


def _build_points(dialogues_path: Path, split_seed: int) -> tuple[SwePoint, ...]:
    payload = json.loads(dialogues_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("dialogue source must be a non-empty list")

    task_ids = sorted({str(record["instance_id"]) for record in payload})
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{task_id}:{split_seed}".encode("utf-8")
        ).hexdigest(),
    )
    fold_of = {task_id: index % FOLDS for index, task_id in enumerate(ranked)}

    points: list[SwePoint] = []
    for record_index, record in enumerate(payload):
        turns = record.get("turns") or []
        if len(turns) < 2:
            continue
        task_id = str(record["instance_id"])
        trajectory_id = f"swebench:{task_id}:{record_index}"
        outputs = [float(turn.get("api_output_tokens") or 0.0) for turn in turns]
        inputs = [float(turn.get("api_input_tokens") or 0.0) for turn in turns]
        toolcalls = [float(turn.get("toolcalls_used") or 0.0) for turn in turns]
        commands_by_turn = [
            " ; ".join(_commands(turn.get("raw_response") or "")) for turn in turns
        ]


        codes_by_turn: list[int | None] = []
        out_stats: dict[str, list[float]] = {
            key: [] for key in ("error", "traceback", "success", "bytes", "length")
        }
        prose_stats: dict[str, list[float]] = {
            key: []
            for key in ("words", "explore", "edit", "verify", "complete", "submit")
        }
        for index, turn in enumerate(turns):
            follower = turns[index + 1] if index + 1 < len(turns) else None
            prompt = (follower or {}).get("user_prompt") or ""
            match = RETURN_CODE.search(prompt)
            codes_by_turn.append(int(match.group(1)) if match else None)
            text = " ".join(OUTPUT_BLOCK.findall(prompt))
            out_stats["error"].append(float(len(ERROR_SIGNAL.findall(text))))
            out_stats["traceback"].append(float(text.count("Traceback")))
            out_stats["success"].append(float(len(SUCCESS_SIGNAL.findall(text))))
            out_stats["bytes"].append(float(len(text.encode("utf-8"))))
            out_stats["length"].append(float(len(text)))

            prose = TOOL_CALL.sub(" ", turn.get("raw_response") or "").lower()
            words = WORD.findall(prose)
            prose_stats["words"].append(float(len(words)))
            prose_stats["explore"].append(
                float(sum(1 for word in words if word in EXPLORE_WORDS))
            )
            prose_stats["edit"].append(
                float(sum(1 for word in words if word in EDIT_WORDS))
            )
            prose_stats["verify"].append(
                float(sum(1 for word in words if word in VERIFY_WORDS))
            )
            prose_stats["complete"].append(
                float(sum(1 for word in words if word in COMPLETE_WORDS))
            )
            prose_stats["submit"].append(1.0 if "submit" in prose else 0.0)
        out_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in out_stats.items()}
        prose_arrays = {
            k: np.asarray(v, dtype=np.float64) for k, v in prose_stats.items()
        }

        first_prompt = str(turns[0].get("user_prompt") or "")
        described = PR_DESCRIPTION.findall(first_prompt)
        description = described[0] if described else first_prompt
        task_features = (
            _log1p(float(len(description))),
            _log1p(float(len(WORD.findall(description.lower())))),
            float(len(CODE_FENCE.findall(description))),
            float(len(PATH_PATTERN.findall(description))),
            float(description.count("Traceback")),
            _log1p(float(len(first_prompt))),
        )

        usable = len(turns) - 1
        for cutoff in range(1, len(turns)):
            remaining = float(sum(outputs[cutoff:]))
            done_outputs = outputs[:cutoff]
            history = (
                float(cutoff),
                _log1p(sum(done_outputs)),
                _log1p(sum(inputs[:cutoff])),
                _log1p(done_outputs[-1]),
                _log1p(float(np.mean(done_outputs))),
                done_outputs[-1] / max(float(np.mean(done_outputs)), 1.0),
                float(sum(toolcalls[:cutoff])),
                _log1p(inputs[cutoff - 1]),
            )
            progress = _progress_block(
                commands_by_turn[:cutoff],
                codes_by_turn[:cutoff],
                {key: value[:cutoff] for key, value in out_arrays.items()},
                {key: value[:cutoff] for key, value in prose_arrays.items()},
            )
            points.append(
                SwePoint(
                    point_id=f"{trajectory_id}:update:{cutoff}",
                    task_id=task_id,
                    trajectory_id=trajectory_id,
                    fold=fold_of[task_id],
                    update_index=cutoff,
                    truth=remaining,
                    sample_weight=1.0 / usable,
                    horizon=float(len(turns) - cutoff),
                    past_rate=float(np.mean(done_outputs)),
                    task_features=task_features,
                    future_costs=tuple(outputs[cutoff:]),
                    history_features=history,
                    progress_features=progress,
                )
            )
    return tuple(points)


def _matrix(
    points: Sequence[SwePoint], *, with_progress: bool, with_task: bool = False
) -> np.ndarray:
    rows = []
    for point in points:
        row = list(point.history_features)
        if with_progress:
            row.extend(point.progress_features)
        if with_task:
            row.extend(point.task_features)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _shuffled_matrix(
    points: Sequence[SwePoint], seed: int
) -> np.ndarray:

    matrix = _matrix(points, with_progress=True)
    width = len(HISTORY_FEATURE_NAMES)
    generator = np.random.default_rng(seed)
    strata: dict[int, list[int]] = {}
    for index, point in enumerate(points):
        strata.setdefault(min(point.update_index, 20), []).append(index)
    for indices in strata.values():
        order = generator.permutation(len(indices))
        source = [indices[position] for position in order]
        matrix[np.asarray(indices), width:] = matrix[np.asarray(source), width:]
    return matrix


def _partition(
    points: Sequence[SwePoint], test_fold: int
) -> tuple[list[SwePoint], list[SwePoint], list[SwePoint]]:
    validation_fold = (test_fold + 1) % FOLDS
    train = [p for p in points if p.fold not in {test_fold, validation_fold}]
    validation = [p for p in points if p.fold == validation_fold]
    test = [p for p in points if p.fold == test_fold]
    if not all((train, validation, test)):
        raise ValueError(f"fold {test_fold} has an empty partition")
    return train, validation, test


def _fit(
    train_matrix: np.ndarray,
    train: Sequence[SwePoint],
    validation_matrix: np.ndarray,
    validation: Sequence[SwePoint],
    *,
    feature_names: Sequence[str],
    seed: int,
    log_target: bool = False,
) -> lgb.Booster:


    def labels(points: Sequence[SwePoint]) -> np.ndarray:
        values = np.asarray([p.truth for p in points], dtype=np.float64)
        return np.log1p(values) if log_target else values

    train_set = lgb.Dataset(
        train_matrix,
        label=labels(train),
        weight=np.asarray([p.sample_weight for p in train]),
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=labels(validation),
        weight=np.asarray([p.sample_weight for p in validation]),
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 20,
            "lambda_l2": 5.0,
            "seed": seed,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=600,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_on_labels(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    validation_weights: np.ndarray,
    *,
    feature_names: Sequence[str],
    seed: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train_matrix,
        label=train_labels,
        weight=train_weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        weight=validation_weights,
        reference=train_set,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 20,
            "lambda_l2": 5.0,
            "seed": seed,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=600,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _offset_grid(max_horizon: int) -> tuple[tuple[int, int, int], ...]:

    grid: list[tuple[int, int, int]] = [(j, j, j) for j in range(1, 16)]
    low = 16
    width = 2
    while low <= max_horizon:
        high = min(low + width - 1, max_horizon)
        grid.append(((low + high) // 2, low, high))
        low = high + 1
        width = min(width * 2, 32)
    return tuple(grid)


def _expand_grid(
    points: Sequence[SwePoint],
    matrix: np.ndarray,
    grid: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    survive: list[float] = []
    cost: list[float] = []
    weight: list[float] = []
    for index, point in enumerate(points):
        horizon = int(point.horizon)
        for representative, low, high in grid:
            rows.append(np.concatenate((matrix[index], [float(representative)])))


            survive.append(1.0 if horizon >= representative else 0.0)
            segment = point.future_costs[low - 1 : high]
            cost.append(float(np.mean(segment)) if segment else 0.0)
            weight.append(point.sample_weight)
    mask = np.asarray(survive) > 0.0
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(survive, dtype=np.float64),
        np.asarray(cost, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
        mask,
    )


def _fit_objective(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    objective: str,
    feature_names: Sequence[str],
    seed: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train[0], label=train[1], weight=train[2],
        feature_name=list(feature_names), free_raw_data=False,
    )
    validation_set = lgb.Dataset(
        validation[0], label=validation[1], weight=validation[2],
        reference=train_set, feature_name=list(feature_names), free_raw_data=False,
    )
    metric = "binary_logloss" if objective == "binary" else "l2"
    return lgb.train(
        {
            "objective": objective,
            "metric": metric,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 40,
            "lambda_l2": 5.0,
            "seed": seed,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=400,
        valid_sets=[validation_set],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )


def _curve(
    model: lgb.Booster,
    matrix: np.ndarray,
    grid: Sequence[tuple[int, int, int]],
    *,
    clip_unit: bool,
) -> np.ndarray:
    count = matrix.shape[0]
    values = np.empty((count, len(grid)), dtype=np.float64)
    for column, (representative, _low, _high) in enumerate(grid):
        augmented = np.concatenate(
            (matrix, np.full((count, 1), float(representative))), axis=1
        )
        values[:, column] = model.predict(
            augmented, num_iteration=model.best_iteration
        )
    if clip_unit:
        return np.minimum.accumulate(np.clip(values, 0.0, 1.0), axis=1)
    return np.maximum(values, 0.0)


def _mae(points: Sequence[SwePoint], predictions: Mapping[str, float]) -> float:
    numerator = sum(
        p.sample_weight * abs(predictions[p.point_id] - p.truth) for p in points
    )
    return numerator / sum(p.sample_weight for p in points)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dialogues = args.dialogues.resolve()
    if not dialogues.is_file():
        raise FileNotFoundError(dialogues)

    points = _build_points(dialogues, args.split_seed)
    arms = ("history", "progress", "shuffled", "progress_log", "history_log")
    composed = (
        "horizon_past_rate",
        "horizon_rate",
        "survival_flat",
        "survival_profile",
        "horizon_rate_mean",
        "horizon_rate_log",
        "blend",
        "task_progress",
        "task_horizon_rate",
    )
    all_arms = (*arms, *composed)
    predictions: dict[str, dict[str, float]] = {arm: {} for arm in all_arms}
    fold_rows: list[dict[str, object]] = []
    latency: dict[str, list[float]] = {arm: [] for arm in arms}

    for test_fold in range(FOLDS):
        train, validation, test = _partition(points, test_fold)
        fold_seed = args.seed + 100 * test_fold
        full_names = (*HISTORY_FEATURE_NAMES, *PROGRESS_FEATURE_NAMES)
        task_names = (*full_names, *TASK_FEATURE_NAMES)
        task_matrices = (
            _matrix(train, with_progress=True, with_task=True),
            _matrix(validation, with_progress=True, with_task=True),
            _matrix(test, with_progress=True, with_task=True),
        )
        history_matrices = (
            _matrix(train, with_progress=False),
            _matrix(validation, with_progress=False),
            _matrix(test, with_progress=False),
        )
        progress_matrices = (
            _matrix(train, with_progress=True),
            _matrix(validation, with_progress=True),
            _matrix(test, with_progress=True),
        )
        built = {
            "history": (*history_matrices, HISTORY_FEATURE_NAMES, False),
            "progress": (*progress_matrices, full_names, False),
            "shuffled": (
                _shuffled_matrix(train, fold_seed + 11),
                progress_matrices[1],
                progress_matrices[2],
                full_names,
                False,
            ),
            "progress_log": (*progress_matrices, full_names, True),
            "history_log": (*history_matrices, HISTORY_FEATURE_NAMES, True),
        }
        row: dict[str, object] = {"test_fold": test_fold}
        for arm in arms:
            train_matrix, validation_matrix, test_matrix, names, log_target = built[arm]
            model = _fit(
                train_matrix,
                train,
                validation_matrix,
                validation,
                feature_names=names,
                seed=fold_seed + 3,
                log_target=log_target,
            )
            start = time.perf_counter()
            raw = model.predict(test_matrix, num_iteration=model.best_iteration)
            scored = np.maximum(np.expm1(raw) if log_target else raw, 0.0)
            latency[arm].append(1000.0 * (time.perf_counter() - start) / len(test))
            fold_predictions = {
                p.point_id: float(value) for p, value in zip(test, scored)
            }
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)


        weights_train = np.asarray([p.sample_weight for p in train])
        weights_validation = np.asarray([p.sample_weight for p in validation])
        horizon_model = _fit_on_labels(
            progress_matrices[0],
            np.asarray([p.horizon for p in train]),
            weights_train,
            progress_matrices[1],
            np.asarray([p.horizon for p in validation]),
            weights_validation,
            feature_names=full_names,
            seed=fold_seed + 21,
        )
        rate_model = _fit_on_labels(
            progress_matrices[0],
            np.asarray([p.truth / max(p.horizon, 1.0) for p in train]),
            weights_train,
            progress_matrices[1],
            np.asarray([p.truth / max(p.horizon, 1.0) for p in validation]),
            weights_validation,
            feature_names=full_names,
            seed=fold_seed + 22,
        )
        horizon_hat = np.maximum(
            horizon_model.predict(
                progress_matrices[2], num_iteration=horizon_model.best_iteration
            ),
            0.0,
        )
        rate_hat = np.maximum(
            rate_model.predict(
                progress_matrices[2], num_iteration=rate_model.best_iteration
            ),
            0.0,
        )


        horizon_mean_model = _fit_objective(
            (progress_matrices[0], np.asarray([p.horizon for p in train]), weights_train),
            (
                progress_matrices[1],
                np.asarray([p.horizon for p in validation]),
                weights_validation,
            ),
            objective="regression",
            feature_names=full_names,
            seed=fold_seed + 41,
        )
        rate_mean_model = _fit_objective(
            (
                progress_matrices[0],
                np.asarray([p.truth / max(p.horizon, 1.0) for p in train]),
                weights_train,
            ),
            (
                progress_matrices[1],
                np.asarray([p.truth / max(p.horizon, 1.0) for p in validation]),
                weights_validation,
            ),
            objective="regression",
            feature_names=full_names,
            seed=fold_seed + 42,
        )
        horizon_log_model = _fit_on_labels(
            progress_matrices[0],
            np.log1p(np.asarray([p.horizon for p in train])),
            weights_train,
            progress_matrices[1],
            np.log1p(np.asarray([p.horizon for p in validation])),
            weights_validation,
            feature_names=full_names,
            seed=fold_seed + 43,
        )
        horizon_mean_hat = np.maximum(
            horizon_mean_model.predict(
                progress_matrices[2], num_iteration=horizon_mean_model.best_iteration
            ),
            0.0,
        )
        rate_mean_hat = np.maximum(
            rate_mean_model.predict(
                progress_matrices[2], num_iteration=rate_mean_model.best_iteration
            ),
            0.0,
        )
        horizon_log_hat = np.maximum(
            np.expm1(
                horizon_log_model.predict(
                    progress_matrices[2],
                    num_iteration=horizon_log_model.best_iteration,
                )
            ),
            0.0,
        )
        past_rate = np.asarray([p.past_rate for p in test])
        for arm, scored in (
            ("horizon_past_rate", horizon_hat * past_rate),
            ("horizon_rate", horizon_hat * rate_hat),
            ("horizon_rate_mean", horizon_mean_hat * rate_mean_hat),
            ("horizon_rate_log", horizon_log_hat * rate_mean_hat),
        ):
            fold_predictions = {
                p.point_id: float(value) for p, value in zip(test, scored)
            }
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)


        grid = _offset_grid(max(int(p.horizon) for p in train))
        train_rows, train_alive, train_cost, train_weight, train_mask = _expand_grid(
            train, progress_matrices[0], grid
        )
        val_rows, val_alive, val_cost, val_weight, val_mask = _expand_grid(
            validation, progress_matrices[1], grid
        )
        grid_names = (*full_names, "future_call_offset")
        survival_model = _fit_objective(
            (train_rows, train_alive, train_weight),
            (val_rows, val_alive, val_weight),
            objective="binary",
            feature_names=grid_names,
            seed=fold_seed + 31,
        )
        profile_model = _fit_objective(
            (train_rows[train_mask], train_cost[train_mask], train_weight[train_mask]),
            (val_rows[val_mask], val_cost[val_mask], val_weight[val_mask]),
            objective="regression",
            feature_names=grid_names,
            seed=fold_seed + 32,
        )
        flat_model = _fit_objective(
            (
                train_rows[train_mask][:, :-1],
                train_cost[train_mask],
                train_weight[train_mask],
            ),
            (val_rows[val_mask][:, :-1], val_cost[val_mask], val_weight[val_mask]),
            objective="regression",
            feature_names=full_names,
            seed=fold_seed + 33,
        )
        widths = np.asarray([high - low + 1 for _rep, low, high in grid], dtype=np.float64)
        survival = _curve(survival_model, progress_matrices[2], grid, clip_unit=True)
        profile = _curve(profile_model, progress_matrices[2], grid, clip_unit=False)
        flat = np.maximum(
            flat_model.predict(
                progress_matrices[2], num_iteration=flat_model.best_iteration
            ),
            0.0,
        )
        for arm, scored in (
            ("survival_flat", (survival * widths).sum(axis=1) * flat),
            ("survival_profile", (survival * profile * widths).sum(axis=1)),
        ):
            fold_predictions = {
                p.point_id: float(value) for p, value in zip(test, np.maximum(scored, 0.0))
            }
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)


        validation_truth = np.asarray([p.truth for p in validation])
        validation_weight = weights_validation
        progress_validation = np.maximum(
            _fit(
                progress_matrices[0],
                train,
                progress_matrices[1],
                validation,
                feature_names=full_names,
                seed=fold_seed + 3,
            ).predict(progress_matrices[1]),
            0.0,
        )
        horizon_validation = np.maximum(
            horizon_model.predict(
                progress_matrices[1], num_iteration=horizon_model.best_iteration
            ),
            0.0,
        ) * np.maximum(
            rate_model.predict(
                progress_matrices[1], num_iteration=rate_model.best_iteration
            ),
            0.0,
        )
        best_weight, best_score = 0.0, float("inf")
        for candidate in np.linspace(0.0, 1.0, 21):
            mixed = candidate * horizon_validation + (1.0 - candidate) * progress_validation
            score = float(
                np.average(np.abs(mixed - validation_truth), weights=validation_weight)
            )
            if score < best_score:
                best_weight, best_score = float(candidate), score
        row["blend_weight"] = best_weight
        blended = best_weight * (horizon_hat * rate_hat) + (1.0 - best_weight) * np.asarray(
            [predictions["progress"][p.point_id] for p in test]
        )
        fold_predictions = {
            p.point_id: float(value) for p, value in zip(test, np.maximum(blended, 0.0))
        }
        predictions["blend"].update(fold_predictions)
        row["blend_mae"] = _mae(test, fold_predictions)


        task_direct = np.maximum(
            _fit(
                task_matrices[0], train, task_matrices[1], validation,
                feature_names=task_names, seed=fold_seed + 51,
            ).predict(task_matrices[2]),
            0.0,
        )
        task_horizon = np.maximum(
            _fit_on_labels(
                task_matrices[0],
                np.asarray([p.horizon for p in train]),
                weights_train,
                task_matrices[1],
                np.asarray([p.horizon for p in validation]),
                weights_validation,
                feature_names=task_names,
                seed=fold_seed + 52,
            ).predict(task_matrices[2]),
            0.0,
        )
        task_rate = np.maximum(
            _fit_on_labels(
                task_matrices[0],
                np.asarray([p.truth / max(p.horizon, 1.0) for p in train]),
                weights_train,
                task_matrices[1],
                np.asarray([p.truth / max(p.horizon, 1.0) for p in validation]),
                weights_validation,
                feature_names=task_names,
                seed=fold_seed + 53,
            ).predict(task_matrices[2]),
            0.0,
        )
        for arm, scored in (
            ("task_progress", task_direct),
            ("task_horizon_rate", task_horizon * task_rate),
        ):
            fold_predictions = {
                p.point_id: float(value) for p, value in zip(test, np.maximum(scored, 0.0))
            }
            predictions[arm].update(fold_predictions)
            row[f"{arm}_mae"] = _mae(test, fold_predictions)

        row["horizon_mae_calls"] = float(
            np.average(
                np.abs(horizon_hat - np.asarray([p.horizon for p in test])),
                weights=np.asarray([p.sample_weight for p in test]),
            )
        )
        fold_rows.append(row)
        print(
            f"fold {test_fold}: "
            + "  ".join(f"{arm}={row[f'{arm}_mae']:.1f}" for arm in arms),
            flush=True,
        )

    results: dict[str, object] = {
        "dialogues": str(dialogues),
        "split_seed": args.split_seed,
        "seed": args.seed,
        "n_points": len(points),
        "n_tasks": len({p.task_id for p in points}),
        "median_horizon": float(
            np.median([p.truth for p in points if p.update_index == 1])
        ),
        "folds": fold_rows,
        "arms": {
            arm: {
                "mae": _mae(points, predictions[arm]),
                "latency_ms_per_point": float(np.mean(latency[arm])),
            }
            for arm in arms
        },
    }
    for arm in composed:
        results["arms"][arm] = {"mae": _mae(points, predictions[arm])}
    for label, candidate, reference in (
        ("progress_vs_history", "progress", "history"),
        ("progress_vs_shuffled", "progress", "shuffled"),
        ("progress_log_vs_history_log", "progress_log", "history_log"),
        ("progress_log_vs_progress", "progress_log", "progress"),
        ("history_log_vs_history", "history_log", "history"),
        ("horizon_past_rate_vs_history", "horizon_past_rate", "history"),
        ("horizon_rate_vs_history", "horizon_rate", "history"),
        ("horizon_rate_vs_progress", "horizon_rate", "progress"),
        ("survival_profile_vs_history", "survival_profile", "history"),
        ("survival_profile_vs_horizon_rate", "survival_profile", "horizon_rate"),
        ("survival_profile_vs_survival_flat", "survival_profile", "survival_flat"),
        ("survival_flat_vs_horizon_rate", "survival_flat", "horizon_rate"),
        ("horizon_rate_mean_vs_history", "horizon_rate_mean", "history"),
        ("horizon_rate_mean_vs_horizon_rate", "horizon_rate_mean", "horizon_rate"),
        ("horizon_rate_log_vs_history", "horizon_rate_log", "history"),
        ("blend_vs_history", "blend", "history"),
        ("blend_vs_horizon_rate", "blend", "horizon_rate"),
        ("task_horizon_rate_vs_history", "task_horizon_rate", "history"),
        ("task_horizon_rate_vs_horizon_rate", "task_horizon_rate", "horizon_rate"),
        ("task_progress_vs_progress", "task_progress", "progress"),
    ):
        results[label] = _paired_bootstrap(
            points,
            predictions[candidate],
            predictions[reference],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )


    buckets = {
        "update_1_2": lambda p: p.update_index <= 2,
        "update_3_5": lambda p: 3 <= p.update_index <= 5,
        "update_6_15": lambda p: 6 <= p.update_index <= 15,
        "update_16_plus": lambda p: p.update_index >= 16,
    }
    stages: dict[str, dict[str, float]] = {}
    for name, keep in buckets.items():
        subset = [p for p in points if keep(p)]
        if not subset:
            continue
        stages[name] = {"n_points": float(len(subset))}
        for arm in all_arms:
            stages[name][arm] = _mae(subset, predictions[arm])
    results["stages"] = stages
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(results["arms"], indent=2))
    print("progress vs history :", json.dumps(results["progress_vs_history"]))
    print("progress vs shuffled:", json.dumps(results["progress_vs_shuffled"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
