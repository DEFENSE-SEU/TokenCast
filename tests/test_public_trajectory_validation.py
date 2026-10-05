from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.experiments import run_public_trajectory_validation as pilot
from scripts.experiments import run_public_task_initialization_refinement as refinement
from scripts.experiments import run_public_task_initialization_validation as task_start
from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import ScoredForecast


def _turn(output: int, new_input: int = 10, input_tokens: int | None = 20) -> str:
    value = {"osl": output, "isl_new": new_input}
    if input_tokens is not None:
        value["isl"] = input_tokens
    return json.dumps(value)


def test_partition_seals_final_holdout_before_pilot_selection() -> None:
    task_ids = [f"task-{index:03d}" for index in range(20)]

    first = pilot._partition_task_ids(
        task_ids,
        seed=7,
        final_holdout_fraction=0.20,
        pilot_tasks=10,
    )
    second = pilot._partition_task_ids(
        tuple(reversed(task_ids)),
        seed=7,
        final_holdout_fraction=0.20,
        pilot_tasks=10,
    )

    assert first == second
    assert len(first.final_holdout_tasks) == 4
    assert len(first.pilot_tasks) == 10
    assert len(first.development_reserve_tasks) == 6
    assert not (set(first.final_holdout_tasks) & set(first.pilot_tasks))


def test_loader_decodes_only_selected_turn_payloads(tmp_path) -> None:
    path = tmp_path / "source.parquet"
    table = pa.table(
        {
            "instance_id": ["selected", "sealed"],
            "dataset": ["source", "source"],
            "model": ["model", "model"],
            "turns": [[_turn(5)], ["invalid-json"]],
        }
    )
    pq.write_table(table, path)

    tasks = pilot._load_selected_tasks(path, frozenset({"selected"}))

    assert tasks == (
        pilot.TraceTask(
            task_id="selected",
            dataset="source",
            model="model",
            turns=(pilot.TraceTurn(5, 10, 20),),
        ),
    )


def test_call_features_depend_only_on_completed_turns() -> None:
    shared = pilot.TraceTurn(10, 20, 30)
    first = pilot.TraceTask(
        task_id="task",
        dataset="source",
        model="model",
        turns=(shared, pilot.TraceTurn(40, 50, 60), pilot.TraceTurn(70, 80, 90)),
    )
    changed_suffix = pilot.TraceTask(
        task_id="task",
        dataset="source",
        model="model",
        turns=(shared, pilot.TraceTurn(400, 500, 600), pilot.TraceTurn(700, 800, 900)),
    )
    fold = {"task": 0}

    first_points = {point.point_id: point for point in pilot._build_points((first,), fold)}
    changed_points = {
        point.point_id: point for point in pilot._build_points((changed_suffix,), fold)
    }

    call_id = "task:call-start:1"
    update_id = "task:task-update:1"
    assert first_points[call_id].features == changed_points[call_id].features
    assert first_points[update_id].features == changed_points[update_id].features
    assert first_points[call_id].target_value != changed_points[call_id].target_value
    assert first_points[update_id].target_value != changed_points[update_id].target_value


def test_each_task_has_unit_weight_at_each_available_position() -> None:
    tasks = (
        pilot.TraceTask(
            task_id="a",
            dataset="source",
            model="model",
            turns=(pilot.TraceTurn(2, 3, None), pilot.TraceTurn(5, 7, 11)),
        ),
        pilot.TraceTask(
            task_id="b",
            dataset="source",
            model="model",
            turns=(
                pilot.TraceTurn(13, 17, 19),
                pilot.TraceTurn(23, 29, 31),
                pilot.TraceTurn(37, 41, 43),
            ),
        ),
    )

    points = pilot._build_points(tasks, {"a": 0, "b": 1})

    for task_id in ("a", "b"):
        for position in (pilot.CALL_START, pilot.TASK_UPDATE):
            weight = sum(
                point.sample_weight
                for point in points
                if point.task_id == task_id and point.position == position
            )
            assert weight == pytest.approx(1.0)


def test_call_clock_caps_sparse_tail_into_last_supported_bucket() -> None:
    points = []
    for task_index in range(3):
        task = pilot.TraceTask(
            task_id=f"task-{task_index}",
            dataset="source",
            model="model",
            turns=tuple(pilot.TraceTurn(10 + index, 5, 8) for index in range(5)),
        )
        points.extend(
            point
            for point in pilot._build_points((task,), {task.task_id: task_index})
            if point.position == pilot.TASK_UPDATE
        )

    model = pilot._fit_call_clock(points, min_bucket_tasks=3)

    assert model.max_bucket == 4
    assert set(model.intervals) == {1, 2, 3, 4}


def test_last_value_centers_on_previous_call_output() -> None:
    predictor = pilot.LogResidualPredictor(
        fallback=7.0,
        residual_interval=(0.0, 0.0, 0.0),
        center_feature="log_last_output_tokens",
    )
    task = pilot.TraceTask(
        task_id="task",
        dataset="source",
        model="model",
        turns=(pilot.TraceTurn(11, 1, 2), pilot.TraceTurn(29, 3, 4)),
    )
    points = [
        point
        for point in pilot._build_points((task,), {"task": 0})
        if point.position == pilot.CALL_START
    ]

    values = predictor.predict(sorted(points, key=lambda point: point.call_index))

    assert values[0][1] == pytest.approx(7.0)
    assert values[1][1] == pytest.approx(11.0)


def test_linear_extrapolation_scales_remaining_calls_by_observed_rate() -> None:
    import numpy as np

    tasks = tuple(
        pilot.TraceTask(
            task_id=f"task-{total}",
            dataset="source",
            model="model",
            turns=tuple(pilot.TraceTurn(10, 5, 8) for _ in range(total)),
        )
        for total in (3, 4, 5)
    )
    train_points = [
        point
        for task in tasks
        for point in pilot._build_points((task,), {task.task_id: 0})
        if point.position == pilot.TASK_UPDATE
    ]

    predictor = pilot._fit_linear_extrapolation(train_points)
    assert predictor.total_calls == (3, 4, 5)

    query = next(
        point
        for point in train_points
        if point.task_id == "task-5" and point.call_index == 2
    )
    values = predictor.predict([query])
    expected_remaining = np.quantile(np.asarray([3.0, 4.0, 5.0]), (0.05, 0.5, 0.95)) - 2.0
    assert values[0] == pytest.approx(10.0 * expected_remaining)

    longer_task = pilot.TraceTask(
        task_id="task-6",
        dataset="source",
        model="model",
        turns=tuple(pilot.TraceTurn(10, 5, 8) for _ in range(6)),
    )
    beyond = next(
        point
        for point in pilot._build_points((longer_task,), {"task-6": 0})
        if point.position == pilot.TASK_UPDATE and point.call_index == 5
    )
    assert predictor.predict([beyond])[0] == pytest.approx((0.0, 0.0, 0.0))


def test_history_gru_is_causal_and_seed_deterministic() -> None:
    shared = pilot.TraceTurn(10, 20, 30)
    tasks = tuple(
        pilot.TraceTask(
            task_id=task_id,
            dataset="source",
            model="model",
            turns=(shared, pilot.TraceTurn(osl, 2, 3), pilot.TraceTurn(osl * 2, 4, 5)),
        )
        for task_id, osl in (("a", 40), ("b", 400), ("c", 4), ("d", 44))
    )
    fold_by_task = {"a": 0, "b": 0, "c": 1, "d": 1}
    points = [
        point
        for point in pilot._build_points(tasks, fold_by_task)
        if point.position == pilot.CALL_START
    ]
    sequences = {task.task_id: task.turns for task in tasks}
    train = [point for point in points if point.fold == 0]
    validation = [point for point in points if point.fold == 1]
    gru_config = {
        "hidden_dim": 4,
        "learning_rate": 0.01,
        "max_epochs": 2,
        "patience": 1,
        "threads": 1,
    }

    def fit():
        return pilot._fit_history_gru(
            train,
            validation,
            sequences=sequences,
            gru_config=gru_config,
            seed=11,
        )

    first, second = fit(), fit()
    cutoff_one = sorted(
        (point for point in points if point.call_index == 1),
        key=lambda point: point.task_id,
    )
    first_values = first.predict(cutoff_one)
    assert first_values == pytest.approx(second.predict(cutoff_one))

    for row in first_values[1:]:
        assert row == pytest.approx(first_values[0])


def test_task_start_loader_reads_only_selected_causal_metadata(tmp_path) -> None:
    path = tmp_path / "metadata.parquet"
    table = pa.table(
        {
            "instance_id": ["selected", "sealed"],
            "repo": ["org/repo", "private/repo"],
            "repo_language": ["Python", ""],
            "problem_statement": ["Fix the parser.", ""],
            "patch": ["future-label", "sealed-future-label"],
        }
    )
    pq.write_table(table, path)

    metadata = task_start._load_selected_metadata(path, frozenset({"selected"}))

    assert metadata == {
        "selected": task_start.TaskMetadata(
            task_id="selected",
            repo="org/repo",
            repo_language="Python",
            problem_statement="Fix the parser.",
        )
    }


def test_task_start_example_uses_complete_output_total() -> None:
    task = pilot.TraceTask(
        task_id="task",
        dataset="source",
        model="model",
        turns=(pilot.TraceTurn(10, 4, 5), pilot.TraceTurn(30, 6, 7)),
    )
    metadata = task_start.TaskMetadata(
        task_id="task",
        repo="org/repo",
        repo_language="Python",
        problem_statement="Fix the parser.",
    )

    example = task_start._build_examples((task,), {"task": metadata}, {"task": 3})[0]

    assert example.total_output_tokens == 40
    assert example.call_count == 2
    assert example.mean_output_tokens == 20
    assert example.call_count * example.mean_output_tokens == example.total_output_tokens
    assert example.fold == 3


def test_task_start_mae_scale_minimizes_weighted_absolute_error() -> None:
    metadata = task_start.TaskMetadata(
        task_id="unused",
        repo="org/repo",
        repo_language="Python",
        problem_statement="Fix the parser.",
    )
    examples = tuple(
        task_start.TaskStartExample(
            task_id=task_id,
            metadata=metadata,
            total_output_tokens=target,
            call_count=1,
            mean_output_tokens=target,
            fold=0,
            model="model",
        )
        for task_id, target in (("a", 10.0), ("b", 40.0), ("c", 120.0))
    )

    scale = task_start._fit_scale({"a": 1.0, "b": 2.0, "c": 4.0}, examples)

    assert scale == 30.0


def test_refined_task_start_loader_reads_only_agent_prompt_fields(tmp_path) -> None:
    path = tmp_path / "metadata.parquet"
    table = pa.table(
        {
            "instance_id": ["selected", "sealed"],
            "repo": ["org/repo", "private/repo"],
            "repo_language": ["Python", ""],
            "problem_statement": ["Fix the parser.", ""],
            "requirements": ["- Preserve ordering.", ""],
            "interface": ["Name: parse", ""],
            "patch": ["future-label", "sealed-future-label"],
        }
    )
    pq.write_table(table, path)

    metadata = refinement._load_metadata(path, frozenset({"selected"}))

    assert metadata == {
        "selected": refinement.LaunchMetadata(
            task_id="selected",
            repo="org/repo",
            repo_language="Python",
            problem_statement="Fix the parser.",
            requirements="- Preserve ordering.",
            interface="Name: parse",
        )
    }


def test_refined_prompt_matches_the_public_agent_prompt_contract() -> None:
    metadata = refinement.LaunchMetadata(
        task_id="task",
        repo="org/repo",
        repo_language="Python",
        problem_statement="Fix the parser.",
        requirements="- Preserve ordering.",
        interface="Name: parse",
    )

    assert refinement._prompt_document(metadata) == (
        "Fix the parser.\n\n"
        "Requirements:\n- Preserve ordering.\n\n"
        "New interfaces introduced:\nName: parse"
    )


def test_refined_hybrid_uses_residual_point_and_fusion_interval() -> None:
    metadata = refinement.LaunchMetadata(
        task_id="task",
        repo="org/repo",
        repo_language="Python",
        problem_statement="Fix the parser.",
        requirements="- Preserve ordering.",
        interface="Name: parse",
    )
    example = task_start.TaskStartExample(
        task_id="task",
        metadata=metadata,
        total_output_tokens=105,
        call_count=2,
        mean_output_tokens=52.5,
        fold=0,
        model="model",
    )
    point_forecast = TokenForecast(
        point_id=example.point_id,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        lower=80,
        point=100,
        upper=120,
        raw_lower=80,
        raw_point=100,
        raw_upper=120,
    )
    interval_forecast = TokenForecast(
        point_id=example.point_id,
        target=PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS,
        lower=70,
        point=90,
        upper=110,
        raw_lower=75,
        raw_point=90,
        raw_upper=105,
    )
    records = {
        "point": [refinement._record("point", example, point_forecast)],
        "interval": [refinement._record("interval", example, interval_forecast)],
    }
    scored = {
        candidate_id: [
            ScoredForecast(
                task_id="task",
                trajectory_id="task",
                forecast=forecast,
                target_value=105,
                sample_weight=1,
            )
        ]
        for candidate_id, forecast in (
            ("point", point_forecast),
            ("interval", interval_forecast),
        )
    }

    refinement._add_hybrid_arm(
        records,
        scored,
        point_arm_id="point",
        interval_arm_id="interval",
        hybrid_id="hybrid",
    )

    forecast = records["hybrid"][0].forecast
    assert (forecast.lower, forecast.point, forecast.upper) == (70, 100, 110)
    assert (forecast.raw_lower, forecast.raw_point, forecast.raw_upper) == (75, 100, 105)


def test_refined_task_start_additional_references_ignore_query_labels() -> None:
    def example(task_id: str, target: float, repo: str, language: str):
        return task_start.TaskStartExample(
            task_id=task_id,
            metadata=refinement.LaunchMetadata(
                task_id=task_id,
                repo=repo,
                repo_language=language,
                problem_statement=f"Fix shared parser case {task_id[-1]}.",
                requirements="- Preserve ordering.\n- Handle malformed input.",
                interface="Path: /parse\nType: object",
            ),
            total_output_tokens=target,
            call_count=2,
            mean_output_tokens=target / 2,
            fold=0,
            model="model",
        )

    train = tuple(
        example(
            f"train-{index}",
            100 + 25 * index,
            "org/parser" if index < 6 else "org/runner",
            "python" if index % 2 else "go",
        )
        for index in range(12)
    )
    query = (
        example("query-0", 1, "org/parser", "python"),
        example("query-1", 2, "org/new", "go"),
    )
    changed_query = tuple(
        task_start.TaskStartExample(
            task_id=row.task_id,
            metadata=row.metadata,
            total_output_tokens=1_000_000,
            call_count=row.call_count,
            mean_output_tokens=500_000,
            fold=row.fold,
            model=row.model,
        )
        for row in query
    )
    config = {
        "length_component": "length",
        "text_component": "text",
        "structure_component": "structure",
        "candidate": "fusion",
        "residual_candidate": "residual",
        "text_ridge_alpha": 0.3,
        "residual_text_ridge_alpha": 3.0,
        "structure_loss": "huber",
        "structure_estimators": 10,
        "structure_learning_rate": 0.05,
        "structure_max_depth": 2,
        "structure_min_samples_leaf": 2,
        "group_reference": "group",
        "retrieval_reference": "retrieval",
        "retrieval_neighbors": 3,
        "quantile_tree_reference": "quantile_tree",
        "extra_trees_reference": "extra_trees",
        "extra_trees_estimators": 10,
        "extra_trees_min_samples_leaf": 2,
        "extra_trees_max_features": 1.0,
    }

    first, _ = refinement._fit_score_models(train, query, refinement=config)
    changed, _ = refinement._fit_score_models(
        train,
        changed_query,
        refinement=config,
    )

    reference_ids = {"group", "retrieval", "quantile_tree", "extra_trees"}
    assert reference_ids <= first.keys()
    assert all(first[arm_id] == changed[arm_id] for arm_id in reference_ids)
    assert all(value > 0 for arm_id in reference_ids for value in first[arm_id].values())
