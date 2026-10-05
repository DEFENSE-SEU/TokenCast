from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.experiments.audit_prediction_export import _parse_prediction
from token_prediction.evaluation import paired_task_bootstrap
from token_prediction.experiments.prediction_export import load_prediction_export


ITERATIONS = 10_000
REFERENCE_ID = "empirical"
SCHEMA_ID = "paired-reference-comparison-v1"


def _task_statistics(candidate: Sequence[Any], reference: Sequence[Any]) -> dict[str, dict[str, float | int]]:
    reference_by_point = {item.record.point_id: item.record for item in reference}
    totals: dict[str, list[Any]] = defaultdict(lambda: [0.0, 0, set(), 0.0, 0.0])
    for item in candidate:
        record = item.record
        other = reference_by_point.get(record.point_id)
        if other is None:
            raise ValueError(f"paired comparison is missing point {record.point_id!r}")
        if (
            record.task_id != other.task_id
            or record.trajectory_id != other.trajectory_id
            or record.condition_id != other.condition_id
            or record.fold != other.fold
            or record.forecast.target != other.forecast.target
            or record.sample_weight != other.sample_weight
        ):
            raise ValueError(f"paired comparison metadata differs at {record.point_id!r}")
        truth = float(item.scored.target_value)
        if not math.isfinite(truth) or truth < 0:
            raise ValueError("paired comparison targets must be finite and non-negative")
        task = totals[record.task_id]
        weight = float(record.sample_weight)
        task[0] += weight
        task[1] += 1
        task[2].add(record.trajectory_id)
        task[3] += weight * abs(float(record.forecast.point) - truth)
        task[4] += weight * abs(float(other.forecast.point) - truth)
    return {
        task_id: {
            "n_points": values[1],
            "n_trajectories": len(values[2]),
            "weight_sum": values[0],
            "candidate_weighted_mae": values[3] / values[0],
            "reference_weighted_mae": values[4] / values[0],
        }
        for task_id, values in sorted(totals.items())
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _seed_mean_comparison(
    comparisons: Sequence[Mapping[str, Mapping[str, float | int]]],
    *,
    candidate_id: str,
    seed: int,
) -> dict[str, Any]:
    task_ids = set(comparisons[0])
    if any(set(item) != task_ids for item in comparisons[1:]):
        raise ValueError("split comparisons use different Task cohorts")
    candidate = {
        task_id: sum(float(item[task_id]["candidate_weighted_mae"]) for item in comparisons)
        / len(comparisons)
        for task_id in sorted(task_ids)
    }
    reference = {
        task_id: sum(float(item[task_id]["reference_weighted_mae"]) for item in comparisons)
        / len(comparisons)
        for task_id in sorted(task_ids)
    }
    tasks = sorted(task_ids)
    deltas = [candidate[task_id] - reference[task_id] for task_id in tasks]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(ITERATIONS):
        sampled = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        candidate_mean = sum(candidate[task_id] for task_id in sampled) / len(sampled)
        reference_mean = sum(reference[task_id] for task_id in sampled) / len(sampled)
        draws.append(candidate_mean - reference_mean)
    return {
        "candidate_id": candidate_id,
        "reference_id": REFERENCE_ID,
        "n_tasks": len(tasks),
        "n_splits": len(comparisons),
        "iterations": ITERATIONS,
        "seed": seed,
        "candidate_mae": sum(candidate.values()) / len(tasks),
        "reference_mae": sum(reference.values()) / len(tasks),
        "mae_delta": sum(deltas) / len(tasks),
        "mae_delta_ci_lower": _percentile(draws, 0.025),
        "mae_delta_ci_upper": _percentile(draws, 0.975),
        "candidate_win_probability": sum(value < 0 for value in draws) / ITERATIONS,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "aggregation": "mean_task_weighted_mae_across_split_seeds",
    }


def _source_record(path: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fit_directory": path.as_posix(),
        "predictions_file": path.joinpath("predictions.jsonl").as_posix(),
        "predictions_bytes": path.joinpath("predictions.jsonl").stat().st_size,
        "run_id": run["run_id"],
        "split_seed": run["split_seed"],
        "fit_seed": run["fit_seed"],
        "dataset_id": run["dataset_id"],
        "evaluation_scope": run["evaluation_scope"],
        "final_holdout_status": run["final_holdout_status"],
    }


def derive(fit_dirs: Iterable[Path]) -> dict[str, Any]:
    split_documents: list[dict[str, Any]] = []
    by_experiment: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for fit_dir in fit_dirs:
        run = json.loads(fit_dir.joinpath("run.json").read_text(encoding="utf-8"))
        documents = load_prediction_export(fit_dir)
        parsed = [_parse_prediction(document) for document in documents]
        groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for item in parsed:
            groups[(item.document["experiment_id"], item.record.candidate_id)].append(item)
        split_comparisons: dict[str, dict[str, Any]] = {}
        for experiment_id in sorted({key[0] for key in groups}):
            candidates = {
                candidate_id: rows
                for (current_experiment, candidate_id), rows in groups.items()
                if current_experiment == experiment_id
            }
            reference = candidates.get(REFERENCE_ID)
            if reference is None:
                raise ValueError(f"{experiment_id} has no {REFERENCE_ID!r} reference")
            for candidate_id, candidate in sorted(candidates.items()):
                if candidate_id == REFERENCE_ID:
                    continue
                truth = {item.record.point_id: item.scored.target_value for item in candidate}
                comparison = paired_task_bootstrap(
                    [item.record for item in candidate],
                    [item.record for item in reference],
                    truth,
                    iterations=ITERATIONS,
                    seed=int(run["split_seed"]),
                )
                task_stats = _task_statistics(candidate, reference)
                split_comparisons.setdefault(experiment_id, {})[candidate_id] = {
                    "comparison": asdict(comparison),
                    "task_sufficient_statistics": task_stats,
                }
                by_experiment[experiment_id][candidate_id].append(
                    (task_stats, asdict(comparison))
                )
        split_documents.append(
            {
                **_source_record(fit_dir, run),
                "experiment_count": len(split_comparisons),
                "candidate_comparison_count": sum(
                    len(candidates) for candidates in split_comparisons.values()
                ),
                "comparisons": split_comparisons,
            }
        )

    pooled: dict[str, dict[str, Any]] = {}
    for experiment_id, candidates in sorted(by_experiment.items()):
        pooled[experiment_id] = {}
        for candidate_id, values in sorted(candidates.items()):
            task_stats = [item[0] for item in values]
            pooled[experiment_id][candidate_id] = {
                "split_comparisons": [item[1] for item in values],
                "seed_mean_comparison": _seed_mean_comparison(
                    task_stats,
                    candidate_id=candidate_id,
                    seed=20260719,
                ),
            }
    return {
        "schema_id": SCHEMA_ID,
        "reference_candidate_id": REFERENCE_ID,
        "iterations": ITERATIONS,
        "bootstrap_unit": "task_id",
        "mae_delta_definition": "candidate_minus_reference",
        "split_count": len(split_documents),
        "split_sources": split_documents,
        "pooled": pooled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_dirs = tuple(sorted(path.resolve() for path in args.fit_dirs))
    if not fit_dirs:
        raise ValueError("at least one fit directory is required")
    if any(not path.is_dir() for path in fit_dirs):
        raise ValueError("every fit directory must exist")
    result = derive(fit_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": args.output.resolve().as_posix(), "status": "passed"}))


if __name__ == "__main__":
    main()
