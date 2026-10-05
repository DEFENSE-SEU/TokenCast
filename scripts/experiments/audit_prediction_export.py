from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import strict_json_object
from token_prediction.dataset import LabelStatus, POINT_WEIGHTING_ID, PredictionTarget
from token_prediction.evaluation import (
    ScoredForecast,
    adjust_comparison_family,
    bootstrap_two_sided_p,
    evaluate_cohort_forecasts,
    evaluate_forecasts,
    evaluate_task_forecasts,
    paired_task_bootstrap,
)
from token_prediction.experiments.candidates import PredictionRecord
from token_prediction.experiments.lineage import ArtifactManifest, verify_artifact
from token_prediction.experiments.prediction_export import (
    DENSE_PREDICTION_EXPORT_SCHEMA_ID,
    LEGACY_PREDICTION_EXPORT_SCHEMA_ID,
    PREDICTION_EXPORT_SCHEMA_ID,
    load_prediction_export,
)
from token_prediction.forecast import TokenForecast
from token_prediction.pipeline import (
    PAIRED_FEATURE_COMPARISON_METHOD,
)


AUDIT_SCHEMA_ID = "scored-prediction-artifact-audit-v2"
FITTED_PACKAGE_STAGE_NAME = "fitted-fold-package"
FITTED_PACKAGE_SCHEMA_VERSION = 1
_SUPPORTED_PREDICTION_EXPORT_SCHEMAS = frozenset(
    {
        LEGACY_PREDICTION_EXPORT_SCHEMA_ID,
        DENSE_PREDICTION_EXPORT_SCHEMA_ID,
        PREDICTION_EXPORT_SCHEMA_ID,
    }
)
_RICH_PREDICTION_EXPORT_SCHEMAS = (
    _SUPPORTED_PREDICTION_EXPORT_SCHEMAS - {LEGACY_PREDICTION_EXPORT_SCHEMA_ID}
)


@dataclass(frozen=True)
class _ParsedPrediction:
    document: Mapping[str, Any]
    scored: ScoredForecast
    record: PredictionRecord


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _field(document: Mapping[str, Any], name: str) -> Any:
    if name not in document:
        raise ValueError(f"prediction record is missing {name}")
    return document[name]


def _text(document: Mapping[str, Any], name: str) -> str:
    value = _field(document, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prediction {name} must be non-empty text")
    return value


def _optional_text(document: Mapping[str, Any], name: str) -> str | None:
    value = _field(document, name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prediction {name} must be non-empty text or null")
    return value


def _integer(document: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = _field(document, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"prediction {name} must be an integer >= {minimum}")
    return value


def _number(
    document: Mapping[str, Any],
    name: str,
    *,
    minimum: float | None = 0.0,
) -> float:
    value = _field(document, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prediction {name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or (minimum is not None and resolved < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"prediction {name} must be {qualifier}")
    return resolved


def _optional_number(document: Mapping[str, Any], name: str) -> float | None:
    value = _field(document, name)
    if value is None:
        return None
    return _number(document, name, minimum=None)


def _parse_prediction(document: Mapping[str, Any]) -> _ParsedPrediction:
    schema_id = _field(document, "prediction_export_schema_id")
    if schema_id not in _SUPPORTED_PREDICTION_EXPORT_SCHEMAS:
        raise ValueError("prediction record uses an unsupported export schema")
    if schema_id in _RICH_PREDICTION_EXPORT_SCHEMAS:
        _text(document, "fitted_artifact_id")
    elif (
        "fitted_artifact_id" in document
        and document["fitted_artifact_id"] is not None
    ):
        _text(document, "fitted_artifact_id")
    if _field(document, "label_status") != LabelStatus.OBSERVED.value:
        raise ValueError("scored prediction records require label_status=observed")
    if _field(document, "bound_direction") is not None or _field(
        document, "bound_value"
    ) is not None:
        raise ValueError("observed prediction records cannot carry censoring bounds")

    target = PredictionTarget(_text(document, "target"))
    point_id = _text(document, "point_id")
    raw_lower = _optional_number(document, "raw_lower")
    raw_point = _optional_number(document, "raw_prediction")
    raw_upper = _optional_number(document, "raw_upper")
    forecast = TokenForecast(
        point_id=point_id,
        target=target,
        lower=_number(document, "lower"),
        point=_number(document, "prediction"),
        upper=_number(document, "upper"),
        latency_ms=_number(document, "latency_ms"),
        overhead_input_tokens=_integer(document, "overhead_input_tokens"),
        overhead_output_tokens=_integer(document, "overhead_output_tokens"),
        raw_lower=raw_lower,
        raw_point=raw_point,
        raw_upper=raw_upper,
    )
    candidate_id = _text(document, "candidate_id")
    task_id = _text(document, "task_id")
    trajectory_id = _text(document, "trajectory_id")
    condition_id = _text(document, "condition_id")
    fold = _integer(document, "fold")
    sample_weight = _number(document, "sample_weight", minimum=0.0)
    if sample_weight == 0.0:
        raise ValueError("prediction sample_weight must be positive")
    target_value = _integer(document, "target_value")
    token_unit_contract_id = _optional_text(document, "token_unit_contract_id")
    cohorts = {
        name: _optional_text(document, name)
        for name in (
            "task_family_id",
            "split_group_id",
            "task_type",
            "actor_model_id",
            "model_family_id",
            "retry_state",
            "termination_state",
            "trajectory_length_bucket",
        )
    }
    scored = ScoredForecast(
        task_id=task_id,
        trajectory_id=trajectory_id,
        forecast=forecast,
        target_value=float(target_value),
        sample_weight=sample_weight,
        **cohorts,
    )
    record = PredictionRecord(
        candidate_id=candidate_id,
        point_id=point_id,
        task_id=task_id,
        trajectory_id=trajectory_id,
        condition_id=condition_id,
        fold=fold,
        target=target,
        forecast=forecast,
        sample_weight=sample_weight,
        token_unit_contract_id=token_unit_contract_id,
        **cohorts,
    )
    return _ParsedPrediction(document=document, scored=scored, record=record)


def _load_fitted_packages(directory: Path) -> dict[str, ArtifactManifest]:
    models_directory = directory / "models"
    packages: dict[str, ArtifactManifest] = {}
    if not models_directory.is_dir():
        return packages
    for manifest_path in sorted(models_directory.rglob("manifest.json")):
        document = _load_json(manifest_path)
        if document.get("stage_name") != FITTED_PACKAGE_STAGE_NAME:
            continue
        manifest = verify_artifact(manifest_path.parent)
        if manifest.schema_version != FITTED_PACKAGE_SCHEMA_VERSION:
            raise ValueError("fitted package uses an unsupported artifact schema")
        _assert_equivalent(
            {
                "fitted_package_id": manifest.metadata.get("model_id"),
                "experiment_run_id": manifest.metadata.get("experiment_run_id"),
                "experiment_id": manifest.metadata.get("experiment_id"),
                "candidate_id": manifest.metadata.get("candidate_id"),
                "candidate_model_revision": manifest.metadata.get(
                    "candidate_model_revision"
                ),
                "fitted_model_revision": manifest.metadata.get("model_revision"),
                "split_seed": manifest.metadata.get("split_seed"),
                "fit_seed": manifest.metadata.get("fit_seed"),
                "fold": manifest.metadata.get("fold"),
                "calibrator_id": manifest.metadata.get("calibrator_id"),
                "dataset_id": manifest.metadata.get("dataset_id"),
                "split_plan_id": manifest.metadata.get("split_plan_id"),
                "token_unit_contract_id": manifest.metadata.get(
                    "token_unit_contract_id"
                ),
            },
            _load_json(manifest_path.parent / "fold.json"),
            path=f"fitted_packages.{manifest.artifact_id}.fold",
        )
        if manifest.artifact_id in packages:
            raise ValueError(f"duplicate fitted artifact id {manifest.artifact_id}")
        packages[manifest.artifact_id] = manifest
    return packages


def _validate_fitted_package_bindings(
    directory: Path,
    run: Mapping[str, Any],
    predictions: Sequence[_ParsedPrediction],
) -> int:
    if run.get("prediction_export_schema_id") == LEGACY_PREDICTION_EXPORT_SCHEMA_ID:
        return 0

    packages = _load_fitted_packages(directory)
    referenced: set[str] = set()
    fit_seed = run.get("fit_seed")
    if fit_seed is None:
        fit_seed = run.get("split_seed")
    for item in predictions:
        artifact_id = _text(item.document, "fitted_artifact_id")
        manifest = packages.get(artifact_id)
        if manifest is None:
            raise ValueError(f"prediction references absent fitted artifact {artifact_id}")
        expected_metadata = {
            "experiment_run_id": item.document["experiment_run_id"],
            "experiment_id": item.document["experiment_id"],
            "candidate_id": item.document["candidate_id"],
            "candidate_model_revision": item.document["model_revision"],
            "dataset_id": item.document["dataset_id"],
            "split_plan_id": item.document["split_plan_id"],
            "split_seed": item.document["split_seed"],
            "fit_seed": fit_seed,
            "fold": item.document["fold"],
            "token_unit_contract_id": item.document["token_unit_contract_id"],
        }
        for name, expected in expected_metadata.items():
            if manifest.metadata.get(name) != expected:
                raise ValueError(
                    f"fitted artifact {artifact_id} metadata {name} disagrees "
                    "with its prediction"
                )
        referenced.add(artifact_id)
    unreferenced = sorted(set(packages) - referenced)
    if unreferenced:
        raise ValueError(f"fitted artifacts have no prediction records: {unreferenced}")
    return len(packages)


def _assert_equivalent(actual: Any, recorded: Any, *, path: str) -> None:
    if isinstance(actual, Mapping):
        if not isinstance(recorded, Mapping):
            raise ValueError(f"{path} type differs from the recomputed value")
        if set(actual) != set(recorded):
            missing = sorted(set(actual) - set(recorded))
            extra = sorted(set(recorded) - set(actual))
            raise ValueError(f"{path} keys differ: missing={missing}, extra={extra}")
        for key in actual:
            _assert_equivalent(actual[key], recorded[key], path=f"{path}.{key}")
        return
    if isinstance(actual, (list, tuple)):
        if not isinstance(recorded, (list, tuple)) or len(actual) != len(recorded):
            raise ValueError(f"{path} sequence shape differs from the recomputed value")
        for index, (actual_item, recorded_item) in enumerate(
            zip(actual, recorded, strict=True)
        ):
            _assert_equivalent(
                actual_item,
                recorded_item,
                path=f"{path}[{index}]",
            )
        return
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(recorded, (int, float))
        and not isinstance(recorded, bool)
    ):
        if not math.isclose(
            float(actual),
            float(recorded),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{path} differs: recomputed={actual!r}, recorded={recorded!r}"
            )
        return
    if actual != recorded:
        raise ValueError(
            f"{path} differs: recomputed={actual!r}, recorded={recorded!r}"
        )


def _alpha(metric_document: Mapping[str, Any], *, path: str) -> float:
    nominal = metric_document.get("nominal_coverage")
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)):
        raise ValueError(f"{path}.nominal_coverage must be numeric")
    alpha = 1.0 - float(nominal)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{path}.nominal_coverage is outside (0, 1)")
    return alpha


def _task_statistics(
    candidate_rows: Sequence[_ParsedPrediction],
    reference_rows: Sequence[_ParsedPrediction],
    *,
    alpha: float,
) -> dict[str, dict[str, float | int]]:
    candidate = evaluate_task_forecasts(
        [item.scored for item in candidate_rows],
        alpha=alpha,
    )
    reference = evaluate_task_forecasts(
        [item.scored for item in reference_rows],
        alpha=alpha,
    )
    if set(candidate) != set(reference):
        raise ValueError("comparison candidates have different Task cohorts")
    return {
        task_id: {
            "n_points": candidate[task_id].n_points,
            "n_trajectories": candidate[task_id].n_trajectories,
            "weight_sum": candidate[task_id].weight_sum,
            "candidate_weighted_mae": candidate[task_id].weighted_mae,
            "reference_weighted_mae": reference[task_id].weighted_mae,
        }
        for task_id in sorted(candidate)
    }


def audit_prediction_export(fit_dir: str | Path) -> dict[str, Any]:
    directory = Path(fit_dir).resolve()
    run = _load_json(directory / "run.json")
    metrics = _load_json(directory / "metrics.json")
    comparisons = _load_json(directory / "comparisons.json")
    predictions = [_parse_prediction(item) for item in load_prediction_export(directory)]

    run_to_prediction = {
        "run_id": "experiment_run_id",
        "prediction_export_schema_id": "prediction_export_schema_id",
        "dataset_id": "dataset_id",
        "dataset_schema_version": "dataset_schema_version",
        "split_plan_id": "split_plan_id",
        "split_seed": "split_seed",
        "source_id": "source_id",
        "source_descriptor_id": "source_descriptor_id",
        "capability_profile_id": "capability_profile_id",
        "input_contract_id": "input_contract_id",
        "point_set_id": "point_set_id",
    }
    for item in predictions:
        for run_field, prediction_field in run_to_prediction.items():
            if run.get(run_field) != item.document.get(prediction_field):
                raise ValueError(
                    f"prediction {prediction_field} disagrees with run.json {run_field}"
                )
        _text(item.document, "experiment_id")
        _text(item.document, "source_event_id")
        _text(item.document, "run_id")
        _text(item.document, "prediction_context_id")
        _integer(item.document, "cutoff_event_seq")
        _text(item.document, "position")
        _field(item.document, "logical_call_id")
        _field(item.document, "attempt_id")
        _field(item.document, "retry_ordinal")
        _field(item.document, "provider")
        _field(item.document, "model_id")
        if item.document["prediction_export_schema_id"] in _RICH_PREDICTION_EXPORT_SCHEMAS:
            _field(item.document, "reasoning_setting")
            _field(item.document, "repository")
            progress_known = _field(item.document, "progress_known_amount")
            if progress_known is not None:
                _number(item.document, "progress_known_amount")
            progress_source = _text(item.document, "progress_known_amount_source")
            if progress_source not in {
                "known_offset_tokens",
                "cumulative_provider_input_plus_output",
                "unavailable",
            }:
                raise ValueError("prediction progress_known_amount_source is unsupported")
            if (progress_known is None) != (progress_source == "unavailable"):
                raise ValueError("prediction progress amount and source disagree")
        _field(item.document, "checkpoint_ordinal")
        known_offset = _field(item.document, "known_offset_tokens")
        if known_offset is not None:
            _number(item.document, "known_offset_tokens")
        if (
            item.document["prediction_export_schema_id"] in _RICH_PREDICTION_EXPORT_SCHEMAS
            and item.document["progress_known_amount_source"] == "known_offset_tokens"
            and item.document["progress_known_amount"] != known_offset
        ):
            raise ValueError("prediction progress amount differs from known offset")

    fitted_package_count = _validate_fitted_package_bindings(
        directory,
        run,
        predictions,
    )

    grouped: dict[tuple[str, str], list[_ParsedPrediction]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for item in predictions:
        experiment_id = str(item.document["experiment_id"])
        candidate_id = item.record.candidate_id
        identity = (experiment_id, candidate_id, item.record.point_id)
        if identity in seen:
            raise ValueError(f"duplicate exported prediction {identity!r}")
        seen.add(identity)
        grouped[(experiment_id, candidate_id)].append(item)

    expected_groups = {
        (experiment_id, candidate_id)
        for experiment_id, candidates in metrics.items()
        for candidate_id in candidates
    }
    if set(grouped) != expected_groups:
        raise ValueError("prediction and metric candidate groups differ")

    alpha_by_group: dict[tuple[str, str], float] = {}
    for experiment_id, candidate_id in sorted(expected_groups):
        metric_entry = metrics[experiment_id][candidate_id]
        rows = grouped[(experiment_id, candidate_id)]
        model_revisions = {item.document["model_revision"] for item in rows}
        if model_revisions != {metric_entry["model_revision"]}:
            raise ValueError(
                f"{experiment_id}.{candidate_id} model revision differs across artifacts"
            )
        metric_document = metric_entry["metrics"]
        alpha = _alpha(
            metric_document,
            path=f"metrics.{experiment_id}.{candidate_id}.metrics",
        )
        alpha_by_group[(experiment_id, candidate_id)] = alpha
        _assert_equivalent(
            evaluate_forecasts([item.scored for item in rows], alpha=alpha),
            metric_document,
            path=f"metrics.{experiment_id}.{candidate_id}.metrics",
        )

        by_fold: dict[int, list[ScoredForecast]] = defaultdict(list)
        for item in rows:
            by_fold[item.record.fold].append(item.scored)
        recomputed_fold_metrics = {
            str(fold): evaluate_forecasts(fold_rows, alpha=alpha)
            for fold, fold_rows in sorted(by_fold.items())
        }
        _assert_equivalent(
            recomputed_fold_metrics,
            metric_entry["fold_metrics"],
            path=f"metrics.{experiment_id}.{candidate_id}.fold_metrics",
        )
        _assert_equivalent(
            evaluate_cohort_forecasts([item.scored for item in rows], alpha=alpha),
            metric_entry["cohort_metrics"],
            path=f"metrics.{experiment_id}.{candidate_id}.cohort_metrics",
        )

    comparison_count = 0
    family_p_values: dict[str, float] = {}
    for experiment_id, arms in comparisons.items():
        for candidate_id, document in arms.items():
            comparison_count += 1
            if document.get("experiment_id") != experiment_id:
                raise ValueError("comparison experiment_id disagrees with its parent key")
            if document.get("candidate_id") != candidate_id:
                raise ValueError("comparison candidate_id disagrees with its parent key")
            if document.get("method") != PAIRED_FEATURE_COMPARISON_METHOD:
                raise ValueError("comparison uses an unsupported method")
            weighting = document.get("weighting_contract")
            if not isinstance(weighting, Mapping) or weighting.get(
                "point_weighting_id"
            ) != POINT_WEIGHTING_ID:
                raise ValueError("comparison uses an unsupported point weighting contract")
            reference_id = str(document.get("reference_id") or "")
            candidate_key = (experiment_id, candidate_id)
            reference_key = (experiment_id, reference_id)
            if candidate_key not in grouped or reference_key not in grouped:
                raise ValueError("comparison references an absent prediction group")
            candidate_rows = grouped[candidate_key]
            reference_rows = grouped[reference_key]
            alpha = alpha_by_group[candidate_key]
            if not math.isclose(
                alpha,
                alpha_by_group[reference_key],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("comparison candidates use different alpha values")
            truth_by_point = {
                item.record.point_id: item.scored.target_value for item in candidate_rows
            }
            reference_truth = {
                item.record.point_id: item.scored.target_value for item in reference_rows
            }
            if truth_by_point != reference_truth:
                raise ValueError("comparison candidates have different target values")
            recomputed = paired_task_bootstrap(
                [item.record for item in candidate_rows],
                [item.record for item in reference_rows],
                truth_by_point,
                iterations=int(document["iterations"]),
                seed=int(document["seed"]),
            )
            for name, value in asdict(recomputed).items():
                _assert_equivalent(
                    value,
                    document.get(name),
                    path=f"comparisons.{experiment_id}.{candidate_id}.{name}",
                )
            _assert_equivalent(
                _task_statistics(candidate_rows, reference_rows, alpha=alpha),
                document["task_sufficient_statistics"],
                path=(
                    f"comparisons.{experiment_id}.{candidate_id}"
                    ".task_sufficient_statistics"
                ),
            )
            family_p_values[f"{experiment_id}/{candidate_id}"] = bootstrap_two_sided_p(
                recomputed.candidate_win_probability,
                iterations=recomputed.iterations,
            )

    correction_path = directory / "comparison_family_correction.json"
    if family_p_values:
        _assert_equivalent(
            adjust_comparison_family(family_p_values),
            _load_json(correction_path),
            path="comparison_family_correction",
        )
    elif correction_path.exists():
        raise ValueError("comparison family correction exists without paired comparisons")

    return {
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "status": "passed",
        "run_id": run["run_id"],
        "prediction_record_count": len(predictions),
        "experiment_count": len(metrics),
        "candidate_group_count": len(grouped),
        "fitted_package_count": fitted_package_count,
        "paired_comparison_count": comparison_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_dir", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit_prediction_export(arguments.fit_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
