from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from token_prediction.canonical_json import strict_json_object


LEGACY_PREDICTION_EXPORT_SCHEMA_ID = "scored-prediction-record-v1"
DENSE_PREDICTION_EXPORT_SCHEMA_ID = "scored-prediction-record-v2"
PREDICTION_EXPORT_SCHEMA_ID = "scored-prediction-record-v3"
PREDICTION_RECORD_DEFAULTS_SCHEMA_ID = "scored-prediction-defaults-v1"
_RUN_DEFAULT_FIELDS = frozenset(
    {
        "experiment_run_id",
        "dataset_id",
        "dataset_schema_version",
        "split_plan_id",
        "split_seed",
        "source_id",
        "source_descriptor_id",
        "capability_profile_id",
        "input_contract_id",
        "point_set_id",
    }
)
_EXPERIMENT_DEFAULT_FIELDS = frozenset(
    {
        "eligibility_policy_id",
        "condition_id",
        "position",
        "target",
        "token_unit_contract_id",
    }
)
_CANDIDATE_DEFAULT_FIELDS = frozenset(
    {
        "model_revision",
        "fitted_artifact_ids",
    }
)


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


def _prediction_defaults(
    run: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
]:
    defaults = run.get("prediction_record_defaults")
    if run.get("prediction_export_schema_id") != PREDICTION_EXPORT_SCHEMA_ID:
        return {}, {}, {}
    if not isinstance(defaults, dict) or set(defaults) != {
        "schema_id",
        "run",
        "experiments",
        "candidates",
    }:
        raise ValueError("compact prediction export requires run defaults")
    if defaults["schema_id"] != PREDICTION_RECORD_DEFAULTS_SCHEMA_ID:
        raise ValueError("prediction record defaults schema is unsupported")
    run_defaults = defaults["run"]
    experiments = defaults["experiments"]
    candidates = defaults["candidates"]
    if (
        not isinstance(run_defaults, dict)
        or not isinstance(experiments, dict)
        or not isinstance(candidates, dict)
    ):
        raise ValueError("prediction record defaults must contain object mappings")
    if set(run_defaults) != _RUN_DEFAULT_FIELDS:
        raise ValueError("prediction run defaults have an invalid field set")
    normalized: dict[str, dict[str, Any]] = {}
    for experiment_id, values in experiments.items():
        if not isinstance(experiment_id, str) or not experiment_id or not isinstance(values, dict):
            raise ValueError("prediction experiment defaults are invalid")
        if set(values) != _EXPERIMENT_DEFAULT_FIELDS:
            raise ValueError("prediction experiment defaults have an invalid field set")
        if set(run_defaults) & set(values):
            raise ValueError("run and experiment prediction defaults overlap")
        normalized[experiment_id] = dict(values)
    if set(candidates) != set(normalized):
        raise ValueError("prediction candidate defaults differ from experiments")
    normalized_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for experiment_id, raw_candidates in candidates.items():
        if not isinstance(raw_candidates, dict) or not raw_candidates:
            raise ValueError("prediction candidate defaults are invalid")
        experiment_candidates: dict[str, dict[str, Any]] = {}
        for candidate_id, values in raw_candidates.items():
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or not isinstance(values, dict)
                or set(values) != _CANDIDATE_DEFAULT_FIELDS
            ):
                raise ValueError("prediction candidate defaults are invalid")
            model_revision = values["model_revision"]
            fitted_artifact_ids = values["fitted_artifact_ids"]
            if (
                not isinstance(model_revision, str)
                or not model_revision
                or not isinstance(fitted_artifact_ids, dict)
                or not fitted_artifact_ids
                or any(
                    not isinstance(fold, str)
                    or not fold.isdecimal()
                    or str(int(fold)) != fold
                    or not isinstance(artifact_id, str)
                    or not artifact_id
                    for fold, artifact_id in fitted_artifact_ids.items()
                )
            ):
                raise ValueError("prediction candidate fold defaults are invalid")
            experiment_candidates[candidate_id] = {
                "model_revision": model_revision,
                "fitted_artifact_ids": dict(fitted_artifact_ids),
            }
        normalized_candidates[experiment_id] = experiment_candidates
    return dict(run_defaults), normalized, normalized_candidates


def iter_prediction_export(directory: str | Path) -> Iterator[dict[str, Any]]:


    root = Path(directory)
    run = _load_json(root / "run.json")
    run_defaults, experiment_defaults, candidate_defaults = _prediction_defaults(run)
    compact = run.get("prediction_export_schema_id") == PREDICTION_EXPORT_SCHEMA_ID
    path = root / "predictions.jsonl"
    count = 0
    try:
        source = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=strict_json_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {path.name} record at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} line {line_number} must be a JSON object")
            if compact:
                if value.get("prediction_export_schema_id") != PREDICTION_EXPORT_SCHEMA_ID:
                    raise ValueError(
                        f"{path.name} line {line_number} disagrees with its run schema"
                    )
                experiment_id = value.get("experiment_id")
                if experiment_id not in experiment_defaults:
                    raise ValueError(
                        f"{path.name} line {line_number} has no experiment defaults"
                    )
                candidate_id = value.get("candidate_id")
                experiment_candidates = candidate_defaults[experiment_id]
                if candidate_id not in experiment_candidates:
                    raise ValueError(
                        f"{path.name} line {line_number} has no candidate defaults"
                    )
                fold = value.get("fold")
                if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
                    raise ValueError(f"{path.name} line {line_number} has an invalid fold")
                candidate = experiment_candidates[candidate_id]
                fitted_artifact_ids = candidate["fitted_artifact_ids"]
                artifact_id = fitted_artifact_ids.get(str(fold))
                if artifact_id is None:
                    raise ValueError(
                        f"{path.name} line {line_number} has no candidate fold defaults"
                    )
                defaults = {
                    **run_defaults,
                    **experiment_defaults[experiment_id],
                    "model_revision": candidate["model_revision"],
                    "fitted_artifact_id": artifact_id,
                }
                for name, expected in defaults.items():
                    if name in value and value[name] != expected:
                        raise ValueError(
                            f"{path.name} line {line_number} overrides default {name}"
                        )
                value = {**defaults, **value}
            count += 1
            yield value
    if not count:
        raise ValueError(f"{path.name} contains no prediction records")


def load_prediction_export(directory: str | Path) -> tuple[dict[str, Any], ...]:
    return tuple(iter_prediction_export(directory))


__all__ = [
    "DENSE_PREDICTION_EXPORT_SCHEMA_ID",
    "LEGACY_PREDICTION_EXPORT_SCHEMA_ID",
    "PREDICTION_EXPORT_SCHEMA_ID",
    "PREDICTION_RECORD_DEFAULTS_SCHEMA_ID",
    "load_prediction_export",
    "iter_prediction_export",
]
