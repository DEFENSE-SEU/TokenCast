from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from token_prediction.contracts import SourceDescriptor
from token_prediction.dataset import (
    PredictionPosition,
    PredictionTarget,
    TaskPartitionAuthority,
    load_task_partition_authority,
    target_requirements,
)
from token_prediction.evaluation import registered_calibrator_ids
from token_prediction.experiments.candidates import (
    AblationAxis,
    AblationSpec,
    CandidateGraph,
    CandidateRole,
    CandidateSpec,
    ExperimentSpec,
)
from token_prediction.features import FeatureGroup, FeatureSet


CONFIG_SCHEMA_VERSION = 2


def _reject_unknown(payload: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {', '.join(unknown)}")


def _required_text(value: object, *, context: str) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"{context} is required")
    return resolved


def _strings(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    result = frozenset(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(value):
        raise ValueError("string lists require unique non-empty values")
    return result


def _load_source_descriptor(project_root: Path, relative_path: str) -> SourceDescriptor:
    path = (project_root / relative_path).resolve()
    if not path.is_file():
        raise ValueError(f"source descriptor does not exist: {relative_path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source descriptor: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("source descriptor must contain one JSON object")
    return SourceDescriptor.from_dict(payload)


@dataclass(frozen=True)
class ConfiguredExperiment:
    experiment_id: str
    position: PredictionPosition
    target: PredictionTarget
    candidate_ids: tuple[str, ...]
    required_features: frozenset[str]
    condition_id: str | None
    token_unit_contract_id: str | None


    calibrator_id: str | None = None


def _experiment_calibrator(value: object, experiment_id: str) -> str | None:
    if value is None:
        return None
    calibrator_id = _required_text(value, context=f"experiment {experiment_id} calibrator")
    if calibrator_id not in registered_calibrator_ids():
        raise ValueError(f"experiment {experiment_id} names an unsupported interval calibrator")
    return calibrator_id


@dataclass(frozen=True)
class SweepPoint:


    value: Any
    candidate_id: str


@dataclass(frozen=True)
class SweepSpec:


    sweep_id: str
    experiment_id: str
    base_candidate_id: str
    param: str
    points: tuple[SweepPoint, ...]


@dataclass(frozen=True)
class ProjectConfig:
    config_id: str
    source_path: Path
    schema_version: int
    workspace: Path
    seed: int
    collection_source: str
    dataset_mode: str
    source_descriptor: SourceDescriptor | None
    source_descriptor_path: str | None
    tick_seconds: float | None
    split_unit: str
    folds: int
    partition_authority: TaskPartitionAuthority | None
    partition_authority_path: str | None
    alpha: float
    calibrator_id: str
    feature_sets: dict[str, FeatureSet]
    candidates: tuple[CandidateSpec, ...]
    experiments: tuple[ConfiguredExperiment, ...]
    sweeps: tuple[SweepSpec, ...] = ()

    def experiment_specs(self) -> tuple[ExperimentSpec, ...]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return tuple(
            ExperimentSpec(
                experiment_id=experiment.experiment_id,
                position=experiment.position,
                target=experiment.target,
                candidates=tuple(by_id[item] for item in experiment.candidate_ids),
                alpha=self.alpha,
                calibrator_id=experiment.calibrator_id or self.calibrator_id,
                required_features=experiment.required_features,
                condition_id=experiment.condition_id,
                token_unit_contract_id=experiment.token_unit_contract_id,
            )
            for experiment in self.experiments
        )


def _feature_set(feature_set_id: str, value: Any) -> FeatureSet:
    if not isinstance(value, dict):
        raise ValueError(f"feature_sets.{feature_set_id} must be a table")
    payload = dict(value)
    _reject_unknown(
        payload,
        {
            "include_all",
            "include_groups",
            "exclude_groups",
            "include_subgroups",
            "exclude_subgroups",
            "include_features",
            "exclude_features",
        },
        f"feature_sets.{feature_set_id}",
    )
    return FeatureSet(
        feature_set_id=feature_set_id,
        include_all=bool(payload.get("include_all", True)),
        include_groups=frozenset(
            FeatureGroup(item) for item in _strings(payload.get("include_groups"))
        ),
        exclude_groups=frozenset(
            FeatureGroup(item) for item in _strings(payload.get("exclude_groups"))
        ),
        include_subgroups=_strings(payload.get("include_subgroups")),
        exclude_subgroups=_strings(payload.get("exclude_subgroups")),
        include_features=_strings(payload.get("include_features")),
        exclude_features=_strings(payload.get("exclude_features")),
    )


def _candidate(
    value: Any,
    *,
    feature_sets: dict[str, FeatureSet],
) -> CandidateSpec:
    if not isinstance(value, dict):
        raise ValueError("candidate entries must be tables")
    item = dict(value)
    _reject_unknown(
        item,
        {
            "id",
            "estimator",
            "feature_set",
            "role",
            "params",
            "initializer_params",
            "graph",
            "ablation",
        },
        "candidate",
    )
    candidate_id = _required_text(item.get("id"), context="candidate.id")
    feature_set_id = _required_text(
        item.get("feature_set"),
        context=f"candidate {candidate_id}.feature_set",
    )
    if feature_set_id not in feature_sets:
        raise ValueError(f"candidate {candidate_id!r} references unknown feature set")

    graph = None
    if item.get("graph") is not None:
        graph_item = dict(item["graph"])
        _reject_unknown(
            graph_item,
            {
                "initializer",
                "updater",
                "lifecycle_schema",
                "seed_policy",
                "inner_split_policy",
            },
            f"candidate {candidate_id} graph",
        )
        graph = CandidateGraph(
            initializer_estimator_id=_required_text(
                graph_item.get("initializer"),
                context=f"candidate {candidate_id} graph.initializer",
            ),
            updater_estimator_id=_required_text(
                graph_item.get("updater"),
                context=f"candidate {candidate_id} graph.updater",
            ),
            lifecycle_schema_id=_required_text(
                graph_item.get("lifecycle_schema"),
                context=f"candidate {candidate_id} graph.lifecycle_schema",
            ),
            seed_policy_id=_required_text(
                graph_item.get("seed_policy"),
                context=f"candidate {candidate_id} graph.seed_policy",
            ),
            inner_split_policy_id=_required_text(
                graph_item.get("inner_split_policy"),
                context=f"candidate {candidate_id} graph.inner_split_policy",
            ),
        )

    role = CandidateRole(str(item.get("role") or CandidateRole.MODEL.value))
    ablation = None
    if item.get("ablation") is not None:
        ablation_item = dict(item["ablation"])
        _reject_unknown(
            ablation_item,
            {"reference_candidate_id", "axis", "allowed_config_paths"},
            f"candidate {candidate_id} ablation",
        )
        ablation = AblationSpec(
            reference_candidate_id=_required_text(
                ablation_item.get("reference_candidate_id"),
                context=f"candidate {candidate_id} ablation.reference_candidate_id",
            ),
            axis=AblationAxis(
                _required_text(
                    ablation_item.get("axis"),
                    context=f"candidate {candidate_id} ablation.axis",
                )
            ),
            allowed_config_paths=_strings(ablation_item.get("allowed_config_paths")),
        )

    return CandidateSpec(
        candidate_id=candidate_id,
        estimator_id=_required_text(
            item.get("estimator"),
            context=f"candidate {candidate_id}.estimator",
        ),
        feature_set=feature_sets[feature_set_id],
        params=dict(item.get("params") or {}),
        initializer_params=dict(item.get("initializer_params") or {}),
        graph=graph,
        role=role,
        ablation=ablation,
    )


def project_root_for(config_path: str | Path) -> Path:


    resolved = Path(config_path).resolve()
    for ancestor in resolved.parents:
        if ancestor.name == "configs":
            return ancestor.parent
    return resolved.parent.parent


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).resolve()
    try:
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid project config: {exc}") from exc
    _reject_unknown(
        payload,
        {
            "config_id",
            "schema_version",
            "workspace",
            "seed",
            "collection",
            "split",
            "interval",
            "feature_sets",
            "candidates",
            "experiments",
            "sweeps",
        },
        "top-level",
    )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    config_id = _required_text(payload.get("config_id"), context="config_id")

    collection = dict(payload.get("collection") or {})
    split = dict(payload.get("split") or {})
    interval = dict(payload.get("interval") or {})
    _reject_unknown(
        collection, {"source", "dataset", "descriptor_path", "tick_seconds"}, "collection"
    )
    _reject_unknown(split, {"unit", "folds", "authority_path"}, "split")
    _reject_unknown(interval, {"alpha", "calibrator"}, "interval")

    tick_value = collection.get("tick_seconds")
    if tick_value is None:
        tick_seconds = None
    else:
        if isinstance(tick_value, bool) or not isinstance(tick_value, (int, float)):
            raise ValueError("collection.tick_seconds must be a number")
        tick_seconds = float(tick_value)
        if not tick_seconds > 0:
            raise ValueError("collection.tick_seconds must be positive")

    project_root = project_root_for(source)
    descriptor_path_value = collection.get("descriptor_path")
    descriptor_path = (
        _required_text(descriptor_path_value, context="collection.descriptor_path")
        if descriptor_path_value is not None
        else None
    )
    source_descriptor = (
        _load_source_descriptor(project_root, descriptor_path)
        if descriptor_path is not None
        else None
    )
    authority_path_value = split.get("authority_path")
    partition_authority_path = (
        _required_text(authority_path_value, context="split.authority_path")
        if authority_path_value is not None
        else None
    )
    if partition_authority_path is None:
        partition_authority = None
    else:
        resolved_authority_path = (project_root / partition_authority_path).resolve()
        if not resolved_authority_path.is_file():
            raise ValueError(
                f"Task partition authority does not exist: {partition_authority_path}"
            )
        partition_authority = load_task_partition_authority(resolved_authority_path)
    if source_descriptor is not None:
        declared_source = _required_text(
            collection.get("source"),
            context="collection.source",
        )
        if declared_source != source_descriptor.source_id:
            raise ValueError(
                f"collection.source {declared_source!r} does not match the "
                f"source descriptor source_id {source_descriptor.source_id!r}"
            )
    dataset_mode = _required_text(
        collection.get("dataset", "auto"),
        context="collection.dataset",
    )
    if dataset_mode not in {"auto", "hierarchical"}:
        raise ValueError("collection.dataset must be auto or hierarchical")
    if dataset_mode == "hierarchical" and source_descriptor is None:
        raise ValueError(
            "collection.dataset=hierarchical requires collection.descriptor_path"
        )

    workspace_value = Path(str(payload.get("workspace") or "workspace"))
    workspace = (
        workspace_value if workspace_value.is_absolute() else project_root / workspace_value
    ).resolve()
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    folds = split.get("folds")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 4:
        raise ValueError("split.folds must be an integer of at least 4")
    split_unit = _required_text(split.get("unit"), context="split.unit")
    if split_unit != "task_id":
        raise ValueError("split.unit must be task_id")

    alpha = float(interval.get("alpha", 0.10))
    if not 0 < alpha < 1:
        raise ValueError("interval.alpha must be in (0, 1)")
    calibrator_id = _required_text(
        interval.get("calibrator"),
        context="interval.calibrator",
    )
    if calibrator_id not in registered_calibrator_ids():
        raise ValueError("unsupported interval calibrator")

    raw_feature_sets = payload.get("feature_sets")
    if not isinstance(raw_feature_sets, dict) or not raw_feature_sets:
        raise ValueError("at least one feature set is required")
    feature_sets = {
        str(name): _feature_set(str(name), value) for name, value in raw_feature_sets.items()
    }

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("at least one candidate is required")
    candidates = tuple(_candidate(value, feature_sets=feature_sets) for value in raw_candidates)
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("candidate ids must be unique")

    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("at least one experiment is required")
    experiments: list[ConfiguredExperiment] = []
    for value in raw_experiments:
        if not isinstance(value, dict):
            raise ValueError("experiment entries must be tables")
        item = dict(value)
        _reject_unknown(
            item,
            {
                "id",
                "position",
                "target",
                "candidates",
                "required_features",
                "condition_id",
                "token_unit_contract_id",
                "calibrator",
            },
            "experiment",
        )
        experiment_id = _required_text(item.get("id"), context="experiment.id")
        ids = tuple(str(entry).strip() for entry in item.get("candidates") or ())
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"experiment {experiment_id} requires unique candidates")
        unknown = set(ids) - candidate_ids
        if unknown:
            raise ValueError(
                f"experiment {experiment_id} references unknown candidates: {sorted(unknown)}"
            )
        condition = item.get("condition_id")
        token_unit = item.get("token_unit_contract_id")
        position = PredictionPosition(
            _required_text(item.get("position"), context="experiment.position")
        )
        target = PredictionTarget(_required_text(item.get("target"), context="experiment.target"))
        if position not in target_requirements(target).positions:
            raise ValueError(
                f"experiment {experiment_id} has unsupported position-target pair: "
                f"{position.value}/{target.value}"
            )
        experiments.append(
            ConfiguredExperiment(
                experiment_id=experiment_id,
                position=position,
                target=target,
                candidate_ids=ids,
                required_features=_strings(item.get("required_features")),
                condition_id=(
                    _required_text(condition, context="experiment.condition_id")
                    if condition is not None
                    else None
                ),
                token_unit_contract_id=(
                    _required_text(
                        token_unit,
                        context="experiment.token_unit_contract_id",
                    )
                    if token_unit is not None
                    else None
                ),
                calibrator_id=_experiment_calibrator(item.get("calibrator"), experiment_id),
            )
        )
    if len({item.experiment_id for item in experiments}) != len(experiments):
        raise ValueError("experiment ids must be unique")

    candidates, experiments, sweeps = _expand_sweeps(
        payload.get("sweeps"),
        candidates=candidates,
        experiments=experiments,
    )

    return ProjectConfig(
        config_id=config_id,
        source_path=source,
        schema_version=CONFIG_SCHEMA_VERSION,
        workspace=workspace,
        seed=seed,
        collection_source=_required_text(
            collection.get("source"),
            context="collection.source",
        ),
        dataset_mode=dataset_mode,
        tick_seconds=tick_seconds,
        source_descriptor=source_descriptor,
        source_descriptor_path=descriptor_path,
        split_unit=split_unit,
        folds=folds,
        partition_authority=partition_authority,
        partition_authority_path=partition_authority_path,
        alpha=alpha,
        calibrator_id=calibrator_id,
        feature_sets=feature_sets,
        candidates=candidates,
        experiments=tuple(experiments),
        sweeps=sweeps,
    )


def _expand_sweeps(
    raw_sweeps: Any,
    *,
    candidates: tuple[CandidateSpec, ...],
    experiments: list[ConfiguredExperiment],
) -> tuple[tuple[CandidateSpec, ...], list[ConfiguredExperiment], tuple[SweepSpec, ...]]:
    if raw_sweeps is None:
        return candidates, experiments, ()
    if not isinstance(raw_sweeps, list) or not raw_sweeps:
        raise ValueError("sweeps must be a non-empty list of tables")

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    experiments_by_id = {experiment.experiment_id: experiment for experiment in experiments}
    expanded_candidates = list(candidates)
    sweeps: list[SweepSpec] = []
    for value in raw_sweeps:
        if not isinstance(value, dict):
            raise ValueError("sweep entries must be tables")
        item = dict(value)
        _reject_unknown(
            item,
            {"id", "experiment", "base_candidate", "param", "values"},
            "sweep",
        )
        sweep_id = _required_text(item.get("id"), context="sweep.id")
        experiment_id = _required_text(
            item.get("experiment"),
            context=f"sweep {sweep_id}.experiment",
        )
        experiment = experiments_by_id.get(experiment_id)
        if experiment is None:
            raise ValueError(f"sweep {sweep_id!r} references unknown experiment")
        base_candidate_id = _required_text(
            item.get("base_candidate"),
            context=f"sweep {sweep_id}.base_candidate",
        )
        base = candidates_by_id.get(base_candidate_id)
        if base is None:
            raise ValueError(f"sweep {sweep_id!r} references unknown base candidate")
        if base_candidate_id not in experiment.candidate_ids:
            raise ValueError(
                f"sweep {sweep_id!r} base candidate is not part of its experiment"
            )
        param = _required_text(item.get("param"), context=f"sweep {sweep_id}.param")
        values = item.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep {sweep_id!r} requires a non-empty value list")
        if any(isinstance(entry, bool) for entry in values):
            raise ValueError(f"sweep {sweep_id!r} values must be numbers or strings")
        numeric = all(isinstance(entry, (int, float)) for entry in values)
        textual = all(isinstance(entry, str) for entry in values)
        if not numeric and not textual:
            raise ValueError(f"sweep {sweep_id!r} values must be all numbers or all strings")
        if len(set(values)) != len(values):
            raise ValueError(f"sweep {sweep_id!r} values must be unique")

        points: list[SweepPoint] = []
        derived_ids: list[str] = []
        for index, entry in enumerate(values):
            if base.params.get(param) == entry:
                points.append(SweepPoint(value=entry, candidate_id=base_candidate_id))
                continue
            derived_id = f"{base_candidate_id}--{sweep_id}-v{index}"
            if derived_id in candidates_by_id:
                raise ValueError(f"sweep candidate id {derived_id!r} already exists")
            derived = replace(
                base,
                candidate_id=derived_id,
                params={**base.params, param: entry},
                role=CandidateRole.ABLATION,
                ablation=AblationSpec(
                    reference_candidate_id=base_candidate_id,
                    axis=AblationAxis.METHOD,
                    allowed_config_paths=frozenset({f"params.{param}"}),
                ),
            )
            candidates_by_id[derived_id] = derived
            expanded_candidates.append(derived)
            derived_ids.append(derived_id)
            points.append(SweepPoint(value=entry, candidate_id=derived_id))

        if derived_ids:
            updated = replace(
                experiment,
                candidate_ids=(*experiment.candidate_ids, *derived_ids),
            )
            experiments_by_id[experiment_id] = updated
            experiments[experiments.index(experiment)] = updated
        sweeps.append(
            SweepSpec(
                sweep_id=sweep_id,
                experiment_id=experiment_id,
                base_candidate_id=base_candidate_id,
                param=param,
                points=tuple(points),
            )
        )
    if len({sweep.sweep_id for sweep in sweeps}) != len(sweeps):
        raise ValueError("sweep ids must be unique")
    return tuple(expanded_candidates), experiments, tuple(sweeps)
