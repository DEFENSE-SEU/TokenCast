from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping

from token_prediction.identifiers import assignment_identifier, identifier

from .family_splits import TaskFamilyRegistry


MODEL_FAMILY_REGISTRY_POLICY_ID = "condition_model_family_registry_v1"
MODEL_EVALUATION_SPLIT_POLICY_ID = "task_model_family_evaluation_v1"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _selection(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be an iterable of identifiers")
    return tuple(sorted({_text(value, label=label) for value in values}))


class ModelEvaluationMode(StrEnum):
    UNSEEN_TASK_FAMILY_SEEN_MODEL_FAMILY = "unseen_task_family_seen_model_family"
    SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_ZERO_SHOT = (
        "seen_task_family_unseen_model_family_zero_shot"
    )
    SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_FEW_SHOT = "seen_task_family_unseen_model_family_few_shot"
    JOINT_UNSEEN_TASK_AND_MODEL_FAMILY = "joint_unseen_task_and_model_family"


class ModelEvaluationRole(StrEnum):
    TRAIN = "train"
    SUPPORT = "support"
    QUERY = "query"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ModelFamilyRegistry:


    registry_id: str
    condition_assignments: tuple[tuple[str, str], ...]
    model_assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for label, assignments in (
            ("condition", self.condition_assignments),
            ("model", self.model_assignments),
        ):
            for left, right in assignments:
                _text(left, label=f"{label}_id")
                _text(right, label="model_id" if label == "condition" else "model_family_id")
            keys = [left for left, _right in assignments]
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{label} assignments must use unique canonical order")
        models = frozenset(model_id for model_id, _family_id in self.model_assignments)
        unresolved = {
            model_id
            for _condition_id, model_id in self.condition_assignments
            if model_id not in models
        }
        if unresolved:
            raise ValueError(f"condition assignments contain unknown models: {sorted(unresolved)}")
        expected = _model_registry_id(self.condition_assignments, self.model_assignments)
        if self.registry_id != expected:
            raise ValueError("model-family registry id does not match its canonical mapping")

    @classmethod
    def from_mappings(
        cls,
        condition_to_model: Mapping[str, str],
        model_to_family: Mapping[str, str],
    ) -> ModelFamilyRegistry:
        conditions = tuple(
            sorted(
                (
                    _text(condition_id, label="condition_id"),
                    _text(model_id, label="model_id"),
                )
                for condition_id, model_id in condition_to_model.items()
            )
        )
        models = tuple(
            sorted(
                (
                    _text(model_id, label="model_id"),
                    _text(family_id, label="model_family_id"),
                )
                for model_id, family_id in model_to_family.items()
            )
        )
        if not conditions or not models:
            raise ValueError("condition and model-family mappings must be non-empty")
        return cls(_model_registry_id(conditions, models), conditions, models)

    @property
    def condition_to_model(self) -> dict[str, str]:
        return dict(self.condition_assignments)

    @property
    def model_to_family(self) -> dict[str, str]:
        return dict(self.model_assignments)

    @property
    def family_ids(self) -> frozenset[str]:
        return frozenset(family_id for _model_id, family_id in self.model_assignments)

    def resolve(self, condition_id: str) -> tuple[str, str]:
        condition_id = _text(condition_id, label="condition_id")
        try:
            model_id = self.condition_to_model[condition_id]
        except KeyError as exc:
            raise ValueError(
                f"condition_id is absent from the model-family registry: {condition_id}"
            ) from exc
        return model_id, self.model_to_family[model_id]


def _model_registry_id(
    conditions: tuple[tuple[str, str], ...],
    models: tuple[tuple[str, str], ...],
) -> str:
    assignments = (
        *((f"condition/{condition_id}", model_id) for condition_id, model_id in conditions),
        *((f"model/{model_id}", family_id) for model_id, family_id in models),
    )
    return assignment_identifier(
        "model-family-registry",
        policy_id=MODEL_FAMILY_REGISTRY_POLICY_ID,
        seed=None,
        assignments=assignments,
    )


@dataclass(frozen=True, order=True)
class ModelEvaluationRunKey:
    source_id: str
    task_id: str
    run_id: str
    condition_id: str

    def __post_init__(self) -> None:
        for name in ("source_id", "task_id", "run_id", "condition_id"):
            _text(getattr(self, name), label=name)

    @property
    def authority_id(self) -> str:
        return identifier(
            "trajectory-authority",
            self.source_id,
            self.task_id,
            self.run_id,
            self.condition_id,
        )


@dataclass(frozen=True)
class ModelEvaluationRun:
    source_id: str
    run_id: str
    task_id: str
    task_family_id: str
    condition_id: str
    model_id: str
    model_family_id: str

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "run_id",
            "task_id",
            "task_family_id",
            "condition_id",
            "model_id",
            "model_family_id",
        ):
            _text(getattr(self, name), label=name)

    @property
    def key(self) -> ModelEvaluationRunKey:
        return ModelEvaluationRunKey(
            source_id=self.source_id,
            task_id=self.task_id,
            run_id=self.run_id,
            condition_id=self.condition_id,
        )

    @property
    def split_group_id(self) -> str:


        return self.task_family_id


def _normalize_runs(
    records: Iterable[Mapping[str, object]],
    *,
    task_registry: TaskFamilyRegistry,
    model_registry: ModelFamilyRegistry,
) -> tuple[ModelEvaluationRun, ...]:
    task_to_family = task_registry.task_to_family
    by_run: dict[ModelEvaluationRunKey, ModelEvaluationRun] = {}
    for index, record in enumerate(records):
        run_id = _text(record.get("run_id"), label=f"record {index} run_id")
        task_id = _text(record.get("task_id"), label=f"record {index} task_id")
        condition_id = _text(
            record.get("condition_id"),
            label=f"record {index} condition_id",
        )
        try:
            task_family_id = task_to_family[task_id]
        except KeyError as exc:
            raise ValueError(
                f"record {index} task is absent from the task-family registry"
            ) from exc
        model_id, model_family_id = model_registry.resolve(condition_id)
        declared_task_group = record.get("split_group_id")
        declared_field = "split_group_id"
        if declared_task_group is None and "split_group_id" not in record:
            declared_task_group = record.get("task_family_id")
            declared_field = "task_family_id"
        if declared_task_group is not None and declared_task_group != task_family_id:
            raise ValueError(f"record {index} {declared_field} drifted from the registry")
        declared_model = record.get("model_id")
        if declared_model is not None and declared_model != model_id:
            raise ValueError(f"record {index} model_id drifted from the condition registry")
        declared_model_family = record.get("model_family_id")
        if declared_model_family is not None and declared_model_family != model_family_id:
            raise ValueError(f"record {index} model_family_id drifted from the registry")
        run = ModelEvaluationRun(
            source_id=_text(record.get("source_id"), label=f"record {index} source_id"),
            run_id=run_id,
            task_id=task_id,
            task_family_id=task_family_id,
            condition_id=condition_id,
            model_id=model_id,
            model_family_id=model_family_id,
        )
        key = run.key
        previous = by_run.get(key)
        if previous is not None and previous != run:
            raise ValueError(
                f"trajectory authority {key.authority_id!r} has inconsistent row metadata"
            )
        by_run[key] = run
    if not by_run:
        raise ValueError("model evaluation requires at least one run")
    return tuple(sorted(by_run.values(), key=lambda run: run.key))


def _role_for(
    mode: ModelEvaluationMode,
    run: ModelEvaluationRun,
    *,
    unseen_task_families: frozenset[str],
    held_out_model_family_id: str,
    support_task_families: frozenset[str],
) -> ModelEvaluationRole:
    unseen_task = run.task_family_id in unseen_task_families
    held_out_model = run.model_family_id == held_out_model_family_id
    if mode == ModelEvaluationMode.UNSEEN_TASK_FAMILY_SEEN_MODEL_FAMILY:
        if held_out_model:
            return ModelEvaluationRole.EXCLUDED
        return ModelEvaluationRole.QUERY if unseen_task else ModelEvaluationRole.TRAIN
    if mode == ModelEvaluationMode.SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_ZERO_SHOT:
        if unseen_task:
            return ModelEvaluationRole.EXCLUDED
        return ModelEvaluationRole.QUERY if held_out_model else ModelEvaluationRole.TRAIN
    if mode == ModelEvaluationMode.SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_FEW_SHOT:
        if unseen_task:
            return ModelEvaluationRole.EXCLUDED
        if not held_out_model:
            return ModelEvaluationRole.TRAIN
        if run.task_family_id in support_task_families:
            return ModelEvaluationRole.SUPPORT
        return ModelEvaluationRole.QUERY
    if unseen_task and held_out_model:
        return ModelEvaluationRole.QUERY
    if not unseen_task and not held_out_model:
        return ModelEvaluationRole.TRAIN
    return ModelEvaluationRole.EXCLUDED


def _split_id(
    *,
    dataset_id: str,
    mode: ModelEvaluationMode,
    task_registry: TaskFamilyRegistry,
    model_registry: ModelFamilyRegistry,
    unseen_task_families: tuple[str, ...],
    held_out_model_family_id: str,
    support_task_families: tuple[str, ...],
    source_ids: tuple[str, ...],
) -> str:
    return identifier(
        "model-evaluation-split",
        MODEL_EVALUATION_SPLIT_POLICY_ID,
        dataset_id,
        task_registry.registry_id,
        model_registry.registry_id,
        mode.value,
        f"unseen-task={'+'.join(unseen_task_families)}",
        f"held-out-model={held_out_model_family_id}",
        f"support-task={'+'.join(support_task_families) or 'none'}",
        f"sources={'+'.join(source_ids)}",
    )


@dataclass(frozen=True)
class ModelEvaluationSplit:
    split_id: str
    dataset_id: str
    mode: ModelEvaluationMode
    task_registry: TaskFamilyRegistry
    model_registry: ModelFamilyRegistry
    unseen_task_families: tuple[str, ...]
    held_out_model_family_id: str
    support_task_families: tuple[str, ...]
    runs: tuple[ModelEvaluationRun, ...]
    assignments: tuple[tuple[ModelEvaluationRunKey, ModelEvaluationRole], ...]

    def __post_init__(self) -> None:
        _text(self.dataset_id, label="dataset_id")
        _text(self.held_out_model_family_id, label="held_out_model_family_id")
        if not isinstance(self.mode, ModelEvaluationMode):
            raise TypeError("mode must be a ModelEvaluationMode")
        if self.runs != tuple(sorted(self.runs, key=lambda run: run.key)):
            raise ValueError("model evaluation runs must use canonical run order")
        run_keys = [run.key for run in self.runs]
        if len(run_keys) != len(set(run_keys)):
            raise ValueError("model evaluation run authorities must be unique")
        unseen = frozenset(self.unseen_task_families)
        support = frozenset(self.support_task_families)
        if self.unseen_task_families != tuple(sorted(unseen)):
            raise ValueError("unseen Task families must use unique canonical order")
        if self.support_task_families != tuple(sorted(support)):
            raise ValueError("support Task families must use unique canonical order")
        observed_tasks = frozenset(run.task_family_id for run in self.runs)
        observed_models = frozenset(run.model_family_id for run in self.runs)
        if not unseen or not unseen < observed_tasks:
            raise ValueError("unseen Task families must be a non-empty proper observed subset")
        seen_tasks = observed_tasks - unseen
        if not support or not support < seen_tasks:
            raise ValueError("support Task families must be a non-empty proper seen subset")
        if self.held_out_model_family_id not in observed_models or len(observed_models) < 2:
            raise ValueError("held-out model family requires at least one observed seen family")
        expected_assignments = tuple(
            (
                run.key,
                _role_for(
                    self.mode,
                    run,
                    unseen_task_families=unseen,
                    held_out_model_family_id=self.held_out_model_family_id,
                    support_task_families=support,
                ),
            )
            for run in self.runs
        )
        if self.assignments != expected_assignments:
            raise ValueError("model evaluation assignments violate the frozen family policy")
        expected_id = _split_id(
            dataset_id=self.dataset_id,
            mode=self.mode,
            task_registry=self.task_registry,
            model_registry=self.model_registry,
            unseen_task_families=self.unseen_task_families,
            held_out_model_family_id=self.held_out_model_family_id,
            support_task_families=self.support_task_families,
            source_ids=tuple(sorted({run.source_id for run in self.runs})),
        )
        if self.split_id != expected_id:
            raise ValueError("model evaluation split id does not match its authority")
        self._validate_roles()

    def _validate_roles(self) -> None:
        if not self.train_run_keys or not self.query_run_keys:
            raise ValueError("model evaluation requires non-empty train and query runs")
        held_out = self.held_out_model_family_id
        if held_out in self.train_model_family_ids:
            raise ValueError("held-out model family entered training")
        if self.mode == ModelEvaluationMode.UNSEEN_TASK_FAMILY_SEEN_MODEL_FAMILY:
            if self.train_task_family_ids & self.query_task_family_ids:
                raise ValueError("unseen Task families entered training")
            if not self.query_model_family_ids <= self.train_model_family_ids:
                raise ValueError("unseen-Task query uses a model family absent from training")
        elif self.mode == ModelEvaluationMode.SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_ZERO_SHOT:
            if self.support_run_keys:
                raise ValueError("zero-shot evaluation cannot contain support runs")
            if not self.query_task_family_ids <= self.train_task_family_ids:
                raise ValueError("zero-shot query contains Task families absent from training")
            if self.query_model_family_ids != {held_out}:
                raise ValueError("zero-shot query must use the held-out model family")
        elif self.mode == ModelEvaluationMode.SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_FEW_SHOT:
            if not self.support_run_keys:
                raise ValueError("few-shot evaluation requires support runs")
            if self.support_task_family_ids != frozenset(self.support_task_families):
                raise ValueError("few-shot support does not cover its frozen Task families")
            if self.support_task_family_ids & self.query_task_family_ids:
                raise ValueError("few-shot support Task families entered query")
            if self.support_model_family_ids != {held_out} or self.query_model_family_ids != {
                held_out
            }:
                raise ValueError("few-shot support and query must use the held-out model family")
            if (
                not (self.support_task_family_ids | self.query_task_family_ids)
                <= self.train_task_family_ids
            ):
                raise ValueError("few-shot tasks are absent from base-model training")
        else:
            if self.train_task_family_ids & self.query_task_family_ids:
                raise ValueError("joint query Task families entered training")
            if self.train_model_family_ids & self.query_model_family_ids:
                raise ValueError("joint query model families entered training")
            if self.query_model_family_ids != {held_out}:
                raise ValueError("joint query must use the held-out model family")
        if self.mode != ModelEvaluationMode.SEEN_TASK_FAMILY_UNSEEN_MODEL_FAMILY_FEW_SHOT:
            if self.support_run_keys:
                raise ValueError("support runs belong only to the few-shot identity")

    @property
    def role_by_run(self) -> dict[ModelEvaluationRunKey, ModelEvaluationRole]:
        return dict(self.assignments)

    def runs_in(self, role: ModelEvaluationRole) -> tuple[ModelEvaluationRun, ...]:
        roles = self.role_by_run
        return tuple(run for run in self.runs if roles[run.key] == role)

    def _values(self, role: ModelEvaluationRole, name: str) -> frozenset[str]:
        return frozenset(getattr(run, name) for run in self.runs_in(role))

    @property
    def train_run_keys(self) -> frozenset[ModelEvaluationRunKey]:
        return frozenset(run.key for run in self.runs_in(ModelEvaluationRole.TRAIN))

    @property
    def support_run_keys(self) -> frozenset[ModelEvaluationRunKey]:
        return frozenset(run.key for run in self.runs_in(ModelEvaluationRole.SUPPORT))

    @property
    def query_run_keys(self) -> frozenset[ModelEvaluationRunKey]:
        return frozenset(run.key for run in self.runs_in(ModelEvaluationRole.QUERY))

    @property
    def train_task_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.TRAIN, "task_id")

    @property
    def support_task_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.SUPPORT, "task_id")

    @property
    def query_task_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.QUERY, "task_id")

    @property
    def train_task_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.TRAIN, "task_family_id")

    @property
    def support_task_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.SUPPORT, "task_family_id")

    @property
    def query_task_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.QUERY, "task_family_id")

    @property
    def train_model_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.TRAIN, "model_family_id")

    @property
    def support_model_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.SUPPORT, "model_family_id")

    @property
    def query_model_family_ids(self) -> frozenset[str]:
        return self._values(ModelEvaluationRole.QUERY, "model_family_id")

    def audit_record(self) -> dict[str, object]:
        def role_record(role: ModelEvaluationRole) -> dict[str, object]:
            runs = self.runs_in(role)
            return {
                "runs": [
                    {
                        "trajectory_authority_id": run.key.authority_id,
                        "source_id": run.source_id,
                        "task_id": run.task_id,
                        "run_id": run.run_id,
                        "condition_id": run.condition_id,
                    }
                    for run in runs
                ],
                "source_ids": sorted({run.source_id for run in runs}),
                "task_ids": sorted({run.task_id for run in runs}),
                "task_family_ids": sorted({run.task_family_id for run in runs}),
                "model_ids": sorted({run.model_id for run in runs}),
                "model_family_ids": sorted({run.model_family_id for run in runs}),
            }

        return {
            "split_id": self.split_id,
            "mode": self.mode.value,
            "train": role_record(ModelEvaluationRole.TRAIN),
            "support": role_record(ModelEvaluationRole.SUPPORT),
            "query": role_record(ModelEvaluationRole.QUERY),
            "excluded": role_record(ModelEvaluationRole.EXCLUDED),
        }

    def validate_records(self, records: Iterable[Mapping[str, object]]) -> None:
        observed = _normalize_runs(
            records,
            task_registry=self.task_registry,
            model_registry=self.model_registry,
        )
        if observed != self.runs:
            raise ValueError("model evaluation records differ from the frozen run roster")


@dataclass(frozen=True)
class ModelEvaluationSuite:
    suite_id: str
    dataset_id: str
    splits: tuple[ModelEvaluationSplit, ...]

    def __post_init__(self) -> None:
        expected_modes = tuple(ModelEvaluationMode)
        if tuple(split.mode for split in self.splits) != expected_modes:
            raise ValueError("model evaluation suite must contain the four canonical identities")
        if any(split.dataset_id != self.dataset_id for split in self.splits):
            raise ValueError("model evaluation suite mixes dataset identities")
        if len({split.split_id for split in self.splits}) != len(self.splits):
            raise ValueError("model evaluation split identities must be unique")
        authority = self.splits[0]
        shared = (
            "task_registry",
            "model_registry",
            "unseen_task_families",
            "held_out_model_family_id",
            "support_task_families",
            "runs",
        )
        if any(
            getattr(split, name) != getattr(authority, name)
            for split in self.splits[1:]
            for name in shared
        ):
            raise ValueError("model evaluation identities do not share one frozen authority")
        expected = _suite_id(self.dataset_id, self.splits)
        if self.suite_id != expected:
            raise ValueError("model evaluation suite id does not match its splits")

    @property
    def by_mode(self) -> dict[ModelEvaluationMode, ModelEvaluationSplit]:
        return {split.mode: split for split in self.splits}

    def audit_record(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "dataset_id": self.dataset_id,
            "splits": [split.audit_record() for split in self.splits],
        }

    def validate_records(self, records: Iterable[Mapping[str, object]]) -> None:
        retained = tuple(records)
        for split in self.splits:
            split.validate_records(retained)


def _suite_id(
    dataset_id: str,
    splits: tuple[ModelEvaluationSplit, ...],
) -> str:
    authority = splits[0]
    return identifier(
        "model-evaluation-suite",
        MODEL_EVALUATION_SPLIT_POLICY_ID,
        dataset_id,
        authority.task_registry.registry_id,
        authority.model_registry.registry_id,
        f"unseen-task={'+'.join(authority.unseen_task_families)}",
        f"held-out-model={authority.held_out_model_family_id}",
        f"support-task={'+'.join(authority.support_task_families)}",
        f"sources={'+'.join(sorted({run.source_id for run in authority.runs}))}",
        f"modes={'+'.join(mode.value for mode in ModelEvaluationMode)}",
    )


def build_model_evaluation_suite(
    records: Iterable[Mapping[str, object]],
    *,
    dataset_id: str,
    task_to_family: Mapping[str, str],
    condition_to_model: Mapping[str, str],
    model_to_family: Mapping[str, str],
    unseen_task_families: Iterable[str],
    held_out_model_family_id: str,
    support_task_families: Iterable[str],
) -> ModelEvaluationSuite:


    dataset_id = _text(dataset_id, label="dataset_id")
    task_registry = TaskFamilyRegistry.from_split_group_mapping(task_to_family)
    model_registry = ModelFamilyRegistry.from_mappings(condition_to_model, model_to_family)
    runs = _normalize_runs(
        records,
        task_registry=task_registry,
        model_registry=model_registry,
    )
    observed_task_families = frozenset(run.task_family_id for run in runs)
    observed_model_families = frozenset(run.model_family_id for run in runs)
    unseen = _selection(unseen_task_families, label="unseen_task_family_id")
    support = _selection(support_task_families, label="support_task_family_id")
    held_out = _text(held_out_model_family_id, label="held_out_model_family_id")
    if not unseen or not frozenset(unseen) < observed_task_families:
        raise ValueError("unseen Task families must be a non-empty proper observed subset")
    seen_task_families = observed_task_families - frozenset(unseen)
    if not support or not frozenset(support) < seen_task_families:
        raise ValueError("support Task families must be a non-empty proper seen subset")
    if held_out not in observed_model_families or len(observed_model_families) < 2:
        raise ValueError("held-out model family requires at least one observed seen family")

    source_ids = tuple(sorted({run.source_id for run in runs}))
    splits: list[ModelEvaluationSplit] = []
    for mode in ModelEvaluationMode:
        assignments = tuple(
            (
                run.key,
                _role_for(
                    mode,
                    run,
                    unseen_task_families=frozenset(unseen),
                    held_out_model_family_id=held_out,
                    support_task_families=frozenset(support),
                ),
            )
            for run in runs
        )
        splits.append(
            ModelEvaluationSplit(
                split_id=_split_id(
                    dataset_id=dataset_id,
                    mode=mode,
                    task_registry=task_registry,
                    model_registry=model_registry,
                    unseen_task_families=unseen,
                    held_out_model_family_id=held_out,
                    support_task_families=support,
                    source_ids=source_ids,
                ),
                dataset_id=dataset_id,
                mode=mode,
                task_registry=task_registry,
                model_registry=model_registry,
                unseen_task_families=unseen,
                held_out_model_family_id=held_out,
                support_task_families=support,
                runs=runs,
                assignments=assignments,
            )
        )
    frozen_splits = tuple(splits)
    return ModelEvaluationSuite(_suite_id(dataset_id, frozen_splits), dataset_id, frozen_splits)


__all__ = [
    "MODEL_EVALUATION_SPLIT_POLICY_ID",
    "MODEL_FAMILY_REGISTRY_POLICY_ID",
    "ModelEvaluationMode",
    "ModelEvaluationRole",
    "ModelEvaluationRun",
    "ModelEvaluationRunKey",
    "ModelEvaluationSplit",
    "ModelEvaluationSuite",
    "ModelFamilyRegistry",
    "build_model_evaluation_suite",
]
