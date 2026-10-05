from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from token_prediction.experiments.crossfit import (
    SUPPORTED_SEED_POLICY_IDS,
    InitializerComponent,
    generate_crossfit_seeds,
)
from token_prediction.dataset import (
    INNER_FOLD_POLICY_ID,
    LIFECYCLE_SCHEMA_VERSION,
    DatasetRow,
    DatasetSlice,
    InnerFoldPartition,
    LabelStatus,
    LifecycleSequence,
    LifecycleSlice,
    PredictionPoint,
    PredictionPosition,
    PredictionTarget,
    SplitPlan,
    SupervisedDataset,
    supported_input_contract_ids_from_capability,
    target_supervised_input_contract_id,
)
from token_prediction.dataset.development import OuterInnerPlan
from token_prediction.estimators import (
    EstimatorRegistry,
    FitCheckpoint,
    FitContext,
    ObservedTransition,
    SessionSeed,
    TokenForecast,
    TrainingExample,
    TrainingView,
)
from token_prediction.estimators.gru_bundle import GRU_BUNDLE_FORMAT
from token_prediction.evaluation import (
    METRIC_SUITE_ID,
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    evaluate_task_forecasts,
    fit_calibrator,
)
from token_prediction.experiments.candidates import (
    CandidateResult,
    CandidateSpec,
    FoldArtifact,
    PredictionRecord,
    _collect_fold_artifact,
)
from token_prediction.features import FeatureSet
from token_prediction.identifiers import identifier
from token_prediction.lifecycle.driver import run_lifecycle_batch
from token_prediction.lifecycle.bundle import (
    CROSS_POSITION_DEDUCT_FORMAT,
    EMPIRICAL_INITIALIZER_FORMAT,
    LIFECYCLE_COMPONENT_SCHEMA_VERSION,
    LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION,
    feature_set_document,
    load_lifecycle_bundle,
    validate_source_provenance,
)


TASK_LIFECYCLE_SCHEMA_ID = f"task_lifecycle_v{LIFECYCLE_SCHEMA_VERSION}"
_PROTOCOL_FEATURES = frozenset(
    {
        "missing_usage_attempts",
        "cumulative_provider_input_tokens",
        "cumulative_provider_output_tokens",
    }
)


def _fit_seed(seed: int, *, outer_fold: int, inner_fold: int | None) -> int:


    inner_part = 0 if inner_fold is None else inner_fold + 1
    return (seed * 1_000_003 + (outer_fold + 1) * 10_007 + inner_part * 1_009) % (2**31 - 1)


def _selected_point(point: PredictionPoint, feature_set: FeatureSet) -> PredictionPoint:
    return point.with_features(feature_set.select(point.features))


def _lifecycle_point(point: PredictionPoint, feature_set: FeatureSet) -> PredictionPoint:
    features = feature_set.select(point.features)
    for name in _PROTOCOL_FEATURES:
        if name in point.features:
            features[name] = point.features[name]
    return point.with_features(features)


def _selected_sequence(
    sequence: LifecycleSequence,
    feature_set: FeatureSet,
) -> LifecycleSequence:
    steps = tuple(
        replace(step, point=_lifecycle_point(step.point, feature_set)) for step in sequence.steps
    )
    steps = (
        replace(
            steps[0],
            label=None,
            status=LabelStatus.MISSING,
            invalid_reason="redacted_task_seed_label",
        ),
        *steps[1:],
    )
    return replace(sequence, steps=steps)


@dataclass
class _SelectedSession:
    delegate: Any
    feature_set: FeatureSet

    def predict(self, point: PredictionPoint) -> TokenForecast:
        return self.delegate.predict(_selected_point(point, self.feature_set))

    def observe(self, transition: ObservedTransition) -> None:
        self.delegate.observe(transition)


@dataclass(frozen=True)
class _SelectedFitted:
    estimator_id: str
    delegate: Any
    feature_set: FeatureSet

    def start(self, context: Any) -> _SelectedSession:
        return _SelectedSession(self.delegate.start(context), self.feature_set)


@dataclass(frozen=True)
class SeededLifecycleTrainingSequence:


    sequence: LifecycleSequence
    session_seed: SessionSeed

    def __post_init__(self) -> None:
        first = self.sequence.steps[0].point
        seed_point = self.session_seed.task_seed_point
        identity = (
            "point_id",
            "task_id",
            "trajectory_id",
            "run_id",
            "condition_id",
            "target",
        )
        if any(getattr(first, name) != getattr(seed_point, name) for name in identity):
            raise ValueError("updater lifecycle sequence does not match its SessionSeed")

    @property
    def seed(self) -> SessionSeed:
        return self.session_seed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sequence, name)


@dataclass(frozen=True)
class _PackedInitializer:
    component: InitializerComponent
    state: Mapping[str, Any]


def _make_initializer_view(
    dataset_slice: DatasetSlice,
    rows: Sequence[DatasetRow],
    weights: Mapping[str, float],
    feature_set: FeatureSet,
    *,
    dataset_id: str,
    partition_name: str,
) -> TrainingView:
    examples = tuple(
        TrainingExample(
            point=_selected_point(row.point, feature_set),
            target_value=float(row.label),
            sample_weight=weights[row.point.point_id],
        )
        for row in sorted(rows, key=lambda item: item.point.point_id)
        if row.label is not None
    )
    if not examples:
        raise ValueError(f"initializer {partition_name} partition is empty")
    return TrainingView(
        dataset_id=dataset_id,
        position=PredictionPosition.TASK_SEED,
        target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        examples=examples,
        input_contract_id=dataset_slice.input_contract_id,
        token_unit_contract_id=dataset_slice.token_unit_contract_id,
    )


def _make_updater_view(
    *,
    dataset_id: str,
    target: PredictionTarget,
    sequences: Sequence[LifecycleSequence],
    seeds: Mapping[str, SessionSeed],
    feature_set: FeatureSet,
    partition_name: str,
    input_contract_id: str,
    token_unit_contract_id: str | None,
) -> TrainingView:
    if not sequences:
        raise ValueError(f"updater {partition_name} lifecycle partition is empty")
    selected = tuple(
        _selected_sequence(sequence, feature_set)
        for sequence in sorted(
            sequences,
            key=lambda item: (item.task_id, item.run_id, item.trajectory_id),
        )
    )
    expected_seed_ids = {sequence.steps[0].point.point_id for sequence in selected}
    if set(seeds) != expected_seed_ids:
        raise ValueError(f"updater {partition_name} seeds do not cover its sequences")
    seeded = tuple(
        SeededLifecycleTrainingSequence(
            sequence,
            seeds[sequence.steps[0].point.point_id],
        )
        for sequence in selected
    )
    examples = tuple(
        TrainingExample(step.point, float(step.label), step.sample_weight)
        for sequence in seeded
        for step in sequence.steps[1:]
        if step.loss_mask and step.label is not None
    )
    if not examples:
        raise ValueError(f"updater {partition_name} scored partition is empty")
    return TrainingView(
        dataset_id=dataset_id,
        position=PredictionPosition.TASK_UPDATE,
        target=target,
        examples=examples,
        lifecycle_sequences=seeded,
        input_contract_id=input_contract_id,
        token_unit_contract_id=token_unit_contract_id,
    )


def _partition_sequences(
    lifecycle_slice: LifecycleSlice,
    tasks: frozenset[str],
) -> tuple[LifecycleSequence, ...]:
    return tuple(sequence for sequence in lifecycle_slice.sequences if sequence.task_id in tasks)


def _scored_predictions(runs: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(prediction for run in runs for prediction in run.scored_predictions)


def _fit_calibrator(
    calibrator_id: str,
    *,
    alpha: float,
    examples: Sequence[CalibrationExample],
) -> Any:
    return fit_calibrator(calibrator_id, alpha=alpha, examples=examples)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _empirical_state(fitted: Any) -> dict[str, Any]:
    delegate = getattr(fitted, "delegate", fitted)
    try:
        state = {
            "estimator_id": str(delegate.estimator_id),
            "target": delegate.target.value,
            "lower": float(delegate.lower),
            "point": float(delegate.point),
            "upper": float(delegate.upper),
            "token_unit_contract_id": delegate.token_unit_contract_id,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("empirical initializer does not expose portable quantiles") from exc
    if not 0 <= state["lower"] <= state["point"] <= state["upper"]:
        raise ValueError("empirical initializer quantiles are invalid")
    return state


def _pack_initializer(
    fitted: Any,
    *,
    initializer_id: str,
    initializer_revision: str,
    inner_fold: int,
    partition: InnerFoldPartition,
    feature_set: FeatureSet,
) -> _PackedInitializer:
    component_id = identifier(
        "initializer-component",
        initializer_revision,
        f"inner-fold={inner_fold}",
        f"fit={'+'.join(sorted(partition.initializer_fit_tasks))}",
        f"validation={'+'.join(sorted(partition.validation_tasks))}",
        f"holdout={'+'.join(sorted(partition.holdout_tasks))}",
    )
    model_revision = identifier(
        "model",
        initializer_id,
        initializer_revision,
        f"inner-fold={inner_fold}",
    )
    artifact_id = identifier("initializer-artifact", component_id)
    component = InitializerComponent(
        inner_fold=inner_fold,
        component_id=component_id,
        model_revision=model_revision,
        bundle_artifact_ids=(artifact_id,),
        fit_tasks=partition.initializer_fit_tasks,
        validation_tasks=partition.validation_tasks,
        holdout_tasks=partition.holdout_tasks,
        fitted=_SelectedFitted(initializer_id, fitted, feature_set),
    )
    return _PackedInitializer(component, _empirical_state(fitted))


def _bundle_artifact(
    fitted_updater: Any,
    *,
    fold: int,
    candidate: CandidateSpec,
    dataset: SupervisedDataset,
    update_slice: DatasetSlice,
    lifecycle_slice: LifecycleSlice,
    split_plan: SplitPlan,
    outer_tasks: Mapping[str, frozenset[str]],
    inner_plan: OuterInnerPlan,
    initializer_revision: str,
    packed_initializers: Sequence[_PackedInitializer],
    seed_set_id: str,
    calibrator_id: str,
    alpha: float,
    calibrator_document: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> FoldArtifact:
    updater_provenance = {
        "role": "lifecycle-updater",
        "candidate_id": candidate.candidate_id,
        "model_revision": candidate.model_revision,
        "dataset_id": dataset.dataset_id,
        "split_plan_id": split_plan.split_plan_id,
        "eligibility_policy_id": update_slice.eligibility_policy_id,
        "lifecycle_context_id": lifecycle_slice.context_id,
        "lifecycle_scoring_policy_id": lifecycle_slice.scoring_policy_id,
        "outer_fold": fold,
        "initializer_revision": initializer_revision,
        "inner_split_id": inner_plan.assignment.assignment_id,
        "seed_set_id": seed_set_id,
        "interval_alpha": alpha,
        "calibrator_id": calibrator_id,
        "calibration_weighting_id": update_slice.weighting_id,
        "token_unit_contract_id": lifecycle_slice.token_unit_contract_id,
    }
    updater_artifact = _collect_fold_artifact(
        fitted_updater,
        fold=fold,
        calibrator=calibrator_document,
        provenance=updater_provenance,
    )
    files: dict[str, bytes] = {"calibrator.json": _json_bytes(calibrator_document)}
    initializer_records: list[dict[str, Any]] = []
    for packed in sorted(
        packed_initializers,
        key=lambda item: item.component.inner_fold,
    ):
        state_path = f"initializers/{packed.component.inner_fold}/state.json"
        files[state_path] = _json_bytes(dict(packed.state))
        initializer_records.append(
            {
                "inner_fold": packed.component.inner_fold,
                "component_id": packed.component.component_id,
                "model_revision": packed.component.model_revision,
                "artifact_ids": list(packed.component.bundle_artifact_ids),
                "state_path": state_path,
                "format": EMPIRICAL_INITIALIZER_FORMAT,
            }
        )

    if candidate.estimator_id == "cross_position_deduct":
        updater_path = "updater/state.json"
        files[updater_path] = _json_bytes(
            {
                "estimator_id": fitted_updater.estimator_id,
                "model_revision": fitted_updater.model_revision,
                "token_unit_contract_id": fitted_updater.token_unit_contract_id,
            }
        )
        updater_record = {
            "format": CROSS_POSITION_DEDUCT_FORMAT,
            "model_revision": fitted_updater.model_revision,
            "state_path": updater_path,
        }
    elif candidate.estimator_id == "gru_residual":
        if updater_artifact is None or updater_artifact.bundle_files is None:
            raise ValueError("GRU updater did not produce a portable bundle")
        prefix = "updater"
        for name, payload in updater_artifact.bundle_files.items():
            files[f"{prefix}/{name}"] = payload
        updater_record = {
            "format": GRU_BUNDLE_FORMAT,
            "model_revision": fitted_updater.fit_report.model_revision,
            "prefix": prefix,
            "provenance": updater_provenance,
        }
    else:
        raise ValueError("unsupported lifecycle updater")

    graph = candidate.graph
    if graph is None:
        raise AssertionError("candidate graph is missing")
    manifest = {
        "bundle_schema_version": LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION,
        "component_schema_version": LIFECYCLE_COMPONENT_SCHEMA_VERSION,
        "bundle_format": "lifecycle-composite",
        "bundle_id": identifier(
            "lifecycle-bundle",
            candidate.model_revision,
            split_plan.split_plan_id,
            f"fold={fold}",
        ),
        "candidate_id": candidate.candidate_id,
        "candidate_graph": graph.to_dict(),
        "model_revision": candidate.model_revision,
        "dataset_id": dataset.dataset_id,
        "dataset_schema_version": dataset.schema_version,
        "source_descriptor_id": lifecycle_slice.source_descriptor_id,
        "capability_profile_id": lifecycle_slice.capability_profile_id,
        "input_contract_id": lifecycle_slice.input_contract_id,
        "token_unit_contract_id": lifecycle_slice.token_unit_contract_id,
        "split_plan_id": split_plan.split_plan_id,
        "outer_fold": fold,
        "outer_partitions": {name: sorted(tasks) for name, tasks in sorted(outer_tasks.items())},
        "inner_split_id": inner_plan.assignment.assignment_id,
        "inner_assignments": [
            {"task_id": task_id, "fold": inner_fold}
            for task_id, inner_fold in inner_plan.assignment.assignments
        ],
        "eligibility_policy_id": update_slice.eligibility_policy_id,
        "lifecycle_context_id": lifecycle_slice.context_id,
        "lifecycle_scoring_policy_id": lifecycle_slice.scoring_policy_id,
        "condition_id": lifecycle_slice.condition_id,
        "target": lifecycle_slice.target.value,
        "feature_set": dict(feature_set_document(candidate.feature_set)),
        "initializer_id": graph.initializer_estimator_id,
        "initializer_revision": initializer_revision,
        "seed_policy_id": graph.seed_policy_id,
        "seed_set_id": seed_set_id,
        "initializers": initializer_records,
        "updater": updater_record,
        "calibrator_id": calibrator_id,
        "calibration_weighting_id": update_slice.weighting_id,
        "interval_alpha": alpha,
        "calibrator_path": "calibrator.json",
        "provenance": dict(source_provenance),
    }
    files["manifest.json"] = _json_bytes(manifest)
    return FoldArtifact(
        fold=fold,
        encoder=None if updater_artifact is None else updater_artifact.encoder,
        fit_report=None if updater_artifact is None else updater_artifact.fit_report,
        feature_importance=(
            None if updater_artifact is None else updater_artifact.feature_importance
        ),
        model_strings=None if updater_artifact is None else updater_artifact.model_strings,
        bundle_files=files,
        calibrator=calibrator_document,
        provenance=manifest,
    )


def _validate_inputs(
    dataset: SupervisedDataset,
    lifecycle_slice: LifecycleSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    *,
    alpha: float,
    seed: int,
) -> tuple[DatasetSlice, dict[str, float]]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if split_plan.dataset_id != dataset.dataset_id:
        raise ValueError("split plan belongs to another dataset")
    if split_plan.folds != 5:
        raise ValueError("lifecycle CV requires five outer folds")
    if split_plan.seed != seed:
        raise ValueError("lifecycle model seed must match the outer split seed")
    if lifecycle_slice.dataset_id != dataset.dataset_id:
        raise ValueError("lifecycle slice belongs to another dataset")
    dataset_ids = (
        dataset.source_descriptor_id,
        dataset.capability_profile_id,
        dataset.input_contract_id,
    )
    slice_ids = (
        lifecycle_slice.source_descriptor_id,
        lifecycle_slice.capability_profile_id,
        lifecycle_slice.input_contract_id,
    )
    if dataset_ids != slice_ids:
        raise ValueError("lifecycle slice identity differs from its dataset")
    supported_contracts = set(
        supported_input_contract_ids_from_capability(
            lifecycle_slice.capability_profile_id
        )
    )
    if lifecycle_slice.source_descriptor_id is not None:
        supported_contracts.add(
            target_supervised_input_contract_id(
                lifecycle_slice.source_descriptor_id,
                lifecycle_slice.capability_profile_id,
            )
        )
    if lifecycle_slice.input_contract_id not in supported_contracts:
        raise ValueError("lifecycle input contract differs from its capability profile")
    if lifecycle_slice.target != PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS:
        raise ValueError("lifecycle CV requires the provider-accounted Task target")
    graph = candidate.graph
    if graph is None or not graph.is_lifecycle:
        raise ValueError("lifecycle CV requires an initializer/updater candidate graph")
    if graph.lifecycle_schema_id != TASK_LIFECYCLE_SCHEMA_ID:
        raise ValueError("candidate lifecycle schema does not match the runtime")
    if graph.seed_policy_id not in SUPPORTED_SEED_POLICY_IDS:
        raise ValueError("candidate seed policy is unsupported")
    if graph.inner_split_policy_id != INNER_FOLD_POLICY_ID:
        raise ValueError("candidate inner split policy does not match the runtime")
    if (
        graph.initializer_estimator_id != "empirical_quantile"
        or graph.updater_estimator_id not in {"cross_position_deduct", "gru_residual"}
        or candidate.estimator_id != graph.updater_estimator_id
    ):
        raise ValueError("lifecycle CV requires an empirical initializer and supported updater")
    split_plan.validate_tasks(lifecycle_slice.task_ids, require_exact=False)

    update_slice = dataset.select(
        PredictionPosition.TASK_UPDATE,
        lifecycle_slice.target,
        condition_id=lifecycle_slice.condition_id,
        token_unit_contract_id=lifecycle_slice.token_unit_contract_id,
    )
    expected = {row.point.point_id for row in update_slice.rows}
    scored = {step.point.point_id for step in lifecycle_slice.scored_steps}
    if not expected or scored != expected:
        raise ValueError("lifecycle scored cohort differs from Task-update eligibility")
    weights = {item.row.point.point_id: item.sample_weight for item in update_slice.weighted_rows()}
    for step in lifecycle_slice.scored_steps:
        if not math.isclose(
            step.sample_weight,
            weights[step.point.point_id],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("lifecycle weights differ from the eligible point cohort")
    return update_slice, weights


def _validated_inner_plans(
    split_plan: SplitPlan,
    inner_plans: Mapping[int, OuterInnerPlan],
) -> dict[int, OuterInnerPlan]:
    expected_folds = set(range(split_plan.folds))
    if not isinstance(inner_plans, Mapping) or set(inner_plans) != expected_folds:
        raise ValueError("inner_plans must exactly cover every outer fold")
    result: dict[int, OuterInnerPlan] = {}
    for outer_fold in range(split_plan.folds):
        plan = inner_plans[outer_fold]
        if not isinstance(plan, OuterInnerPlan):
            raise TypeError("inner plan values must be OuterInnerPlan instances")
        if (
            plan.outer_test_fold != outer_fold
            or plan.split_seed != split_plan.seed
            or plan.outer_split_plan_id != split_plan.split_plan_id
        ):
            raise ValueError("inner plan differs from its outer split")
        if plan.assignment.task_ids != split_plan.partition(outer_fold).train_tasks:
            raise ValueError("inner plan task universe must equal the outer-train partition")
        result[outer_fold] = plan
    return result


def _condition_inner_partition(
    plan: OuterInnerPlan,
    *,
    inner_fold: int,
    condition_train_tasks: frozenset[str],
) -> InnerFoldPartition:
    partition = plan.assignment.partition(inner_fold)
    return InnerFoldPartition(
        holdout_fold=inner_fold,
        initializer_fit_tasks=partition.initializer_fit_tasks & condition_train_tasks,
        validation_tasks=partition.validation_tasks & condition_train_tasks,
        holdout_tasks=partition.holdout_tasks & condition_train_tasks,
    )


def run_lifecycle_candidate_cv(
    dataset: SupervisedDataset,
    lifecycle_slice: LifecycleSlice,
    split_plan: SplitPlan,
    candidate: CandidateSpec,
    registry: EstimatorRegistry,
    *,
    alpha: float,
    calibrator_id: str,
    seed: int,
    fit_seed: int | None = None,
    inner_plans: Mapping[int, OuterInnerPlan],
    source_provenance: Mapping[str, Any],
    fit_checkpoint_factory: Callable[[int], FitCheckpoint | None] | None = None,
) -> CandidateResult:


    if fit_seed is None:
        fit_seed = seed

    update_slice, expected_weights = _validate_inputs(
        dataset,
        lifecycle_slice,
        split_plan,
        candidate,
        alpha=alpha,
        seed=seed,
    )
    plans = _validated_inner_plans(split_plan, inner_plans)
    provenance = validate_source_provenance(
        source_provenance,
        source_descriptor_id=lifecycle_slice.source_descriptor_id,
        capability_profile_id=lifecycle_slice.capability_profile_id,
        require_lifecycle_capabilities=True,
    )
    pre_slice = dataset.select(
        PredictionPosition.TASK_SEED,
        PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS,
        condition_id=lifecycle_slice.condition_id,
        token_unit_contract_id=lifecycle_slice.token_unit_contract_id,
    )
    expected_pre = {sequence.steps[0].point.point_id for sequence in lifecycle_slice.sequences}
    if {row.point.point_id for row in pre_slice.rows} != expected_pre:
        raise ValueError("Task-seed cohort differs from lifecycle sequence starts")
    pre_weights = {
        item.row.point.point_id: item.sample_weight for item in pre_slice.weighted_rows()
    }
    graph = candidate.graph
    if graph is None:
        raise AssertionError("candidate graph is missing")

    records: list[PredictionRecord] = []
    scored_all: list[ScoredForecast] = []
    fold_metrics: dict[int, Mapping[str, float | int | str]] = {}
    artifacts: list[FoldArtifact] = []
    parameter_text = json.dumps(
        candidate.initializer_params,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    for outer_fold in range(split_plan.folds):
        partition = split_plan.partition(outer_fold)
        available = lifecycle_slice.task_ids
        outer_tasks = {
            "train": available & partition.train_tasks,
            "validation": available & partition.validation_tasks,
            "calibration": available & partition.calibration_tasks,
            "test": available & partition.test_tasks,
        }
        if any(not tasks for tasks in outer_tasks.values()):
            raise ValueError(f"outer fold {outer_fold} has an empty lifecycle partition")
        inner_plan = plans[outer_fold]
        initializer_revision = identifier(
            "initializer-ensemble",
            graph.initializer_estimator_id,
            candidate.candidate_id,
            split_plan.assignment_id,
            f"outer-fold={outer_fold}",
            inner_plan.assignment.assignment_id,
            f"seed={seed}",
            f"params={parameter_text}",
        )
        packed_initializers: list[_PackedInitializer] = []
        # Restrict inner fitting and validation to the outer training set to prevent leakage.
        for inner_fold in range(inner_plan.assignment.folds):
            inner_partition = _condition_inner_partition(
                inner_plan,
                inner_fold=inner_fold,
                condition_train_tasks=outer_tasks["train"],
            )
            fit_rows = tuple(
                row
                for row in pre_slice.rows
                if row.point.task_id in inner_partition.initializer_fit_tasks
            )
            validation_rows = tuple(
                row
                for row in pre_slice.rows
                if row.point.task_id in inner_partition.validation_tasks
            )
            fitted_initializer = registry.create(
                graph.initializer_estimator_id,
                candidate.initializer_params,
            ).fit(
                _make_initializer_view(
                    pre_slice,
                    fit_rows,
                    pre_weights,
                    candidate.feature_set,
                    dataset_id=dataset.dataset_id,
                    partition_name=f"outer-{outer_fold}/inner-{inner_fold}/fit",
                ),
                _make_initializer_view(
                    pre_slice,
                    validation_rows,
                    pre_weights,
                    candidate.feature_set,
                    dataset_id=dataset.dataset_id,
                    partition_name=f"outer-{outer_fold}/inner-{inner_fold}/validation",
                ),
                FitContext(
                    seed=_fit_seed(
                        fit_seed,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                    ),
                    fold=inner_fold,
                    interval_alpha=alpha,
                ),
            )
            packed_initializers.append(
                _pack_initializer(
                    fitted_initializer,
                    initializer_id=graph.initializer_estimator_id,
                    initializer_revision=initializer_revision,
                    inner_fold=inner_fold,
                    partition=inner_partition,
                    feature_set=candidate.feature_set,
                )
            )

        external_tasks = frozenset(
            task for role in ("validation", "calibration", "test") for task in outer_tasks[role]
        )
        selected_pre_points = tuple(
            _lifecycle_point(sequence.steps[0].point, candidate.feature_set)
            for sequence in lifecycle_slice.sequences
            if sequence.task_id in outer_tasks["train"] | external_tasks
        )
        seed_set = generate_crossfit_seeds(
            selected_pre_points,
            tuple(packed.component for packed in packed_initializers),
            dataset_id=dataset.dataset_id,
            input_contract_id=lifecycle_slice.input_contract_id,
            initializer_id=graph.initializer_estimator_id,
            initializer_revision=initializer_revision,
            inner_split_id=inner_plan.assignment.assignment_id,
            oof_tasks=outer_tasks["train"],
            external_tasks=external_tasks,
            seed_policy_id=graph.seed_policy_id,
        )

        train_sequences = _partition_sequences(lifecycle_slice, outer_tasks["train"])
        validation_sequences = _partition_sequences(
            lifecycle_slice,
            outer_tasks["validation"],
        )
        train_seed_ids = {sequence.steps[0].point.point_id for sequence in train_sequences}
        validation_seed_ids = {
            sequence.steps[0].point.point_id for sequence in validation_sequences
        }
        fitted_updater = registry.create(candidate.estimator_id, candidate.params).fit(
            _make_updater_view(
                dataset_id=dataset.dataset_id,
                target=lifecycle_slice.target,
                sequences=train_sequences,
                seeds={point_id: seed_set.by_point_id[point_id] for point_id in train_seed_ids},
                feature_set=candidate.feature_set,
                partition_name=f"outer-{outer_fold}/train",
                input_contract_id=lifecycle_slice.input_contract_id,
                token_unit_contract_id=lifecycle_slice.token_unit_contract_id,
            ),
            _make_updater_view(
                dataset_id=dataset.dataset_id,
                target=lifecycle_slice.target,
                sequences=validation_sequences,
                seeds={
                    point_id: seed_set.by_point_id[point_id] for point_id in validation_seed_ids
                },
                feature_set=candidate.feature_set,
                partition_name=f"outer-{outer_fold}/validation",
                input_contract_id=lifecycle_slice.input_contract_id,
                token_unit_contract_id=lifecycle_slice.token_unit_contract_id,
            ),
            FitContext(
                seed=_fit_seed(fit_seed, outer_fold=outer_fold, inner_fold=None),
                fold=outer_fold,
                interval_alpha=alpha,
                checkpoint=(
                    fit_checkpoint_factory(outer_fold)
                    if fit_checkpoint_factory is not None
                    else None
                ),
            ),
        )

        calibration_sequences = _partition_sequences(
            lifecycle_slice,
            outer_tasks["calibration"],
        )
        calibration_seed_ids = {
            sequence.steps[0].point.point_id for sequence in calibration_sequences
        }
        calibration_runs = run_lifecycle_batch(
            fitted_updater,
            calibration_sequences,
            {point_id: seed_set.by_point_id[point_id] for point_id in calibration_seed_ids},
            select_point=lambda point: _lifecycle_point(point, candidate.feature_set),
        )
        calibration_predictions = _scored_predictions(calibration_runs)
        if not calibration_predictions:
            raise ValueError(f"outer fold {outer_fold} calibration set is empty")
        calibrator = _fit_calibrator(
            calibrator_id,
            alpha=alpha,
            examples=tuple(
                CalibrationExample(
                    prediction.step.point.task_id,
                    prediction.forecast,
                    float(prediction.step.label),
                    sample_weight=expected_weights[prediction.step.point.point_id],
                )
                for prediction in calibration_predictions
                if prediction.step.label is not None
            ),
        )
        calibrator_document = calibrator.to_dict()
        artifact = _bundle_artifact(
            fitted_updater,
            fold=outer_fold,
            candidate=candidate,
            dataset=dataset,
            update_slice=update_slice,
            lifecycle_slice=lifecycle_slice,
            split_plan=split_plan,
            outer_tasks=outer_tasks,
            inner_plan=inner_plan,
            initializer_revision=initializer_revision,
            packed_initializers=packed_initializers,
            seed_set_id=seed_set.seed_set_id,
            calibrator_id=calibrator_id,
            alpha=alpha,
            calibrator_document=calibrator_document,
            source_provenance=provenance,
        )

        test_sequences = _partition_sequences(lifecycle_slice, outer_tasks["test"])
        test_seed_ids = {sequence.steps[0].point.point_id for sequence in test_sequences}
        test_runs = run_lifecycle_batch(
            fitted_updater,
            test_sequences,
            {point_id: seed_set.by_point_id[point_id] for point_id in test_seed_ids},
            select_point=lambda point: _lifecycle_point(point, candidate.feature_set),
        )
        test_predictions = _scored_predictions(test_runs)
        if not test_predictions:
            raise ValueError(f"outer fold {outer_fold} test set is empty")

        loaded = load_lifecycle_bundle(
            dict(artifact.bundle_files or {}),
            expected_source_provenance=provenance,
        )
        replayed = loaded.run_calibrated(test_sequences)
        for expected_run, actual_run in zip(test_runs, replayed, strict=True):
            expected_forecasts = tuple(
                calibrator.transform(item.forecast) for item in expected_run.predictions
            )
            actual_forecasts = tuple(item.forecast for item in actual_run.predictions)
            for expected_forecast, actual_forecast in zip(
                expected_forecasts,
                actual_forecasts,
                strict=True,
            ):
                if (
                    replace(
                        expected_forecast,
                        latency_ms=actual_forecast.latency_ms,
                    )
                    != actual_forecast
                ):
                    raise ValueError("lifecycle bundle reload changed a test forecast")
        artifacts.append(artifact)

        fold_scored: list[ScoredForecast] = []
        for prediction in test_predictions:
            step = prediction.step
            if step.label is None:
                raise AssertionError("scored lifecycle prediction has no label")
            forecast = calibrator.transform(prediction.forecast)
            weight = expected_weights[step.point.point_id]
            records.append(
                PredictionRecord(
                    candidate_id=candidate.candidate_id,
                    point_id=step.point.point_id,
                    task_id=step.point.task_id,
                    trajectory_id=step.point.trajectory_id,
                    condition_id=step.point.condition_id,
                    fold=outer_fold,
                    target=step.point.target,
                    forecast=forecast,
                    sample_weight=weight,
                    token_unit_contract_id=step.point.token_unit_contract_id,
                )
            )
            scored = ScoredForecast(
                task_id=step.point.task_id,
                trajectory_id=step.point.trajectory_id,
                forecast=forecast,
                target_value=float(step.label),
                sample_weight=weight,
            )
            fold_scored.append(scored)
            scored_all.append(scored)
        fold_metrics[outer_fold] = evaluate_forecasts(fold_scored, alpha=alpha)

    expected_point_ids = {row.point.point_id for row in update_slice.rows}
    actual_point_ids = [record.point_id for record in records]
    if len(actual_point_ids) != len(set(actual_point_ids)) or set(actual_point_ids) != (
        expected_point_ids
    ):
        raise ValueError("lifecycle CV must predict every eligible update exactly once")
    return CandidateResult(
        candidate_id=candidate.candidate_id,
        model_revision=candidate.model_revision,
        dataset_id=dataset.dataset_id,
        split_plan_id=split_plan.split_plan_id,
        eligibility_policy_id=update_slice.eligibility_policy_id,
        position=PredictionPosition.TASK_UPDATE,
        target=lifecycle_slice.target,
        condition_id=lifecycle_slice.condition_id,
        calibrator_id=calibrator_id,
        alpha=alpha,
        metric_suite_id=METRIC_SUITE_ID,
        predictions=tuple(sorted(records, key=lambda record: record.point_id)),
        metrics=evaluate_forecasts(scored_all, alpha=alpha),
        fold_metrics=fold_metrics,
        task_metrics={
            task_id: task_metric.to_dict()
            for task_id, task_metric in evaluate_task_forecasts(
                scored_all,
                alpha=alpha,
            ).items()
        },
        fold_artifacts=tuple(artifacts),
    )


__all__ = [
    "LIFECYCLE_COMPOSITE_BUNDLE_SCHEMA_VERSION",
    "SeededLifecycleTrainingSequence",
    "TASK_LIFECYCLE_SCHEMA_ID",
    "run_lifecycle_candidate_cv",
]
