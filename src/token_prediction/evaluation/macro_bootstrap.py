from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal, Sequence
from token_prediction.evaluation.formal_intervals import _finite_float


MACRO_WAPE_BOOTSTRAP_ID = "paired_cluster_stratified_macro_wape_v1"
ClusterKey = Literal["task_id", "repo_id"]


@dataclass(frozen=True)
class MacroWapePrediction:


    method_id: str
    row_id: str
    task_id: str
    stratum_id: str
    target: float
    prediction: float
    sample_weight: float = 1.0
    repo_id: str | None = None


@dataclass(frozen=True)
class MacroWapeBootstrapComparison:


    candidate_id: str
    reference_id: str
    cluster_key: ClusterKey
    strata: tuple[str, ...]
    n_rows: int
    n_clusters: int
    iterations: int
    seed: int
    observed_candidate_macro_wape: float
    observed_reference_macro_wape: float
    observed_improvement_pct: float
    improvement_ci_lower_pct: float
    improvement_ci_upper_pct: float
    candidate_win_probability: float
    discarded_replicates: int
    bootstrap_improvements_pct: tuple[float, ...] = field(repr=False)
    confidence_level: float = 0.95
    interval_method: str = "percentile"
    bootstrap_id: str = MACRO_WAPE_BOOTSTRAP_ID


@dataclass(frozen=True)
class _ResolvedPredictions:
    method_id: str
    by_row_id: dict[str, MacroWapePrediction]


@dataclass(frozen=True)
class _ClusterStratumTotals:
    weighted_candidate_error: float
    weighted_reference_error: float
    weighted_absolute_target: float


def _non_empty_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _resolve_strata(strata: Sequence[str]) -> tuple[str, ...]:
    if isinstance(strata, (str, bytes)):
        raise TypeError("strata must be a sequence of stratum ids, not text")
    resolved = tuple(_non_empty_text(stratum, field_name="stratum id") for stratum in strata)
    if not resolved:
        raise ValueError("at least one predeclared stratum is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("predeclared strata must be unique")
    return resolved


def _resolve_predictions(
    rows: Sequence[MacroWapePrediction],
    *,
    role: str,
) -> _ResolvedPredictions:
    if isinstance(rows, (str, bytes)):
        raise TypeError(f"{role} predictions must be rows, not text")
    try:
        resolved_rows = tuple(rows)
    except TypeError as exc:
        raise TypeError(f"{role} predictions must be a sequence") from exc
    if not resolved_rows:
        raise ValueError(f"{role} prediction set is empty")

    by_row_id: dict[str, MacroWapePrediction] = {}
    method_ids: set[str] = set()
    for row in resolved_rows:
        if not isinstance(row, MacroWapePrediction):
            raise TypeError(f"{role} rows must be MacroWapePrediction instances")
        method_id = _non_empty_text(row.method_id, field_name=f"{role} method_id")
        row_id = _non_empty_text(row.row_id, field_name=f"{role} row_id")
        _non_empty_text(row.task_id, field_name=f"{role} task_id")
        _non_empty_text(row.stratum_id, field_name=f"{role} stratum_id")
        if row.repo_id is not None:
            _non_empty_text(row.repo_id, field_name=f"{role} repo_id")
        target = _finite_float(row.target, field_name=f"{role} target")
        if target < 0:
            raise ValueError(f"{role} targets must be non-negative")
        _finite_float(row.prediction, field_name=f"{role} prediction")
        weight = _finite_float(row.sample_weight, field_name=f"{role} sample_weight")
        if weight < 0:
            raise ValueError(f"{role} sample weights must be non-negative")
        if row_id in by_row_id:
            raise ValueError(f"{role} contains duplicate row_id {row_id!r}")
        method_ids.add(method_id)
        by_row_id[row_id] = row

    if len(method_ids) != 1:
        raise ValueError(f"{role} rows must have exactly one method_id")
    return _ResolvedPredictions(method_ids.pop(), by_row_id)


def _validate_declared_strata(
    resolved: _ResolvedPredictions,
    *,
    strata: tuple[str, ...],
    role: str,
) -> None:
    observed = {row.stratum_id for row in resolved.by_row_id.values()}
    declared = set(strata)
    missing = declared - observed
    unexpected = observed - declared
    if missing or unexpected:
        raise ValueError(
            f"{role} strata do not match the predeclared strata "
            f"(missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r})"
        )


def _macro_wape(
    resolved: _ResolvedPredictions,
    *,
    strata: tuple[str, ...],
) -> float:
    error_terms: dict[str, list[float]] = {stratum: [] for stratum in strata}
    target_terms: dict[str, list[float]] = {stratum: [] for stratum in strata}
    for row_id in sorted(resolved.by_row_id):
        row = resolved.by_row_id[row_id]
        weight = float(row.sample_weight)
        target = float(row.target)
        prediction = float(row.prediction)
        error_terms[row.stratum_id].append(weight * abs(prediction - target))
        target_terms[row.stratum_id].append(weight * abs(target))

    stratum_wapes: list[float] = []
    for stratum in strata:
        denominator = math.fsum(target_terms[stratum])
        if denominator <= 0:
            raise ValueError(f"stratum {stratum!r} has no positive weighted target denominator")
        stratum_wapes.append(math.fsum(error_terms[stratum]) / denominator)
    return math.fsum(stratum_wapes) / len(strata)


def stratified_macro_wape(
    rows: Sequence[MacroWapePrediction],
    *,
    strata: Sequence[str],
) -> float:


    resolved_strata = _resolve_strata(strata)
    resolved = _resolve_predictions(rows, role="prediction")
    _validate_declared_strata(resolved, strata=resolved_strata, role="prediction")
    return _macro_wape(resolved, strata=resolved_strata)


def _validate_pairing(
    candidate: _ResolvedPredictions,
    reference: _ResolvedPredictions,
) -> None:
    candidate_ids = set(candidate.by_row_id)
    reference_ids = set(reference.by_row_id)
    if candidate_ids != reference_ids:
        raise ValueError("candidate and reference rows are not exactly paired")
    for row_id in sorted(candidate_ids):
        candidate_row = candidate.by_row_id[row_id]
        reference_row = reference.by_row_id[row_id]
        candidate_signature = (
            candidate_row.task_id,
            candidate_row.repo_id,
            candidate_row.stratum_id,
            float(candidate_row.target),
            float(candidate_row.sample_weight),
        )
        reference_signature = (
            reference_row.task_id,
            reference_row.repo_id,
            reference_row.stratum_id,
            float(reference_row.target),
            float(reference_row.sample_weight),
        )
        if candidate_signature != reference_signature:
            raise ValueError(f"paired metadata differs at row {row_id!r}")


def _cluster_id(row: MacroWapePrediction, *, cluster_key: ClusterKey) -> str:
    if cluster_key == "task_id":
        return row.task_id
    if cluster_key == "repo_id":
        if row.repo_id is None:
            raise ValueError("repo_id is required when cluster_key='repo_id'")
        return row.repo_id
    raise ValueError("cluster_key must be 'task_id' or 'repo_id'")


def _cluster_totals(
    candidate: _ResolvedPredictions,
    reference: _ResolvedPredictions,
    *,
    cluster_key: ClusterKey,
    strata: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[_ClusterStratumTotals, ...], ...]]:
    terms: dict[str, dict[str, tuple[list[float], list[float], list[float]]]] = {}
    for row_id in sorted(candidate.by_row_id):
        candidate_row = candidate.by_row_id[row_id]
        reference_row = reference.by_row_id[row_id]
        cluster_id = _cluster_id(candidate_row, cluster_key=cluster_key)
        cluster_strata = terms.setdefault(cluster_id, {})
        candidate_errors, reference_errors, targets = cluster_strata.setdefault(
            candidate_row.stratum_id,
            ([], [], []),
        )
        weight = float(candidate_row.sample_weight)
        target = float(candidate_row.target)
        candidate_errors.append(weight * abs(float(candidate_row.prediction) - target))
        reference_errors.append(weight * abs(float(reference_row.prediction) - target))
        targets.append(weight * abs(target))

    cluster_ids = tuple(sorted(terms))
    totals: list[tuple[_ClusterStratumTotals, ...]] = []
    for cluster_id in cluster_ids:
        stratum_totals: list[_ClusterStratumTotals] = []
        for stratum in strata:
            candidate_errors, reference_errors, targets = terms[cluster_id].get(
                stratum,
                ([], [], []),
            )
            stratum_totals.append(
                _ClusterStratumTotals(
                    weighted_candidate_error=math.fsum(candidate_errors),
                    weighted_reference_error=math.fsum(reference_errors),
                    weighted_absolute_target=math.fsum(targets),
                )
            )
        totals.append(tuple(stratum_totals))
    return cluster_ids, tuple(totals)


def _replicate_improvement(
    multiplicities: Sequence[int],
    totals: Sequence[Sequence[_ClusterStratumTotals]],
) -> float | None:
    candidate_wapes: list[float] = []
    reference_wapes: list[float] = []
    for stratum_index in range(len(totals[0])):
        denominator = math.fsum(
            multiplicity * cluster[stratum_index].weighted_absolute_target
            for multiplicity, cluster in zip(multiplicities, totals)
        )
        if denominator <= 0:
            return None
        candidate_numerator = math.fsum(
            multiplicity * cluster[stratum_index].weighted_candidate_error
            for multiplicity, cluster in zip(multiplicities, totals)
        )
        reference_numerator = math.fsum(
            multiplicity * cluster[stratum_index].weighted_reference_error
            for multiplicity, cluster in zip(multiplicities, totals)
        )
        candidate_wapes.append(candidate_numerator / denominator)
        reference_wapes.append(reference_numerator / denominator)

    candidate_macro = math.fsum(candidate_wapes) / len(candidate_wapes)
    reference_macro = math.fsum(reference_wapes) / len(reference_wapes)
    if reference_macro <= 0:
        return None
    return 100.0 * (reference_macro - candidate_macro) / reference_macro


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def paired_cluster_macro_wape_bootstrap(
    candidate_rows: Sequence[MacroWapePrediction],
    reference_rows: Sequence[MacroWapePrediction],
    *,
    strata: Sequence[str],
    cluster_key: ClusterKey = "task_id",
    iterations: int = 10_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> MacroWapeBootstrapComparison:


    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if cluster_key not in ("task_id", "repo_id"):
        raise ValueError("cluster_key must be 'task_id' or 'repo_id'")
    confidence = _finite_float(confidence_level, field_name="confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    resolved_strata = _resolve_strata(strata)
    candidate = _resolve_predictions(candidate_rows, role="candidate")
    reference = _resolve_predictions(reference_rows, role="reference")
    _validate_declared_strata(candidate, strata=resolved_strata, role="candidate")
    _validate_declared_strata(reference, strata=resolved_strata, role="reference")
    _validate_pairing(candidate, reference)


    observed_candidate = _macro_wape(candidate, strata=resolved_strata)
    observed_reference = _macro_wape(reference, strata=resolved_strata)
    if observed_reference <= 0:
        raise ValueError("observed reference macro WAPE must be positive")
    observed_improvement = 100.0 * (observed_reference - observed_candidate) / observed_reference

    cluster_ids, totals = _cluster_totals(
        candidate,
        reference,
        cluster_key=cluster_key,
        strata=resolved_strata,
    )
    n_clusters = len(cluster_ids)
    rng = random.Random(seed)
    improvements: list[float] = []
    discarded = 0
    max_draws = max(iterations * 20, iterations + 1_000)
    draws = 0
    while len(improvements) < iterations and draws < max_draws:
        draws += 1
        # Resample whole clusters together for both methods to preserve their paired errors.
        multiplicities = [0] * n_clusters
        for _ in range(n_clusters):
            multiplicities[rng.randrange(n_clusters)] += 1
        improvement = _replicate_improvement(multiplicities, totals)
        if improvement is None:
            discarded += 1
            continue
        improvements.append(improvement)
    if len(improvements) != iterations:
        raise ValueError(
            "unable to draw enough valid bootstrap replicates; check stratum "
            "coverage and reference errors within clusters"
        )

    sorted_improvements = sorted(improvements)
    tail = (1.0 - confidence) / 2.0
    return MacroWapeBootstrapComparison(
        candidate_id=candidate.method_id,
        reference_id=reference.method_id,
        cluster_key=cluster_key,
        strata=resolved_strata,
        n_rows=len(candidate.by_row_id),
        n_clusters=n_clusters,
        iterations=iterations,
        seed=seed,
        observed_candidate_macro_wape=observed_candidate,
        observed_reference_macro_wape=observed_reference,
        observed_improvement_pct=observed_improvement,
        improvement_ci_lower_pct=_percentile(sorted_improvements, tail),
        improvement_ci_upper_pct=_percentile(sorted_improvements, 1.0 - tail),
        candidate_win_probability=(sum(value > 0 for value in improvements) / iterations),
        discarded_replicates=discarded,
        confidence_level=confidence,
        bootstrap_improvements_pct=tuple(improvements),
    )
