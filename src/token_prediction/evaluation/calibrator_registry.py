from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .calibration import (
    CalibrationExample,
    FittedCalibrator,
    FittedExpansionCalibrator,
    IdentityCalibrator,
    IntervalCalibrator,
    TaskMaxConformalCalibrator,
)


CalibratorFactory = Callable[[float], IntervalCalibrator]
FittedCalibratorLoader = Callable[[Mapping[str, Any]], FittedCalibrator]

_FACTORIES: dict[str, CalibratorFactory] = {}
_FITTED_LOADERS: dict[str, FittedCalibratorLoader] = {}


def register_calibrator(
    calibrator_id: str,
    factory: CalibratorFactory,
    fitted_loader: FittedCalibratorLoader,
) -> None:
    key = str(calibrator_id).strip()
    if not key:
        raise ValueError("calibrator_id is required")
    if key in _FACTORIES:
        raise ValueError(f"calibrator {key!r} is already registered")
    _FACTORIES[key] = factory
    _FITTED_LOADERS[key] = fitted_loader


def registered_calibrator_ids() -> frozenset[str]:
    return frozenset(_FACTORIES)


def fit_calibrator(
    calibrator_id: str,
    *,
    alpha: float,
    examples: Sequence[CalibrationExample],
) -> FittedCalibrator:
    try:
        factory = _FACTORIES[calibrator_id]
    except KeyError as exc:
        raise ValueError(f"unknown calibrator {calibrator_id!r}") from exc
    return factory(alpha).fit(examples)


def load_fitted_calibrator(document: Mapping[str, Any]) -> FittedCalibrator:
    calibrator_id = document.get("calibrator_id")
    if not isinstance(calibrator_id, str) or not calibrator_id.strip():
        raise ValueError("fitted calibrator document requires calibrator_id")
    try:
        loader = _FITTED_LOADERS[calibrator_id]
    except KeyError as exc:
        raise ValueError(f"unknown calibrator {calibrator_id!r}") from exc
    return loader(document)


register_calibrator(
    "task_max_conformal",
    lambda alpha: TaskMaxConformalCalibrator(alpha=alpha),
    FittedExpansionCalibrator.from_dict,
)
register_calibrator(
    "none",
    lambda alpha: IdentityCalibrator(alpha=alpha),
    FittedExpansionCalibrator.from_dict,
)


def _register_conformal_variants() -> None:

    from .conformal import (
        FamilySampledPointConformalCalibrator,
        FittedGroupExpansionCalibrator,
        FittedMultiplicativeCalibrator,
        FittedWidthScaledCalibrator,
        FittedMagnitudeBinCalibrator,
        MagnitudeBinConformalCalibrator,
        MondrianConformalCalibrator,
        FittedResidualCalibrator,
        PointConformalCalibrator,
        PointResidualConformalCalibrator,
        PointWidthScaledConformalCalibrator,
        TaskMaxLogConformalCalibrator,
        TaskMaxWidthScaledConformalCalibrator,
        TaskHierarchicalPointConformalCalibrator,
        TaskQuantileConformalCalibrator,
        TaskWeightedPointConformalCalibrator,
        WeightedTaskMaxConformalCalibrator,
    )

    register_calibrator(
        "point_residual_conformal",
        lambda alpha: PointResidualConformalCalibrator(alpha=alpha),
        FittedResidualCalibrator.from_dict,
    )
    register_calibrator(
        "point_width_scaled_conformal",
        lambda alpha: PointWidthScaledConformalCalibrator(alpha=alpha),
        FittedWidthScaledCalibrator.from_dict,
    )
    register_calibrator(
        "magnitude_bin_conformal",
        lambda alpha: MagnitudeBinConformalCalibrator(alpha=alpha),
        FittedMagnitudeBinCalibrator.from_dict,
    )

    for calibrator_class in (
        PointConformalCalibrator,
        TaskWeightedPointConformalCalibrator,
        TaskHierarchicalPointConformalCalibrator,
        FamilySampledPointConformalCalibrator,
        TaskQuantileConformalCalibrator,
    ):
        register_calibrator(
            calibrator_class.calibrator_id,
            lambda alpha, _cls=calibrator_class: _cls(alpha=alpha),
            FittedExpansionCalibrator.from_dict,
        )

    register_calibrator(
        "weighted_task_max_conformal",
        lambda alpha: WeightedTaskMaxConformalCalibrator(alpha=alpha),
        FittedExpansionCalibrator.from_dict,
    )
    register_calibrator(
        "task_max_log_conformal",
        lambda alpha: TaskMaxLogConformalCalibrator(alpha=alpha),
        FittedMultiplicativeCalibrator.from_dict,
    )
    register_calibrator(
        "task_max_width_scaled_conformal",
        lambda alpha: TaskMaxWidthScaledConformalCalibrator(alpha=alpha),
        FittedWidthScaledCalibrator.from_dict,
    )
    for dimension in ("task_type", "model_family_id", "retry_state"):
        register_calibrator(
            f"mondrian_{dimension}_conformal",
            lambda alpha, _dimension=dimension: MondrianConformalCalibrator(
                alpha=alpha,
                group_dimension=_dimension,
            ),
            FittedGroupExpansionCalibrator.from_dict,
        )


_register_conformal_variants()
