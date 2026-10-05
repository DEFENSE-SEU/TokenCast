from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist, fmean

from token_prediction.estimators.base import TokenForecast


WIDTH_TO_SIGMA_ASSUMPTION = "normal-interval-v1"


class UncertaintyDecompositionError(ValueError):
    pass


def _sigma_from_width(width: float, *, alpha: float) -> float:


    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return width / (2.0 * z) if z > 0 else 0.0


def decompose_forecast_uncertainty(
    members: Sequence[TokenForecast],
    *,
    alpha: float = 0.10,
) -> dict[str, float | int | str]:


    if not members:
        raise UncertaintyDecompositionError("decomposition needs at least one member")
    if not 0.0 < alpha < 1.0:
        raise UncertaintyDecompositionError("alpha must be in (0, 1)")
    if len({forecast.point_id for forecast in members}) != 1:
        raise UncertaintyDecompositionError("members predict different points")
    if len({forecast.target for forecast in members}) != 1:
        raise UncertaintyDecompositionError("members predict different targets")

    points = [float(forecast.point) for forecast in members]
    widths = [float(forecast.upper) - float(forecast.lower) for forecast in members]
    if any(width < 0 for width in widths):
        raise UncertaintyDecompositionError("a member interval is inverted")

    mean_point = fmean(points)

    epistemic = fmean((value - mean_point) ** 2 for value in points)
    aleatoric = fmean(_sigma_from_width(width, alpha=alpha) ** 2 for width in widths)
    total = aleatoric + epistemic
    return {
        "member_count": len(members),
        "alpha": alpha,
        "mean_point": mean_point,
        "mean_member_width": fmean(widths),
        "point_spread": max(points) - min(points),
        "epistemic_variance": epistemic,
        "aleatoric_variance": aleatoric,
        "total_variance": total,
        "epistemic_share": (epistemic / total) if total > 0 else 0.0,
        "width_conversion": WIDTH_TO_SIGMA_ASSUMPTION,
    }


def mixture_quantile(
    members: Sequence[TokenForecast],
    level: float,
    *,
    alpha: float = 0.10,
) -> float:


    if not members:
        raise UncertaintyDecompositionError("a mixture needs at least one member")
    if not 0.0 <= level <= 1.0:
        raise UncertaintyDecompositionError("level must be in [0, 1]")
    sigmas = [
        max(1e-9, _sigma_from_width(float(f.upper) - float(f.lower), alpha=alpha))
        for f in members
    ]
    points = [float(f.point) for f in members]
    if level <= 0.0:
        return min(points[i] - 8.0 * sigmas[i] for i in range(len(members)))
    if level >= 1.0:
        return max(points[i] + 8.0 * sigmas[i] for i in range(len(members)))

    def cumulative(x: float) -> float:
        return fmean(
            NormalDist(points[i], sigmas[i]).cdf(x) for i in range(len(members))
        )

    low = min(points[i] - 8.0 * sigmas[i] for i in range(len(members)))
    high = max(points[i] + 8.0 * sigmas[i] for i in range(len(members)))
    for _ in range(96):
        middle = (low + high) / 2.0
        if cumulative(middle) < level:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def mixture_forecast(
    members: Sequence[TokenForecast],
    *,
    alpha: float = 0.10,
) -> TokenForecast:


    if not members:
        raise UncertaintyDecompositionError("a mixture needs at least one member")
    lower = mixture_quantile(members, alpha / 2.0, alpha=alpha)
    point = mixture_quantile(members, 0.5, alpha=alpha)
    upper = mixture_quantile(members, 1.0 - alpha / 2.0, alpha=alpha)
    first = members[0]
    return TokenForecast(
        point_id=first.point_id,
        target=first.target,
        lower=max(0.0, min(lower, point)),
        point=max(0.0, point),
        upper=max(point, upper),
        latency_ms=sum(f.latency_ms for f in members),
        overhead_input_tokens=sum(f.overhead_input_tokens for f in members),
        overhead_output_tokens=sum(f.overhead_output_tokens for f in members),
    )


__all__ = [
    "UncertaintyDecompositionError",
    "WIDTH_TO_SIGMA_ASSUMPTION",
    "decompose_forecast_uncertainty",
    "mixture_forecast",
    "mixture_quantile",
]
