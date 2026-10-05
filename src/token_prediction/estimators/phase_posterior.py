from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset.schema import PredictionPoint

PHASE_FEATURE_SUFFIX = "_phase_posterior"


@dataclass(frozen=True)
class PhasePosterior:
    source_feature: str
    centroids: tuple[tuple[float, ...], ...]
    temperature: float
    keep_source: bool = False

    def __post_init__(self) -> None:
        if not self.source_feature.strip():
            raise ValueError("phase posterior requires a source feature name")
        if len(self.centroids) < 2:
            raise ValueError("phase posterior requires at least two centroids")
        if len({len(centroid) for centroid in self.centroids}) != 1:
            raise ValueError("phase centroids must share one width")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("phase temperature must be positive and finite")

    @property
    def feature_name(self) -> str:
        return f"{self.source_feature}{PHASE_FEATURE_SUFFIX}"

    @classmethod
    def fit(
        cls,
        points: Sequence[PredictionPoint],
        *,
        source_feature: str,
        clusters: int,
        seed: int,
        keep_source: bool = False,
    ) -> "PhasePosterior":
        import numpy as np
        from sklearn.cluster import KMeans

        vectors = np.asarray(
            [
                point.features[source_feature]
                for point in points
                if isinstance(point.features.get(source_feature), tuple)
            ],
            dtype=float,
        )
        if vectors.ndim != 2 or len(vectors) < clusters:
            raise ValueError(
                f"phase posterior needs at least {clusters} training vectors for "
                f"{source_feature!r}, got {len(vectors)}"
            )
        model = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(vectors)
        centroids = np.asarray(model.cluster_centers_, dtype=float)
        squared = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        temperature = float(np.median(squared.min(axis=1)))
        return cls(
            source_feature=source_feature,
            centroids=tuple(tuple(float(v) for v in centroid) for centroid in centroids),
            temperature=temperature if temperature > 0 else 1.0,
            keep_source=keep_source,
        )

    def posterior(self, vector: Sequence[float]) -> tuple[float, ...]:
        squared = [
            math.fsum((a - b) * (a - b) for a, b in zip(vector, centroid, strict=True))
            for centroid in self.centroids
        ]
        lowest = min(squared)
        weights = [math.exp(-(value - lowest) / self.temperature) for value in squared]
        total = math.fsum(weights)
        return tuple(weight / total for weight in weights)

    def apply(self, point: PredictionPoint) -> PredictionPoint:
        features = dict(point.features)
        vector = features.get(self.source_feature)
        if not self.keep_source:
            features.pop(self.source_feature, None)
        features[self.feature_name] = (
            self.posterior(vector) if isinstance(vector, tuple) else None
        )
        return point.with_features(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_feature": self.source_feature,
            "centroids": [list(centroid) for centroid in self.centroids],
            "temperature": self.temperature,
            "keep_source": self.keep_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhasePosterior":
        return cls(
            source_feature=str(value["source_feature"]),
            centroids=tuple(tuple(float(v) for v in centroid) for centroid in value["centroids"]),
            temperature=float(value["temperature"]),
            keep_source=bool(value.get("keep_source", False)),
        )


__all__ = ["PHASE_FEATURE_SUFFIX", "PhasePosterior"]
