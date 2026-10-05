from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from token_prediction.dataset.schema import PredictionPoint, PredictionPosition
from token_prediction.estimators.tabular_encoder import FoldTabularEncoder
from token_prediction.features.selection import (
    DEFAULT_FEATURE_CATALOG,
    FeatureCatalog,
)


class SharedEncoderError(ValueError):
    pass


@dataclass(frozen=True)
class SharedPositionEncoder:


    encoder: FoldTabularEncoder
    positions: tuple[PredictionPosition, ...]
    shared_features: tuple[str, ...]
    position_features: Mapping[PredictionPosition, tuple[str, ...]]

    @property
    def schema_id(self) -> str:
        return self.encoder.schema.schema_id

    def feature_names(self) -> tuple[str, ...]:
        return self.encoder.schema.feature_names

    def transform(self, points: Sequence[PredictionPoint]):


        return self.encoder.transform(points)

    def exclusive_features(self, position: PredictionPosition) -> tuple[str, ...]:


        others = {
            name
            for other, names in self.position_features.items()
            if other != position
            for name in names
        }
        return tuple(
            name for name in self.position_features.get(position, ()) if name not in others
        )


def fit_shared_position_encoder(
    points_by_position: Mapping[PredictionPosition, Sequence[PredictionPoint]],
    *,
    catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
) -> SharedPositionEncoder:


    if not points_by_position:
        raise SharedEncoderError("a shared encoder needs at least one position")
    populated = {
        position: tuple(points)
        for position, points in points_by_position.items()
        if points
    }
    if not populated:
        raise SharedEncoderError("every position was empty")

    pooled: list[PredictionPoint] = []
    for position in sorted(populated, key=str):
        pooled.extend(populated[position])
    point_ids = [point.point_id for point in pooled]
    if len(point_ids) != len(set(point_ids)):
        raise SharedEncoderError("pooled point ids collide across positions")

    per_position = {
        position: tuple(
            sorted({name for point in points for name in point.features})
        )
        for position, points in populated.items()
    }
    shared: set[str] | None = None
    for names in per_position.values():
        shared = set(names) if shared is None else shared & set(names)

    return SharedPositionEncoder(
        encoder=FoldTabularEncoder.fit(pooled, catalog=catalog),
        positions=tuple(sorted(populated, key=str)),
        shared_features=tuple(sorted(shared or ())),
        position_features=per_position,
    )


__all__ = [
    "SharedEncoderError",
    "SharedPositionEncoder",
    "fit_shared_position_encoder",
]
