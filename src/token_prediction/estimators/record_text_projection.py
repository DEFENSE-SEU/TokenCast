from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset.schema import PredictionPoint
from token_prediction.features.tool_reference import projection_feature_name

RECORD_TEXT_PROJECTION_SCHEMA_VERSION = 1


def _embed(texts: Sequence[str]) -> list[tuple[float, ...]]:
    from token_prediction.features.e5_embedding import embed_texts

    return embed_texts(texts, prefix="query")


def _distinct_texts(points: Sequence[PredictionPoint], sources: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for point in points
            for source in sources
            if isinstance(value := point.features.get(source), str) and value
        )
    )


@dataclass(frozen=True, eq=False)
class RecordTextProjection:
    sources: tuple[str, ...]
    dim: int
    mean: Any
    components: Any

    @classmethod
    def fit(
        cls,
        points: Sequence[PredictionPoint],
        *,
        sources: Sequence[str],
        dim: int,
    ) -> "RecordTextProjection":
        import numpy as np

        if int(dim) < 1:
            raise ValueError("record text projection dim must be positive")
        texts = _distinct_texts(points, sources)
        if len(texts) < 2:
            raise ValueError("record text projection needs at least two distinct training texts")
        matrix = np.asarray(_embed(texts), dtype=np.float64)
        mean = matrix.mean(axis=0)
        _, _, rows = np.linalg.svd(matrix - mean, full_matrices=False)
        components = rows[: max(1, min(int(dim), rows.shape[0]))]

        pivots = np.abs(components).argmax(axis=1)
        signs = np.sign(components[np.arange(components.shape[0]), pivots])
        signs[signs == 0] = 1.0
        return cls(
            sources=tuple(sources),
            dim=int(components.shape[0]),
            mean=mean,
            components=components * signs[:, None],
        )

    def apply_many(self, points: Sequence[PredictionPoint]) -> tuple[PredictionPoint, ...]:
        import numpy as np

        texts = _distinct_texts(points, self.sources)
        projected: dict[str, tuple[float, ...]] = {}
        if texts:
            vectors = (np.asarray(_embed(texts), dtype=np.float64) - self.mean) @ self.components.T
            projected = {text: tuple(float(v) for v in row) for text, row in zip(texts, vectors)}
        missing = (0.0,) * self.dim
        dropped = set(self.sources)
        out: list[PredictionPoint] = []
        for point in points:
            features = {name: value for name, value in point.features.items() if name not in dropped}
            for source in self.sources:
                value = point.features.get(source)
                features[projection_feature_name(source)] = (
                    projected[value] if isinstance(value, str) and value else missing
                )
            out.append(point.with_features(features))
        return tuple(out)

    def apply(self, point: PredictionPoint) -> PredictionPoint:
        return self.apply_many((point,))[0]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": RECORD_TEXT_PROJECTION_SCHEMA_VERSION,
            "sources": list(self.sources),
            "dim": self.dim,
        }

    def to_bytes(self) -> bytes:
        import numpy as np

        buffer = io.BytesIO()
        np.savez(buffer, mean=self.mean, components=self.components)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, manifest: Mapping[str, Any], payload: bytes) -> "RecordTextProjection":
        import numpy as np

        if manifest.get("schema_version") != RECORD_TEXT_PROJECTION_SCHEMA_VERSION:
            raise ValueError("unsupported record text projection schema version")
        with np.load(io.BytesIO(payload)) as archive:
            return cls(
                sources=tuple(str(name) for name in manifest["sources"]),
                dim=int(manifest["dim"]),
                mean=np.asarray(archive["mean"], dtype=np.float64),
                components=np.asarray(archive["components"], dtype=np.float64),
            )


__all__ = [
    "RECORD_TEXT_PROJECTION_SCHEMA_VERSION",
    "RecordTextProjection",
    "projection_feature_name",
]
