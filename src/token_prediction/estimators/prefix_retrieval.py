from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset.schema import PredictionPoint

RETRIEVAL_FEATURE_SUFFIX = "_retrieval"
RETRIEVAL_STATS: tuple[str, ...] = (
    "median",
    "mean",
    "q25",
    "q75",
    "min",
    "max",
    "nearest_label",
    "nearest_similarity",
    "mean_similarity",
)
RETRIEVAL_REFERENCE_UNITS = ("point", "task")
RETRIEVAL_SUMMARY_WEIGHTINGS = ("uniform", "nonnegative_cosine")
RETRIEVAL_LABEL_POLICIES = ("observed", "shuffled")
RETRIEVAL_NEIGHBOR_POLICIES = ("similarity", "shuffled")
PREFIX_RETRIEVAL_SCHEMA_VERSION = 2
LINEAR_WEIGHTED_QUANTILE_POLICY = "linear-weighted-type7-equal-v1"
QUERY_LEAVE_TASK_OUT_LABEL_SHUFFLE = "query-leave-task-out-v2"
_LEGACY_QUANTILE_POLICY = "legacy-mixed-v1"
_LEGACY_LABEL_SHUFFLE = "fit-global-v1"
_BATCH_ROWS = 512


def _unit_rows(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _weighted_quantile(values: Any, weights: Any, quantile: float) -> float:
    import numpy as np

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    positive = ordered_weights > 0
    ordered_values = ordered_values[positive]
    ordered_weights = ordered_weights[positive]
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / ordered_weights.sum()
    target = (quantile * (len(ordered_values) - 1) + 0.5) / len(ordered_values)
    return float(np.interp(target, positions, ordered_values))


def _legacy_weighted_quantile(values: Any, weights: Any, quantile: float) -> float:
    import numpy as np

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


@dataclass(frozen=True, eq=False)
class PrefixRetrieval:
    source_feature: str
    neighbors: int
    vectors: Any
    labels: Any
    task_ids: tuple[str, ...]
    reference_unit: str = "point"
    summary_weighting: str = "uniform"
    label_policy: str = "observed"
    neighbor_policy: str = "similarity"
    control_seed: int = 0
    quantile_policy: str = LINEAR_WEIGHTED_QUANTILE_POLICY
    label_shuffle_policy: str = QUERY_LEAVE_TASK_OUT_LABEL_SHUFFLE

    def __post_init__(self) -> None:
        if not self.source_feature.strip():
            raise ValueError("prefix retrieval requires a source feature name")
        if isinstance(self.neighbors, bool) or int(self.neighbors) < 1:
            raise ValueError("prefix retrieval requires at least one neighbour")
        if self.vectors.ndim != 2 or len(self.vectors) == 0:
            raise ValueError("prefix retrieval requires a non-empty reference matrix")
        if len(self.labels) != len(self.vectors) or len(self.task_ids) != len(self.vectors):
            raise ValueError("prefix retrieval references must align")
        if self.reference_unit not in RETRIEVAL_REFERENCE_UNITS:
            raise ValueError(f"unsupported prefix retrieval reference unit {self.reference_unit!r}")
        if self.reference_unit == "task" and len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task retrieval requires one reference per Task")
        if self.summary_weighting not in RETRIEVAL_SUMMARY_WEIGHTINGS:
            raise ValueError(f"unsupported retrieval summary weighting {self.summary_weighting!r}")
        if self.label_policy not in RETRIEVAL_LABEL_POLICIES:
            raise ValueError(f"unsupported retrieval label policy {self.label_policy!r}")
        if self.neighbor_policy not in RETRIEVAL_NEIGHBOR_POLICIES:
            raise ValueError(f"unsupported retrieval neighbor policy {self.neighbor_policy!r}")
        if self.quantile_policy not in {
            LINEAR_WEIGHTED_QUANTILE_POLICY,
            _LEGACY_QUANTILE_POLICY,
        }:
            raise ValueError(f"unsupported retrieval quantile policy {self.quantile_policy!r}")
        if self.label_shuffle_policy not in {
            QUERY_LEAVE_TASK_OUT_LABEL_SHUFFLE,
            _LEGACY_LABEL_SHUFFLE,
        }:
            raise ValueError(
                f"unsupported retrieval label shuffle policy {self.label_shuffle_policy!r}"
            )
        if (
            isinstance(self.control_seed, bool)
            or not isinstance(self.control_seed, int)
            or self.control_seed < 0
        ):
            raise ValueError("retrieval control seed must be a non-negative integer")

    @property
    def feature_name(self) -> str:
        return f"{self.source_feature}{RETRIEVAL_FEATURE_SUFFIX}"

    @classmethod
    def fit(
        cls,
        points: Sequence[PredictionPoint],
        labels: Sequence[float],
        *,
        source_feature: str,
        neighbors: int,
        sample_weights: Sequence[float] | None = None,
        reference_unit: str = "point",
        summary_weighting: str = "uniform",
        label_policy: str = "observed",
        neighbor_policy: str = "similarity",
        control_seed: int = 0,
    ) -> "PrefixRetrieval":
        import numpy as np

        if len(points) != len(labels):
            raise ValueError("prefix retrieval needs one label per point")
        if reference_unit not in RETRIEVAL_REFERENCE_UNITS:
            raise ValueError(f"unsupported prefix retrieval reference unit {reference_unit!r}")
        if summary_weighting not in RETRIEVAL_SUMMARY_WEIGHTINGS:
            raise ValueError(f"unsupported retrieval summary weighting {summary_weighting!r}")
        if label_policy not in RETRIEVAL_LABEL_POLICIES:
            raise ValueError(f"unsupported retrieval label policy {label_policy!r}")
        if neighbor_policy not in RETRIEVAL_NEIGHBOR_POLICIES:
            raise ValueError(f"unsupported retrieval neighbor policy {neighbor_policy!r}")
        if isinstance(control_seed, bool) or not isinstance(control_seed, int) or control_seed < 0:
            raise ValueError("retrieval control seed must be a non-negative integer")
        weights = tuple(1.0 for _ in points) if sample_weights is None else tuple(sample_weights)
        if len(weights) != len(points):
            raise ValueError("prefix retrieval needs one sample weight per point")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("prefix retrieval sample weights must be finite and positive")
        kept = [
            (point, float(label), float(weight))
            for point, label, weight in zip(points, labels, weights)
            if isinstance(point.features.get(source_feature), tuple)
        ]
        if not kept:
            raise ValueError(f"prefix retrieval found no training vectors for {source_feature!r}")
        kept.sort(key=lambda item: (item[0].task_id, item[0].point_id))
        if reference_unit == "task":
            references: list[tuple[str, Any, float]] = []
            for task_id in sorted({point.task_id for point, _, _ in kept}):
                rows = [item for item in kept if item[0].task_id == task_id]
                task_weights = np.asarray([weight for _, _, weight in rows], dtype=float)
                references.append(
                    (
                        task_id,
                        np.average(
                            np.asarray(
                                [point.features[source_feature] for point, _, _ in rows],
                                dtype=np.float32,
                            ),
                            axis=0,
                            weights=task_weights,
                        ),
                        float(
                            np.average(
                                np.asarray([label for _, label, _ in rows], dtype=float),
                                weights=task_weights,
                            )
                        ),
                    )
                )
        else:
            references = [
                (point.task_id, point.features[source_feature], label)
                for point, label, _ in kept
            ]
        vectors = np.asarray([vector for _, vector, _ in references], dtype=np.float32)
        return cls(
            source_feature=source_feature,
            neighbors=int(neighbors),
            vectors=_unit_rows(vectors),
            labels=np.asarray([label for _, _, label in references], dtype=float),
            task_ids=tuple(task_id for task_id, _, _ in references),
            reference_unit=reference_unit,
            summary_weighting=summary_weighting,
            label_policy=label_policy,
            neighbor_policy=neighbor_policy,
            control_seed=control_seed,
        )

    def summaries(self, vectors: Any, task_ids: Sequence[str]) -> Any:


        import numpy as np

        queries = _unit_rows(np.asarray(vectors, dtype=np.float32))
        reference_tasks = np.asarray(self.task_ids)
        query_tasks = np.asarray(list(task_ids))
        out = np.full((len(queries), len(RETRIEVAL_STATS)), np.nan)
        for start in range(0, len(queries), _BATCH_ROWS):
            stop = start + _BATCH_ROWS
            similarity = queries[start:stop] @ self.vectors.T
            similarity[query_tasks[start:stop, None] == reference_tasks[None, :]] = -np.inf
            k = min(self.neighbors, similarity.shape[1])
            for row, row_similarity in enumerate(similarity):
                eligible = np.flatnonzero(np.isfinite(row_similarity))
                count = min(k, len(eligible))
                if not count:
                    continue
                if self.neighbor_policy == "shuffled" and len(eligible) > count:
                    shuffled = eligible.tolist()
                    random.Random(
                        f"prefix-retrieval-neighbor-shuffle-v1:"
                        f"{self.control_seed}:{query_tasks[start + row]}"
                    ).shuffle(shuffled)
                    selected = np.asarray(shuffled[:count], dtype=int)
                elif len(eligible) > count:
                    values = row_similarity[eligible]
                    threshold = -np.partition(-values, count - 1)[count - 1]
                    higher = eligible[values > threshold]
                    tied = np.sort(eligible[values == threshold])
                    selected = np.concatenate((higher, tied[: count - len(higher)]))
                else:
                    selected = eligible
                selected = selected[
                    np.lexsort((selected, -row_similarity[selected]))
                ]
                if (
                    self.label_policy == "shuffled"
                    and self.label_shuffle_policy == QUERY_LEAVE_TASK_OUT_LABEL_SHUFFLE
                ):
                    shuffled_labels = self.labels[eligible].tolist()
                    random.Random(
                        f"prefix-retrieval-label-shuffle-v2:"
                        f"{self.control_seed}:{query_tasks[start + row]}"
                    ).shuffle(shuffled_labels)
                    chosen = np.asarray(shuffled_labels, dtype=float)[
                        np.searchsorted(eligible, selected)
                    ]
                else:
                    chosen = self.labels[selected]
                chosen_sims = row_similarity[selected]
                if self.summary_weighting == "nonnegative_cosine":
                    summary_weights = np.maximum(chosen_sims.astype(float), 0.0)
                    if not summary_weights.any():
                        summary_weights = np.ones(len(chosen), dtype=float)
                    quantile_function = (
                        _legacy_weighted_quantile
                        if self.quantile_policy == _LEGACY_QUANTILE_POLICY
                        else _weighted_quantile
                    )
                    median = quantile_function(chosen, summary_weights, 0.5)
                    mean = np.average(chosen, weights=summary_weights)
                    q25 = quantile_function(chosen, summary_weights, 0.25)
                    q75 = quantile_function(chosen, summary_weights, 0.75)
                    mean_similarity = np.average(chosen_sims, weights=summary_weights)
                else:
                    median = np.median(chosen)
                    mean = chosen.mean()
                    q25 = np.quantile(chosen, 0.25)
                    q75 = np.quantile(chosen, 0.75)
                    mean_similarity = chosen_sims.mean()
                out[start + row] = (
                    median,
                    mean,
                    q25,
                    q75,
                    chosen.min(),
                    chosen.max(),
                    chosen[0],
                    chosen_sims[0],
                    mean_similarity,
                )
        return out

    def apply_many(self, points: Sequence[PredictionPoint]) -> tuple[PredictionPoint, ...]:
        import numpy as np

        indexed = [
            index
            for index, point in enumerate(points)
            if isinstance(point.features.get(self.source_feature), tuple)
        ]
        summaries: dict[int, tuple[float, ...] | None] = {}
        if indexed:
            rows = self.summaries(
                [points[index].features[self.source_feature] for index in indexed],
                [points[index].task_id for index in indexed],
            )
            for index, row in zip(indexed, rows):
                summaries[index] = None if np.isnan(row).any() else tuple(float(v) for v in row)
        return tuple(
            point.with_features({**point.features, self.feature_name: summaries.get(index)})
            for index, point in enumerate(points)
        )

    def apply(self, point: PredictionPoint) -> PredictionPoint:
        return self.apply_many((point,))[0]

    def manifest(self) -> dict[str, Any]:


        return {
            "schema_version": PREFIX_RETRIEVAL_SCHEMA_VERSION,
            "source_feature": self.source_feature,
            "neighbors": self.neighbors,
            "reference_unit": self.reference_unit,
            "summary_weighting": self.summary_weighting,
            "label_policy": self.label_policy,
            "neighbor_policy": self.neighbor_policy,
            "control_seed": self.control_seed,
            "quantile_policy": self.quantile_policy,
            "label_shuffle_policy": self.label_shuffle_policy,
            "reference_count": len(self.task_ids),
            "distinct_task_count": len(set(self.task_ids)),
            "external_query_neighbor_count": min(self.neighbors, len(self.task_ids)),
        }

    def to_bytes(self) -> bytes:


        import io

        import numpy as np

        buffer = io.BytesIO()
        np.savez(
            buffer,
            vectors=self.vectors,
            labels=self.labels,
            task_ids=np.asarray(self.task_ids, dtype=str),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, manifest: Mapping[str, Any], payload: bytes) -> "PrefixRetrieval":
        import io

        import numpy as np

        schema_version = manifest.get("schema_version", 1)
        if isinstance(schema_version, bool) or schema_version not in {
            1,
            PREFIX_RETRIEVAL_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported prefix retrieval schema version")
        if schema_version == PREFIX_RETRIEVAL_SCHEMA_VERSION and not {
            "quantile_policy",
            "label_shuffle_policy",
        } <= set(manifest):
            raise ValueError("prefix retrieval schema v2 requires explicit control policies")
        with np.load(io.BytesIO(payload)) as archive:
            fitted = cls(
                source_feature=str(manifest["source_feature"]),
                neighbors=int(manifest["neighbors"]),
                vectors=np.asarray(archive["vectors"], dtype=np.float32),
                labels=np.asarray(archive["labels"], dtype=float),
                task_ids=tuple(str(item) for item in archive["task_ids"]),
                reference_unit=str(manifest.get("reference_unit", "point")),
                summary_weighting=str(manifest.get("summary_weighting", "uniform")),
                label_policy=str(manifest.get("label_policy", "observed")),
                neighbor_policy=str(manifest.get("neighbor_policy", "similarity")),
                control_seed=manifest.get("control_seed", 0),
                quantile_policy=str(
                    manifest.get("quantile_policy", _LEGACY_QUANTILE_POLICY)
                ),
                label_shuffle_policy=str(
                    manifest.get("label_shuffle_policy", _LEGACY_LABEL_SHUFFLE)
                ),
            )
        expected_support = {
            "reference_count": len(fitted.task_ids),
            "distinct_task_count": len(set(fitted.task_ids)),
            "external_query_neighbor_count": min(fitted.neighbors, len(fitted.task_ids)),
        }
        for name, expected in expected_support.items():
            if name in manifest and manifest[name] != expected:
                raise ValueError(f"retrieval manifest {name} does not match its arrays")
        return fitted


__all__ = [
    "RETRIEVAL_FEATURE_SUFFIX",
    "LINEAR_WEIGHTED_QUANTILE_POLICY",
    "PREFIX_RETRIEVAL_SCHEMA_VERSION",
    "QUERY_LEAVE_TASK_OUT_LABEL_SHUFFLE",
    "RETRIEVAL_LABEL_POLICIES",
    "RETRIEVAL_NEIGHBOR_POLICIES",
    "RETRIEVAL_REFERENCE_UNITS",
    "RETRIEVAL_STATS",
    "RETRIEVAL_SUMMARY_WEIGHTINGS",
    "PrefixRetrieval",
]
