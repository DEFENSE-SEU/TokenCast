from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint
from token_prediction.features import DEFAULT_FEATURE_CATALOG, FeatureCatalog
from token_prediction.identifiers import identifier
from token_prediction.estimators.tabular_encoder import _category


NEURAL_ENCODER_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_]+")


class OptionalNeuralDependencyError(RuntimeError):
    pass


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise OptionalNeuralDependencyError(
            "neural estimation requires optional dependencies; install token-prediction[neural]"
        ) from exc
    return np


def _safe_name(value: str) -> str:
    result = _SAFE_NAME.sub("_", value).strip("_") or "feature"
    return f"f_{result}" if result[0].isdigit() else result


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, description: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{description} keys do not match schema; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _strict_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} keys must be strings")
    return value


def _strict_list(value: Any, *, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list")
    return value


def _strict_string(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _strict_int(value: Any, *, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def _strict_float(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _numeric(value: Any, *, feature_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"numeric feature {feature_name!r} must be an int, float, or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"numeric feature {feature_name!r} must be finite or None")
    return result


def _vector(value: Any, *, feature_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise ValueError(f"vector feature {feature_name!r} must be a tuple or None")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"vector feature {feature_name!r} contains a non-numeric value")
        parsed = float(item)
        if not math.isfinite(parsed):
            raise ValueError(f"vector feature {feature_name!r} must contain finite values")
        result.append(parsed)
    return tuple(result)


def _fitted_location(values: Sequence[float | None]) -> tuple[float, float, float]:
    observed = [value for value in values if value is not None]
    if not observed:


        return 0.0, 0.0, 1.0
    imputation = float(median(observed))
    completed = [imputation if value is None else value for value in values]
    mean = sum(completed) / len(completed)
    variance = sum((value - mean) ** 2 for value in completed) / len(completed)
    scale = math.sqrt(variance)
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    return imputation, float(mean), float(scale)


@dataclass(frozen=True)
class NumericTransform:
    feature_name: str
    median: float
    mean: float
    scale: float

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("numeric feature name is required")
        if not all(math.isfinite(value) for value in (self.median, self.mean, self.scale)):
            raise ValueError("numeric transform statistics must be finite")
        if self.scale <= 0:
            raise ValueError("numeric transform scale must be positive")


@dataclass(frozen=True)
class CategoryTransform:
    feature_name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("category feature name is required")
        if any(not isinstance(value, str) for value in self.values):
            raise ValueError("category values must be strings")
        if tuple(sorted(set(self.values))) != self.values:
            raise ValueError("category values must be sorted and unique")


@dataclass(frozen=True)
class VectorTransform:
    feature_name: str
    width: int
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("vector feature name is required")
        if self.width <= 0:
            raise ValueError("vector width must be positive")
        if not all(len(values) == self.width for values in (self.medians, self.means, self.scales)):
            raise ValueError("vector transform statistics do not match its width")
        if not all(
            math.isfinite(value)
            for values in (self.medians, self.means, self.scales)
            for value in values
        ):
            raise ValueError("vector transform statistics must be finite")
        if any(value <= 0 for value in self.scales):
            raise ValueError("vector transform scales must be positive")


@dataclass(frozen=True)
class NeuralEncoderSchema:
    numeric: tuple[NumericTransform, ...]
    categories: tuple[CategoryTransform, ...]
    vectors: tuple[VectorTransform, ...]
    dropped_all_missing_vectors: tuple[str, ...] = ()
    schema_version: int = NEURAL_ENCODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NEURAL_ENCODER_SCHEMA_VERSION:
            raise ValueError(f"unsupported neural encoder schema version {self.schema_version}")
        names = tuple(
            item.feature_name
            for group in (self.numeric, self.categories, self.vectors)
            for item in group
        )
        if len(names) != len(set(names)):
            raise ValueError("a source feature may occur in only one encoder transform")
        for description, group in (
            ("numeric", self.numeric),
            ("category", self.categories),
            ("vector", self.vectors),
        ):
            group_names = tuple(item.feature_name for item in group)
            if group_names != tuple(sorted(group_names)):
                raise ValueError(f"{description} transforms must be source-name sorted")
        if tuple(sorted(set(self.dropped_all_missing_vectors))) != self.dropped_all_missing_vectors:
            raise ValueError("dropped vector names must be sorted and unique")
        if set(names) & set(self.dropped_all_missing_vectors):
            raise ValueError("encoded and dropped vector features overlap")

    @property
    def feature_names(self) -> tuple[str, ...]:
        result: list[str] = []
        transforms: dict[str, tuple[str, Any]] = {
            item.feature_name: ("numeric", item) for item in self.numeric
        }
        transforms.update({item.feature_name: ("category", item) for item in self.categories})
        transforms.update({item.feature_name: ("vector", item) for item in self.vectors})
        for feature_name in sorted(transforms):
            dtype, transform = transforms[feature_name]
            safe = _safe_name(feature_name)
            if dtype == "numeric":
                result.extend((f"{safe}__value", f"{safe}__missing"))
            elif dtype == "category":
                result.append(f"{safe}__missing")
                result.extend(
                    f"{safe}__category_{index:04d}" for index in range(len(transform.values))
                )
                result.append(f"{safe}__unknown")
            else:
                result.extend(f"{safe}__v{index:04d}" for index in range(transform.width))
                result.append(f"{safe}__missing")
        if len(result) != len(set(result)):
            raise ValueError("encoded feature names collide after normalization")
        return tuple(result)

    @property
    def output_width(self) -> int:
        return len(self.feature_names)

    @property
    def source_features(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.feature_name
                for group in (self.numeric, self.categories, self.vectors)
                for item in group
            )
        )

    @property
    def schema_id(self) -> str:
        return identifier(
            "neural-encoder",
            f"schema-v{self.schema_version}",
            "+".join(self.feature_names) or "empty",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "numeric": [
                {
                    "feature_name": item.feature_name,
                    "median": item.median,
                    "mean": item.mean,
                    "scale": item.scale,
                }
                for item in self.numeric
            ],
            "categories": [
                {"feature_name": item.feature_name, "values": list(item.values)}
                for item in self.categories
            ],
            "vectors": [
                {
                    "feature_name": item.feature_name,
                    "width": item.width,
                    "medians": list(item.medians),
                    "means": list(item.means),
                    "scales": list(item.scales),
                }
                for item in self.vectors
            ],
            "dropped_all_missing_vectors": list(self.dropped_all_missing_vectors),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NeuralEncoderSchema":
        document = _strict_mapping(value, description="neural encoder schema")
        _strict_keys(
            document,
            {
                "schema_version",
                "numeric",
                "categories",
                "vectors",
                "dropped_all_missing_vectors",
            },
            description="neural encoder schema",
        )
        version = _strict_int(
            document["schema_version"], description="encoder schema version", minimum=1
        )
        numeric: list[NumericTransform] = []
        for index, raw in enumerate(
            _strict_list(document["numeric"], description="numeric transforms")
        ):
            item = _strict_mapping(raw, description=f"numeric transform {index}")
            _strict_keys(
                item,
                {"feature_name", "median", "mean", "scale"},
                description=f"numeric transform {index}",
            )
            numeric.append(
                NumericTransform(
                    feature_name=_strict_string(
                        item["feature_name"], description="numeric feature name"
                    ),
                    median=_strict_float(item["median"], description="numeric median"),
                    mean=_strict_float(item["mean"], description="numeric mean"),
                    scale=_strict_float(item["scale"], description="numeric scale"),
                )
            )
        categories: list[CategoryTransform] = []
        for index, raw in enumerate(
            _strict_list(document["categories"], description="category transforms")
        ):
            item = _strict_mapping(raw, description=f"category transform {index}")
            _strict_keys(
                item,
                {"feature_name", "values"},
                description=f"category transform {index}",
            )
            categories.append(
                CategoryTransform(
                    feature_name=_strict_string(
                        item["feature_name"], description="category feature name"
                    ),
                    values=tuple(
                        _strict_string(entry, description="category value")
                        for entry in _strict_list(item["values"], description="category values")
                    ),
                )
            )
        vectors: list[VectorTransform] = []
        for index, raw in enumerate(
            _strict_list(document["vectors"], description="vector transforms")
        ):
            item = _strict_mapping(raw, description=f"vector transform {index}")
            _strict_keys(
                item,
                {"feature_name", "width", "medians", "means", "scales"},
                description=f"vector transform {index}",
            )
            vectors.append(
                VectorTransform(
                    feature_name=_strict_string(
                        item["feature_name"], description="vector feature name"
                    ),
                    width=_strict_int(item["width"], description="vector width", minimum=1),
                    medians=tuple(
                        _strict_float(entry, description="vector median")
                        for entry in _strict_list(item["medians"], description="vector medians")
                    ),
                    means=tuple(
                        _strict_float(entry, description="vector mean")
                        for entry in _strict_list(item["means"], description="vector means")
                    ),
                    scales=tuple(
                        _strict_float(entry, description="vector scale")
                        for entry in _strict_list(item["scales"], description="vector scales")
                    ),
                )
            )
        dropped = tuple(
            _strict_string(entry, description="dropped vector name")
            for entry in _strict_list(
                document["dropped_all_missing_vectors"],
                description="dropped vector names",
            )
        )
        return cls(
            schema_version=version,
            numeric=tuple(numeric),
            categories=tuple(categories),
            vectors=tuple(vectors),
            dropped_all_missing_vectors=dropped,
        )


@dataclass(frozen=True)
class EncodedNeuralBatch:
    point_ids: tuple[str, ...]
    matrix: Any
    feature_names: tuple[str, ...]


class NeuralFeatureEncoder:


    def __init__(self, schema: NeuralEncoderSchema) -> None:
        self.schema = schema

    @classmethod
    def fit(
        cls,
        train_points: Sequence[PredictionPoint],
        *,
        catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
    ) -> "NeuralFeatureEncoder":
        if not train_points:
            raise ValueError("encoder training points are empty")
        point_ids = [point.point_id for point in train_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("encoder training point ids must be unique")
        source_names = sorted(
            {feature_name for point in train_points for feature_name in point.features}
        )
        numeric: list[NumericTransform] = []
        categories: list[CategoryTransform] = []
        vectors: list[VectorTransform] = []
        dropped_vectors: list[str] = []
        for source_name in source_names:
            spec = catalog.get(source_name)
            values = [point.features.get(source_name) for point in train_points]
            if spec.dtype == "numeric":
                parsed = [_numeric(value, feature_name=source_name) for value in values]
                imputation, mean, scale = _fitted_location(parsed)
                numeric.append(NumericTransform(source_name, imputation, mean, scale))
            elif spec.dtype == "category":
                parsed_categories = [_category(value, feature_name=source_name) for value in values]
                categories.append(
                    CategoryTransform(
                        source_name,
                        tuple(sorted({value for value in parsed_categories if value is not None})),
                    )
                )
            elif spec.dtype == "vector":
                parsed_vectors = [_vector(value, feature_name=source_name) for value in values]
                widths = {len(value) for value in parsed_vectors if value is not None}
                if not widths:
                    dropped_vectors.append(source_name)
                    continue
                if len(widths) != 1:
                    raise ValueError(
                        f"vector feature {source_name!r} has inconsistent train-fold widths: "
                        f"{sorted(widths)}"
                    )
                width = next(iter(widths))
                if width <= 0:
                    raise ValueError(f"vector feature {source_name!r} must not be empty")
                statistics = [
                    _fitted_location(
                        [value[index] if value is not None else None for value in parsed_vectors]
                    )
                    for index in range(width)
                ]
                vectors.append(
                    VectorTransform(
                        source_name,
                        width,
                        tuple(item[0] for item in statistics),
                        tuple(item[1] for item in statistics),
                        tuple(item[2] for item in statistics),
                    )
                )
            else:
                raise ValueError(
                    f"feature {source_name!r} declares unsupported dtype {spec.dtype!r}"
                )


        schema = NeuralEncoderSchema(
            numeric=tuple(numeric),
            categories=tuple(categories),
            vectors=tuple(vectors),
            dropped_all_missing_vectors=tuple(sorted(dropped_vectors)),
        )
        return cls(schema)

    def transform(self, points: Sequence[PredictionPoint]) -> EncodedNeuralBatch:
        if not points:
            raise ValueError("cannot encode an empty point sequence")
        np = _require_numpy()
        numeric = {item.feature_name: item for item in self.schema.numeric}
        categories = {item.feature_name: item for item in self.schema.categories}
        vectors = {item.feature_name: item for item in self.schema.vectors}
        source_features = self.schema.source_features
        output_width = self.schema.output_width
        rows: list[list[float]] = []
        for point in points:
            row: list[float] = []
            for source_name in source_features:
                raw = point.features.get(source_name)
                if source_name in numeric:
                    transform = numeric[source_name]
                    value = _numeric(raw, feature_name=source_name)
                    missing = value is None
                    completed = transform.median if missing else value
                    if completed is None:
                        raise AssertionError("neural encoder transform has no imputation value")
                    # Preserve missingness even when imputation equals an observed value.
                    row.extend(
                        (
                            (completed - transform.mean) / transform.scale,
                            float(missing),
                        )
                    )
                elif source_name in categories:
                    transform = categories[source_name]
                    value = _category(raw, feature_name=source_name)
                    if value is None:
                        row.extend((1.0, *(0.0 for _ in transform.values), 0.0))
                    else:
                        row.append(0.0)
                        row.extend(float(value == known) for known in transform.values)
                        row.append(float(value not in transform.values))
                else:
                    transform = vectors[source_name]
                    value = _vector(raw, feature_name=source_name)
                    if value is not None and len(value) != transform.width:
                        raise ValueError(
                            f"vector feature {source_name!r} has width {len(value)}; "
                            f"expected {transform.width}"
                        )
                    missing = value is None
                    completed = transform.medians if missing else value
                    if completed is None:
                        raise AssertionError("neural encoder transform has no imputation value")
                    row.extend(
                        (value - mean) / scale
                        for value, mean, scale in zip(
                            completed, transform.means, transform.scales, strict=True
                        )
                    )
                    row.append(float(missing))
            rows.append(row)
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != output_width:
            raise AssertionError("neural encoder produced an invalid matrix shape")
        if not bool(np.isfinite(matrix).all()):
            raise AssertionError("neural encoder produced a non-finite value")
        return EncodedNeuralBatch(
            point_ids=tuple(point.point_id for point in points),
            matrix=matrix,
            feature_names=self.schema.feature_names,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.schema.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NeuralFeatureEncoder":
        return cls(NeuralEncoderSchema.from_dict(value))


__all__ = [
    "EncodedNeuralBatch",
    "NEURAL_ENCODER_SCHEMA_VERSION",
    "NeuralEncoderSchema",
    "NeuralFeatureEncoder",
    "OptionalNeuralDependencyError",
]
