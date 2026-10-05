from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset import PredictionPoint
from token_prediction.features import DEFAULT_FEATURE_CATALOG, FeatureCatalog
from token_prediction.identifiers import identifier


ENCODER_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_]+")
_SUPPORTED_DTYPES = frozenset({"numeric", "category", "vector"})


def _safe_column_name(value: str) -> str:
    name = _SAFE_NAME.sub("_", value).strip("_") or "feature"
    if name[0].isdigit():
        name = f"f_{name}"
    return name


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tabular estimators require NumPy; install token-prediction[estimators]"
        ) from exc
    return np


def _number(value: Any, *, feature_name: str) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"numeric feature {feature_name!r} must be an int, float, or None")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"numeric feature {feature_name!r} must be finite or None")
    return parsed


def _category(value: Any, *, feature_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"category feature {feature_name!r} must be a string or None")
    return value


def _vector(value: Any, *, feature_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise ValueError(f"vector feature {feature_name!r} must be a tuple or None")
    parsed: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"vector feature {feature_name!r} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"vector feature {feature_name!r} must contain finite values")
        parsed.append(number)
    return tuple(parsed)


@dataclass(frozen=True)
class EncodedColumn:
    name: str
    source_feature: str
    dtype: str
    vector_index: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported encoded dtype {self.dtype!r}")
        if self.dtype == "vector" and self.vector_index is None:
            raise ValueError("vector columns require vector_index")
        if self.dtype != "vector" and self.vector_index is not None:
            raise ValueError("only vector columns may declare vector_index")


@dataclass(frozen=True)
class CategoryVocabulary:
    feature_name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class VectorDimension:
    feature_name: str
    width: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("vector width must be positive")


@dataclass(frozen=True)
class EncoderSchema:
    columns: tuple[EncodedColumn, ...]
    category_vocabularies: tuple[CategoryVocabulary, ...]
    vector_dimensions: tuple[VectorDimension, ...]
    dropped_all_missing_vectors: tuple[str, ...] = ()
    schema_version: int = ENCODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENCODER_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported encoder schema version {self.schema_version}; "
                f"expected {ENCODER_SCHEMA_VERSION}"
            )
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("encoded feature names must be unique")
        category_names = [item.feature_name for item in self.category_vocabularies]
        vector_names = [item.feature_name for item in self.vector_dimensions]
        if len(category_names) != len(set(category_names)):
            raise ValueError("category vocabulary names must be unique")
        if len(vector_names) != len(set(vector_names)):
            raise ValueError("vector dimension names must be unique")
        category_sources = {
            column.source_feature for column in self.columns if column.dtype == "category"
        }
        vector_sources = {
            column.source_feature for column in self.columns if column.dtype == "vector"
        }
        if category_sources != set(category_names):
            raise ValueError("category columns and vocabularies do not match")
        if vector_sources != set(vector_names):
            raise ValueError("vector columns and dimensions do not match")
        for vector in self.vector_dimensions:
            indices = sorted(
                int(column.vector_index)
                for column in self.columns
                if column.dtype == "vector"
                and column.source_feature == vector.feature_name
                and column.vector_index is not None
            )
            if indices != list(range(vector.width)):
                raise ValueError(
                    f"vector columns for {vector.feature_name!r} do not cover its width"
                )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def source_features(self) -> tuple[str, ...]:
        return tuple(column.source_feature for column in self.columns)

    @property
    def categorical_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, column in enumerate(self.columns) if column.dtype == "category"
        )

    @property
    def schema_id(self) -> str:
        return identifier(
            "tabular-encoder",
            f"schema-v{self.schema_version}",
            "+".join(self.feature_names) or "empty",
        )

    def category_mapping(self, feature_name: str) -> dict[str, int]:
        for item in self.category_vocabularies:
            if item.feature_name == feature_name:
                return {value: index for index, value in enumerate(item.values)}
        raise KeyError(f"no category vocabulary for {feature_name!r}")

    def vector_width(self, feature_name: str) -> int:
        for item in self.vector_dimensions:
            if item.feature_name == feature_name:
                return item.width
        raise KeyError(f"no vector dimension for {feature_name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "columns": [
                {
                    "name": item.name,
                    "source_feature": item.source_feature,
                    "dtype": item.dtype,
                    "vector_index": item.vector_index,
                }
                for item in self.columns
            ],
            "category_vocabularies": [
                {"feature_name": item.feature_name, "values": list(item.values)}
                for item in self.category_vocabularies
            ],
            "vector_dimensions": [
                {"feature_name": item.feature_name, "width": item.width}
                for item in self.vector_dimensions
            ],
            "dropped_all_missing_vectors": list(self.dropped_all_missing_vectors),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderSchema":
        return cls(
            schema_version=int(value.get("schema_version") or 0),
            columns=tuple(
                EncodedColumn(
                    name=str(item["name"]),
                    source_feature=str(item["source_feature"]),
                    dtype=str(item["dtype"]),
                    vector_index=(
                        int(item["vector_index"]) if item.get("vector_index") is not None else None
                    ),
                )
                for item in value.get("columns") or ()
            ),
            category_vocabularies=tuple(
                CategoryVocabulary(
                    feature_name=str(item["feature_name"]),
                    values=tuple(str(entry) for entry in item.get("values") or ()),
                )
                for item in value.get("category_vocabularies") or ()
            ),
            vector_dimensions=tuple(
                VectorDimension(
                    feature_name=str(item["feature_name"]),
                    width=int(item["width"]),
                )
                for item in value.get("vector_dimensions") or ()
            ),
            dropped_all_missing_vectors=tuple(
                str(item) for item in value.get("dropped_all_missing_vectors") or ()
            ),
        )


@dataclass(frozen=True)
class EncodedBatch:
    point_ids: tuple[str, ...]
    matrix: Any
    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]


class FoldTabularEncoder:


    def __init__(self, schema: EncoderSchema) -> None:
        self.schema = schema

    @classmethod
    def fit(
        cls,
        train_points: Sequence[PredictionPoint],
        *,
        catalog: FeatureCatalog = DEFAULT_FEATURE_CATALOG,
    ) -> "FoldTabularEncoder":
        if not train_points:
            raise ValueError("encoder training points are empty")
        point_ids = [point.point_id for point in train_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("encoder training point ids must be unique")

        source_names = sorted(
            {feature_name for point in train_points for feature_name in point.features}
        )
        columns: list[EncodedColumn] = []
        vocabularies: list[CategoryVocabulary] = []
        vector_dimensions: list[VectorDimension] = []
        dropped_vectors: list[str] = []

        for source_name in source_names:
            spec = catalog.get(source_name)
            if spec.dtype not in _SUPPORTED_DTYPES:
                raise ValueError(
                    f"feature {source_name!r} declares unsupported dtype {spec.dtype!r}"
                )
            safe_name = _safe_column_name(source_name)
            values = [point.features.get(source_name) for point in train_points]
            if spec.dtype == "numeric":
                for value in values:
                    _number(value, feature_name=source_name)
                columns.append(EncodedColumn(safe_name, source_name, "numeric"))
                continue
            if spec.dtype == "category":
                categories = tuple(
                    sorted(
                        {
                            parsed
                            for value in values
                            if (parsed := _category(value, feature_name=source_name)) is not None
                        }
                    )
                )
                vocabularies.append(CategoryVocabulary(source_name, categories))
                columns.append(EncodedColumn(safe_name, source_name, "category"))
                continue

            widths: set[int] = set()
            for value in values:
                parsed_vector = _vector(value, feature_name=source_name)
                if parsed_vector is not None:
                    widths.add(len(parsed_vector))
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
            vector_dimensions.append(VectorDimension(source_name, width))
            columns.extend(
                EncodedColumn(
                    f"{safe_name}__v{index:04d}",
                    source_name,
                    "vector",
                    vector_index=index,
                )
                for index in range(width)
            )

        schema = EncoderSchema(
            columns=tuple(columns),
            category_vocabularies=tuple(vocabularies),
            vector_dimensions=tuple(vector_dimensions),
            dropped_all_missing_vectors=tuple(sorted(dropped_vectors)),
        )
        return cls(schema)

    def transform(self, points: Sequence[PredictionPoint]) -> EncodedBatch:
        if not points:
            raise ValueError("cannot encode an empty point sequence")
        np = _require_numpy()
        category_maps = {
            item.feature_name: {value: index for index, value in enumerate(item.values)}
            for item in self.schema.category_vocabularies
        }
        vector_widths = {item.feature_name: item.width for item in self.schema.vector_dimensions}
        rows: list[list[float]] = []
        for point in points:
            row: list[float] = []
            parsed_vectors: dict[str, tuple[float, ...] | None] = {}
            for column in self.schema.columns:
                raw = point.features.get(column.source_feature)
                if column.dtype == "numeric":
                    row.append(_number(raw, feature_name=column.source_feature))
                elif column.dtype == "category":
                    parsed = _category(raw, feature_name=column.source_feature)
                    row.append(float(category_maps[column.source_feature].get(parsed, -1)))
                else:
                    if column.source_feature not in parsed_vectors:
                        vector = _vector(raw, feature_name=column.source_feature)
                        expected_width = vector_widths[column.source_feature]
                        if vector is not None and len(vector) != expected_width:
                            raise ValueError(
                                f"vector feature {column.source_feature!r} has width "
                                f"{len(vector)}, expected {expected_width}"
                            )
                        parsed_vectors[column.source_feature] = vector
                    vector = parsed_vectors[column.source_feature]
                    row.append(
                        math.nan if vector is None else vector[int(column.vector_index or 0)]
                    )
            rows.append(row)
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (len(points), len(self.schema.columns)):
            raise AssertionError("encoded matrix shape does not match encoder schema")
        return EncodedBatch(
            point_ids=tuple(point.point_id for point in points),
            matrix=matrix,
            feature_names=self.schema.feature_names,
            categorical_indices=self.schema.categorical_indices,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.schema.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoldTabularEncoder":
        return cls(EncoderSchema.from_dict(value))
