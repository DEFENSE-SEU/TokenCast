from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from token_prediction.dataset.schema import PredictionPoint

TEXT_EMBEDDING_METHODS = ("bge", "e5", "tfidf_svd")
TEXT_EMBEDDING_SCHEMA_VERSION = 1
_MAX_VOCABULARY = 20_000


def _texts(points: Sequence[PredictionPoint], source_feature: str) -> list[str]:
    return [
        value
        for point in points
        if isinstance(value := point.features.get(source_feature), str) and value
    ]


@dataclass(frozen=True, eq=False)
class TextEmbeddingTransform:
    source_feature: str
    method: str
    dim: int
    vocabulary: tuple[str, ...] = ()
    idf: Any = None
    components: Any = None

    def __post_init__(self) -> None:
        if self.method not in TEXT_EMBEDDING_METHODS:
            raise ValueError(f"unsupported text embedding method {self.method!r}")
        if not self.source_feature.strip():
            raise ValueError("text embedding transform requires a source feature name")
        if isinstance(self.dim, bool) or int(self.dim) < 1:
            raise ValueError("text embedding dim must be a positive integer")
        if self.method == "tfidf_svd" and (
            not self.vocabulary or self.idf is None or self.components is None
        ):
            raise ValueError("tfidf_svd transform requires a fitted vocabulary")

    @property
    def feature_name(self) -> str:
        return f"{self.source_feature}_{self.method}_embedding"

    @classmethod
    def fit(
        cls,
        points: Sequence[PredictionPoint],
        *,
        source_feature: str,
        method: str,
        dim: int = 256,
        seed: int = 0,
    ) -> "TextEmbeddingTransform":
        if method not in TEXT_EMBEDDING_METHODS:
            raise ValueError(f"unsupported text embedding method {method!r}")
        texts = _texts(points, source_feature)
        if not texts:
            raise ValueError(f"text embedding transform found no training text for {source_feature!r}")
        if method == "bge":
            from token_prediction.features.bge_embedding import BGE_EMBEDDING_DIM

            return cls(source_feature=source_feature, method=method, dim=BGE_EMBEDDING_DIM)
        if method == "e5":
            from token_prediction.features.e5_embedding import E5_EMBEDDING_DIM

            return cls(source_feature=source_feature, method=method, dim=E5_EMBEDDING_DIM)

        import numpy as np
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=_MAX_VOCABULARY)
        matrix = vectorizer.fit_transform(texts)
        n_components = max(1, min(int(dim), matrix.shape[0] - 1, matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        svd.fit(matrix)
        return cls(
            source_feature=source_feature,
            method=method,
            dim=n_components,
            vocabulary=tuple(vectorizer.get_feature_names_out()),
            idf=np.asarray(vectorizer.idf_, dtype=np.float64),
            components=np.asarray(svd.components_, dtype=np.float64),
        )

    def _embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if self.method == "bge":
            from token_prediction.features.bge_embedding import embed_texts

            return embed_texts(texts)
        if self.method == "e5":
            from token_prediction.features.e5_embedding import embed_texts

            return embed_texts(texts)

        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(vocabulary={term: index for index, term in enumerate(self.vocabulary)})
        counts = vectorizer.transform(texts).toarray().astype(np.float64)
        weighted = counts * self.idf[None, :]
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        normalized = np.divide(weighted, norms, out=np.zeros_like(weighted), where=norms > 0)
        projected = normalized @ self.components.T
        return [tuple(float(v) for v in row) for row in projected]

    def apply_many(self, points: Sequence[PredictionPoint]) -> tuple[PredictionPoint, ...]:
        indexed = [
            index
            for index, point in enumerate(points)
            if isinstance(point.features.get(self.source_feature), str)
            and point.features[self.source_feature]
        ]
        vectors: dict[int, tuple[float, ...]] = {}
        if indexed:
            embedded = self._embed([points[index].features[self.source_feature] for index in indexed])
            vectors = dict(zip(indexed, embedded))
        return tuple(
            point.with_features({**point.features, self.feature_name: vectors.get(index)})
            for index, point in enumerate(points)
        )

    def apply(self, point: PredictionPoint) -> PredictionPoint:
        return self.apply_many((point,))[0]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": TEXT_EMBEDDING_SCHEMA_VERSION,
            "source_feature": self.source_feature,
            "method": self.method,
            "dim": self.dim,
            "vocabulary_size": len(self.vocabulary),
        }

    def to_bytes(self) -> bytes:


        if self.method != "tfidf_svd":
            return b""
        import io

        import numpy as np

        buffer = io.BytesIO()
        np.savez(
            buffer,
            vocabulary=np.asarray(self.vocabulary, dtype=str),
            idf=self.idf,
            components=self.components,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, manifest: Mapping[str, Any], payload: bytes) -> "TextEmbeddingTransform":
        schema_version = manifest.get("schema_version", 1)
        if schema_version != TEXT_EMBEDDING_SCHEMA_VERSION:
            raise ValueError("unsupported text embedding schema version")
        source_feature = str(manifest["source_feature"])
        method = str(manifest["method"])
        dim = int(manifest["dim"])
        if method != "tfidf_svd":
            return cls(source_feature=source_feature, method=method, dim=dim)
        import io

        import numpy as np

        with np.load(io.BytesIO(payload)) as archive:
            return cls(
                source_feature=source_feature,
                method=method,
                dim=dim,
                vocabulary=tuple(str(item) for item in archive["vocabulary"]),
                idf=np.asarray(archive["idf"], dtype=np.float64),
                components=np.asarray(archive["components"], dtype=np.float64),
            )


__all__ = [
    "TEXT_EMBEDDING_METHODS",
    "TEXT_EMBEDDING_SCHEMA_VERSION",
    "TextEmbeddingTransform",
]
