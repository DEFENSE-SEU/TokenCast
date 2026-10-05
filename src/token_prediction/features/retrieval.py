from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from token_prediction.contracts import SourceCapabilities
from token_prediction.dataset.schema import PredictionPoint
from token_prediction.dataset.development import DevelopmentProtocol
from token_prediction.features.reducer import FeatureValue
from token_prediction.identifiers import identifier
from token_prediction.features.text_embedding import _terms


RETRIEVAL_SCHEMA_VERSION = 3
RETRIEVAL_POLICY_ID = "fold_fitted_word_unigram_bigram_tfidf_v3"
RETRIEVAL_FEATURE_NAMES = frozenset(
    {
        "retrieval_total_tokens_median",
        "retrieval_total_tokens_iqr",
        "retrieval_call_count_median",
        "retrieval_mean_similarity",
        "retrieval_neighbor_count",
    }
)
_WORD = re.compile(r"[\w]+", re.UNICODE)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _task_set_id(task_ids: Iterable[str]) -> str:
    tasks = tuple(sorted({_text(task_id, name="task_id") for task_id in task_ids}))
    return identifier("task-set", "v1", "retrieval-fit", "+".join(tasks) or "empty")


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _sparse_vector(
    text: str,
    vocabulary_index: Mapping[str, int],
    idf: Sequence[float],
) -> tuple[tuple[int, float], ...]:
    counts = Counter(term for term in _terms(text) if term in vocabulary_index)
    weighted = {
        vocabulary_index[term]: (1.0 + math.log(count)) * idf[vocabulary_index[term]]
        for term, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return ()
    return tuple(sorted((index, value / norm) for index, value in weighted.items()))


def _cosine(
    left: Sequence[tuple[int, float]],
    right: Sequence[tuple[int, float]],
) -> float:
    left_values = dict(left)
    right_values = dict(right)
    return sum(
        left_values[index] * right_values[index]
        for index in left_values.keys() & right_values.keys()
    )


@dataclass(frozen=True)
class RetrievalExample:
    task_id: str
    task_text: str
    total_tokens: float
    call_count: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, name="task_id"))
        object.__setattr__(self, "task_text", _text(self.task_text, name="task_text"))
        for name in ("total_tokens", "call_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RetrievalFeatures:
    total_tokens_median: float | None
    total_tokens_iqr: float | None
    call_count_median: float | None
    mean_similarity: float | None
    neighbor_count: int

    def __post_init__(self) -> None:
        if isinstance(self.neighbor_count, bool) or not isinstance(self.neighbor_count, int):
            raise TypeError("retrieval neighbor count must be an integer")
        if self.neighbor_count < 0:
            raise ValueError("retrieval neighbor count must be non-negative")
        values = (
            self.total_tokens_median,
            self.total_tokens_iqr,
            self.call_count_median,
            self.mean_similarity,
        )
        if self.neighbor_count == 0 and any(value is not None for value in values):
            raise ValueError("empty retrieval results must use missing feature values")
        if self.neighbor_count > 0 and any(value is None for value in values):
            raise ValueError("non-empty retrieval results require every feature value")
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("retrieval feature values must be finite and non-negative")

    def as_point_features(self) -> dict[str, FeatureValue]:
        return {
            "retrieval_total_tokens_median": self.total_tokens_median,
            "retrieval_total_tokens_iqr": self.total_tokens_iqr,
            "retrieval_call_count_median": self.call_count_median,
            "retrieval_mean_similarity": self.mean_similarity,
            "retrieval_neighbor_count": self.neighbor_count,
        }


@dataclass(frozen=True)
class _RetrievalDocument:
    document_id: str
    task_id: str
    vector: tuple[tuple[int, float], ...]
    total_tokens: float
    call_count: float

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, name="retrieval task_id")
        object.__setattr__(self, "task_id", task_id)
        expected = identifier("retrieval-document", "v1", task_id)
        if self.document_id != expected:
            raise ValueError("retrieval document id does not match its task")
        for name in ("total_tokens", "call_count"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("retrieval document totals must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "task_id": self.task_id,
            "vector": [[index, value] for index, value in self.vector],
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


@dataclass(frozen=True)
class RetrievalProvenance:
    dataset_id: str
    development_protocol_id: str
    split_plan_id: str
    split_seed: int
    outer_fold: int
    capability_profile_id: str
    input_contract_id: str
    fit_task_set_id: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "development_protocol_id",
            "split_plan_id",
            "capability_profile_id",
            "input_contract_id",
            "fit_task_set_id",
        ):
            _text(getattr(self, name), name=name)
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise TypeError("retrieval split seed must be an integer")
        if isinstance(self.outer_fold, bool) or not isinstance(self.outer_fold, int):
            raise TypeError("retrieval outer fold must be an integer")
        if self.outer_fold < 0:
            raise ValueError("retrieval outer fold must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "development_protocol_id": self.development_protocol_id,
            "split_plan_id": self.split_plan_id,
            "split_seed": self.split_seed,
            "outer_fold": self.outer_fold,
            "capability_profile_id": self.capability_profile_id,
            "input_contract_id": self.input_contract_id,
            "fit_task_set_id": self.fit_task_set_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RetrievalProvenance:
        expected = {
            "dataset_id",
            "development_protocol_id",
            "split_plan_id",
            "split_seed",
            "outer_fold",
            "capability_profile_id",
            "input_contract_id",
            "fit_task_set_id",
        }
        if set(value) != expected:
            raise ValueError("retrieval provenance has missing or extra fields")
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in ("split_seed", "outer_fold")
        ):
            raise TypeError("retrieval split seed and outer fold must be integers")
        return cls(
            dataset_id=_text(value["dataset_id"], name="dataset_id"),
            development_protocol_id=_text(
                value["development_protocol_id"], name="development_protocol_id"
            ),
            split_plan_id=_text(value["split_plan_id"], name="split_plan_id"),
            split_seed=value["split_seed"],
            outer_fold=value["outer_fold"],
            capability_profile_id=_text(
                value["capability_profile_id"], name="capability_profile_id"
            ),
            input_contract_id=_text(value["input_contract_id"], name="input_contract_id"),
            fit_task_set_id=_text(value["fit_task_set_id"], name="fit_task_set_id"),
        )


def _outer_train_provenance(
    *,
    development_protocol: DevelopmentProtocol,
    split_seed: int,
    outer_fold: int,
    capabilities: SourceCapabilities,
    input_contract_id: str,
) -> tuple[RetrievalProvenance, frozenset[str]]:
    if (
        development_protocol.parent_capability_profile_id != capabilities.capability_profile_id
        or development_protocol.development_dataset.capability_profile_id
        != capabilities.capability_profile_id
    ):
        raise ValueError("retrieval capabilities do not match the development protocol")
    if (
        development_protocol.parent_input_contract_id != input_contract_id
        or development_protocol.development_dataset.input_contract_id != input_contract_id
    ):
        raise ValueError("retrieval input contract does not match the development protocol")
    matching_plans = tuple(
        plan for plan in development_protocol.outer_plans if plan.seed == split_seed
    )
    if len(matching_plans) != 1:
        raise ValueError("retrieval split seed does not identify one outer split plan")
    split_plan = matching_plans[0]
    if split_plan.dataset_id != development_protocol.development_dataset.dataset_id:
        raise ValueError("retrieval outer split is not bound to the development dataset")
    partition = split_plan.partition(outer_fold)
    return (
        RetrievalProvenance(
            dataset_id=development_protocol.development_dataset.dataset_id,
            development_protocol_id=development_protocol.protocol_id,
            split_plan_id=split_plan.split_plan_id,
            split_seed=split_plan.seed,
            outer_fold=outer_fold,
            capability_profile_id=capabilities.capability_profile_id,
            input_contract_id=input_contract_id,
            fit_task_set_id=_task_set_id(partition.train_tasks),
        ),
        partition.train_tasks,
    )


def _retrieval_id(
    provenance: RetrievalProvenance,
    *,
    policy_id: str,
    k: int,
    max_features: int,
) -> str:
    return identifier(
        "retriever",
        f"v{RETRIEVAL_SCHEMA_VERSION}",
        provenance.development_protocol_id,
        provenance.split_plan_id,
        f"outer-fold={provenance.outer_fold}",
        policy_id,
        f"k={k}",
        f"max-features={max_features}",
    )


@dataclass(frozen=True)
class FoldFittedTfidfRetriever:
    retrieval_id: str
    vocabulary: tuple[str, ...]
    idf: tuple[float, ...]
    documents: tuple[_RetrievalDocument, ...]
    provenance: RetrievalProvenance
    k: int = 5
    max_features: int = 4096
    policy_id: str = RETRIEVAL_POLICY_ID
    schema_version: int = RETRIEVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RETRIEVAL_SCHEMA_VERSION:
            raise ValueError("unsupported retrieval schema version")
        if self.policy_id != RETRIEVAL_POLICY_ID:
            raise ValueError("unsupported retrieval policy")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.k, self.max_features)
        ):
            raise TypeError("retrieval k and max_features must be integers")
        if self.k <= 0 or self.max_features <= 0:
            raise ValueError("retrieval k and max_features must be positive")
        if not self.vocabulary or len(self.vocabulary) != len(self.idf):
            raise ValueError("retrieval vocabulary and idf must be non-empty and aligned")
        if len(self.vocabulary) > self.max_features:
            raise ValueError("retrieval vocabulary exceeds max_features")
        if self.vocabulary != tuple(sorted(self.vocabulary)):
            raise ValueError("retrieval vocabulary must use canonical order")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("retrieval vocabulary terms must be unique")
        if any(not math.isfinite(value) or value <= 0 for value in self.idf):
            raise ValueError("retrieval idf values must be finite and positive")
        task_ids = [document.task_id for document in self.documents]
        if not task_ids or task_ids != sorted(task_ids):
            raise ValueError("retrieval documents must use canonical task order")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("retrieval task ids must be unique")
        if self.provenance.fit_task_set_id != _task_set_id(task_ids):
            raise ValueError("retrieval fit task set differs from its documents")
        for document in self.documents:
            indices = [index for index, _value in document.vector]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("retrieval sparse vectors must use unique canonical indices")
            if any(index < 0 or index >= len(self.vocabulary) for index in indices):
                raise ValueError("retrieval sparse vector index is out of range")
            if any(not math.isfinite(value) or value <= 0 for _index, value in document.vector):
                raise ValueError("retrieval sparse vector values must be finite and positive")
        expected_id = _retrieval_id(
            self.provenance,
            policy_id=self.policy_id,
            k=self.k,
            max_features=self.max_features,
        )
        if self.retrieval_id != expected_id:
            raise ValueError("retrieval id does not match its natural key")

    @property
    def fit_task_set_id(self) -> str:
        return self.provenance.fit_task_set_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "retrieval_id": self.retrieval_id,
            "policy_id": self.policy_id,
            "k": self.k,
            "max_features": self.max_features,
            "provenance": self.provenance.to_dict(),
            "vocabulary": list(self.vocabulary),
            "idf": list(self.idf),
            "documents": [document.to_dict() for document in self.documents],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def validate_provenance(
        self,
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_id: str,
    ) -> None:
        expected, _fit_tasks = _outer_train_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_id=input_contract_id,
        )
        if self.provenance != expected:
            raise ValueError("retrieval provenance does not match the outer train partition")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_id: str,
    ) -> FoldFittedTfidfRetriever:
        expected = {
            "schema_version",
            "retrieval_id",
            "policy_id",
            "k",
            "max_features",
            "provenance",
            "vocabulary",
            "idf",
            "documents",
        }
        if set(value) != expected:
            raise ValueError("retrieval bundle has missing or extra fields")
        for name in ("schema_version", "k", "max_features"):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise TypeError(f"retrieval {name} must be an integer")
        raw_provenance = value["provenance"]
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("retrieval provenance must be an object")
        raw_documents = value["documents"]
        if not isinstance(raw_documents, list):
            raise TypeError("retrieval documents must be an array")
        documents: list[_RetrievalDocument] = []
        document_keys = {"document_id", "task_id", "vector", "total_tokens", "call_count"}
        for raw in raw_documents:
            if not isinstance(raw, Mapping) or set(raw) != document_keys:
                raise ValueError("retrieval document has missing or extra fields")
            raw_vector = raw["vector"]
            if not isinstance(raw_vector, list):
                raise TypeError("retrieval vector must be an array")
            vector: list[tuple[int, float]] = []
            for item in raw_vector:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or isinstance(item[0], bool)
                    or not isinstance(item[0], int)
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], (int, float))
                ):
                    raise TypeError("retrieval vector entries must be [integer, number]")
                vector.append((item[0], float(item[1])))
            if any(
                isinstance(raw[name], bool) or not isinstance(raw[name], (int, float))
                for name in ("total_tokens", "call_count")
            ):
                raise TypeError("retrieval document totals must be numeric")
            documents.append(
                _RetrievalDocument(
                    document_id=_text(raw["document_id"], name="document_id"),
                    task_id=_text(raw["task_id"], name="task_id"),
                    vector=tuple(vector),
                    total_tokens=float(raw["total_tokens"]),
                    call_count=float(raw["call_count"]),
                )
            )
        raw_vocabulary = value["vocabulary"]
        raw_idf = value["idf"]
        if not isinstance(raw_vocabulary, list) or not all(
            isinstance(item, str) for item in raw_vocabulary
        ):
            raise TypeError("retrieval vocabulary must be an array of strings")
        if not isinstance(raw_idf, list) or not all(
            not isinstance(item, bool) and isinstance(item, (int, float)) for item in raw_idf
        ):
            raise TypeError("retrieval idf must be an array of numbers")
        fitted = cls(
            retrieval_id=_text(value["retrieval_id"], name="retrieval_id"),
            vocabulary=tuple(raw_vocabulary),
            idf=tuple(float(item) for item in raw_idf),
            documents=tuple(documents),
            provenance=RetrievalProvenance.from_dict(raw_provenance),
            k=value["k"],
            max_features=value["max_features"],
            policy_id=_text(value["policy_id"], name="policy_id"),
            schema_version=value["schema_version"],
        )
        fitted.validate_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_id=input_contract_id,
        )
        return fitted

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes,
        *,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_id: str,
    ) -> FoldFittedTfidfRetriever:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("retrieval bundle must be UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise TypeError("retrieval bundle root must be an object")
        return cls.from_dict(
            value,
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_id=input_contract_id,
        )

    def transform(
        self,
        *,
        task_id: str,
        task_text: str,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_id: str,
    ) -> RetrievalFeatures:
        self.validate_provenance(
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_id=input_contract_id,
        )
        task_id = _text(task_id, name="retrieval query task_id")
        task_text = _text(task_text, name="retrieval query task_text")
        vocabulary_index = {term: index for index, term in enumerate(self.vocabulary)}
        query = _sparse_vector(task_text, vocabulary_index, self.idf)
        if not query:
            return RetrievalFeatures(None, None, None, None, 0)
        scored = [
            (_cosine(query, document.vector), document)
            for document in self.documents
            if document.task_id != task_id
        ]
        neighbors = [
            (similarity, document)
            for similarity, document in sorted(
                scored,
                key=lambda item: (-item[0], item[1].task_id),
            )[: self.k]
            if similarity > 0
        ]
        if not neighbors:
            return RetrievalFeatures(None, None, None, None, 0)
        totals = [document.total_tokens for _similarity, document in neighbors]
        calls = [document.call_count for _similarity, document in neighbors]
        similarities = [similarity for similarity, _document in neighbors]
        return RetrievalFeatures(
            total_tokens_median=float(median(totals)),
            total_tokens_iqr=_quantile(totals, 0.75) - _quantile(totals, 0.25),
            call_count_median=float(median(calls)),
            mean_similarity=sum(similarities) / len(similarities),
            neighbor_count=len(neighbors),
        )

    def transform_point(
        self,
        point: PredictionPoint,
        *,
        task_text: str,
        development_protocol: DevelopmentProtocol,
        split_seed: int,
        outer_fold: int,
        capabilities: SourceCapabilities,
        input_contract_id: str,
    ) -> PredictionPoint:
        overlap = RETRIEVAL_FEATURE_NAMES & set(point.features)
        if overlap:
            raise ValueError(
                f"point already contains retrieval features: {', '.join(sorted(overlap))}"
            )
        features = self.transform(
            task_id=point.task_id,
            task_text=task_text,
            development_protocol=development_protocol,
            split_seed=split_seed,
            outer_fold=outer_fold,
            capabilities=capabilities,
            input_contract_id=input_contract_id,
        )
        return point.with_features({**point.features, **features.as_point_features()})


def fit_fold_tfidf_retriever(
    examples: Iterable[RetrievalExample],
    *,
    capabilities: SourceCapabilities,
    input_contract_id: str,
    development_protocol: DevelopmentProtocol,
    split_seed: int,
    outer_fold: int,
    k: int = 5,
    max_features: int = 4096,
) -> FoldFittedTfidfRetriever:
    provenance, fit_task_ids = _outer_train_provenance(
        development_protocol=development_protocol,
        split_seed=split_seed,
        outer_fold=outer_fold,
        capabilities=capabilities,
        input_contract_id=input_contract_id,
    )
    if k <= 0 or max_features <= 0:
        raise ValueError("retrieval k and max_features must be positive")
    resolved = tuple(examples)
    if len(resolved) < 2:
        raise ValueError("retrieval fit requires at least two tasks")
    task_ids = [example.task_id for example in resolved]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("retrieval fit task ids must be unique")
    if frozenset(task_ids) != fit_task_ids:
        raise ValueError("retrieval examples must exactly match the outer-train partition")

    # Vocabulary and IDF inherit the same training boundary as the retrieved labels.
    term_sets = [set(_terms(example.task_text)) for example in resolved]
    document_frequency = Counter(term for terms in term_sets for term in terms)
    ranked = sorted(document_frequency, key=lambda term: (-document_frequency[term], term))
    vocabulary = tuple(sorted(ranked[:max_features]))
    if not vocabulary:
        raise ValueError("retrieval task text produced an empty vocabulary")
    idf = tuple(
        math.log((1 + len(resolved)) / (1 + document_frequency[term])) + 1.0 for term in vocabulary
    )
    vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
    documents = tuple(
        sorted(
            (
                _RetrievalDocument(
                    document_id=identifier("retrieval-document", "v1", example.task_id),
                    task_id=example.task_id,
                    vector=_sparse_vector(example.task_text, vocabulary_index, idf),
                    total_tokens=float(example.total_tokens),
                    call_count=float(example.call_count),
                )
                for example in resolved
            ),
            key=lambda document: document.task_id,
        )
    )
    return FoldFittedTfidfRetriever(
        retrieval_id=_retrieval_id(
            provenance,
            policy_id=RETRIEVAL_POLICY_ID,
            k=k,
            max_features=max_features,
        ),
        vocabulary=vocabulary,
        idf=idf,
        documents=documents,
        provenance=provenance,
        k=k,
        max_features=max_features,
    )


__all__ = [
    "RETRIEVAL_FEATURE_NAMES",
    "RETRIEVAL_POLICY_ID",
    "RETRIEVAL_SCHEMA_VERSION",
    "FoldFittedTfidfRetriever",
    "RetrievalExample",
    "RetrievalFeatures",
    "RetrievalProvenance",
    "fit_fold_tfidf_retriever",
]
