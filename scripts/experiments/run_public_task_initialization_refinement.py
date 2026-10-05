from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from scripts.experiments.run_public_task_initialization_validation import (
    TaskStartExample,
    _build_examples,
    _constant_forecasts,
    _empirical_interval,
    _fit_log_residual_interval,
    _fit_scale,
    _fold_partition,
    _prediction_document,
    _score_forecasts,
)
from scripts.experiments.run_public_trajectory_validation import (
    _assign_folds,
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast
from token_prediction.evaluation import (
    CalibrationExample,
    ScoredForecast,
    evaluate_forecasts,
    fit_calibrator,
    paired_task_bootstrap,
)
from token_prediction.experiments.candidates import PredictionRecord
from token_prediction.config import project_root_for


CONDITION_ID = "public-task-start-full-prompt-v1"
TARGET = PredictionTarget.TASK_UNKNOWN_REMAINING_TOKENS
TOKEN_UNIT_CONTRACT_ID = "provider-output-token-v1"
EMPIRICAL_ID = "empirical_quantile"
ADDITIONAL_REFERENCE_KEYS = (
    "group_reference",
    "retrieval_reference",
    "quantile_tree_reference",
    "extra_trees_reference",
    "embedding_reference",
    "proxy_lm_reference",
)

ENCODER_WINDOW_TOKENS = 512


@dataclass(frozen=True)
class LaunchMetadata:
    task_id: str
    repo: str
    repo_language: str
    problem_statement: str
    requirements: str
    interface: str


def _additional_reference_ids(refinement: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(refinement[key]) for key in ADDITIONAL_REFERENCE_KEYS if key in refinement)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_validation_v1.toml"),
    )
    parser.add_argument("--trace-data", type=Path)
    parser.add_argument("--task-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _load_metadata(
    path: Path,
    selected_task_ids: frozenset[str],
) -> dict[str, LaunchMetadata]:
    columns = [
        "instance_id",
        "repo",
        "repo_language",
        "problem_statement",
        "requirements",
        "interface",
    ]
    table = pq.read_table(
        path,
        columns=columns,
        filters=[("instance_id", "in", sorted(selected_task_ids))],
    )
    rows = table.to_pylist()
    observed = {str(row["instance_id"]) for row in rows}
    if observed != selected_task_ids:
        missing = sorted(selected_task_ids - observed)
        raise ValueError(f"selected Task-start metadata is missing: {missing[:3]}")
    result = {}
    for row in rows:
        values = {name: row[name] for name in columns[1:]}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("refined Task-start metadata fields must be non-empty strings")
        task_id = str(row["instance_id"])
        result[task_id] = LaunchMetadata(task_id=task_id, **values)
    return result


def _prompt_document(metadata: LaunchMetadata) -> str:
    return (
        f"{metadata.problem_statement}\n\n"
        f"Requirements:\n{metadata.requirements}\n\n"
        f"New interfaces introduced:\n{metadata.interface}"
    )


def _structure_values(metadata: LaunchMetadata) -> tuple[float, ...]:
    prompt = _prompt_document(metadata)
    texts = (
        metadata.problem_statement,
        metadata.requirements,
        metadata.interface,
        prompt,
    )
    return (
        *(math.log1p(len(text)) for text in texts),
        *(math.log1p(len(text.split())) for text in texts),
        math.log1p(metadata.requirements.count("\n-") + metadata.requirements.count(" - ")),
        math.log1p(metadata.interface.lower().count("path")),
        math.log1p(metadata.interface.lower().count("type:")),
        math.log1p(prompt.count("`")),
        math.log1p(prompt.count("\n")),
    )


_PROMPT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def _torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_pretrained_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


@lru_cache(maxsize=1)
def _load_text_encoder(model_id: str, revision: str):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, revision=revision)
    model.eval()
    device = _torch_device()
    model.to(device)
    return _load_pretrained_tokenizer(model_id, revision), model, torch.device(device)


def _load_proxy_backbone(model_id: str, revision: str):


    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, revision=revision)


def _prompt_embedding(text: str, tokenizer, model, device) -> np.ndarray:
    import torch

    content_window = ENCODER_WINDOW_TOKENS - 2
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    windows = [
        token_ids[start : start + content_window]
        for start in range(0, len(token_ids), content_window)
    ] or [[]]
    vectors = []
    weights = []
    with torch.no_grad():
        for window_ids in windows:
            input_ids = torch.tensor(
                [[tokenizer.cls_token_id, *window_ids, tokenizer.sep_token_id]],
                device=device,
            )
            hidden = model(input_ids=input_ids).last_hidden_state[0]
            vectors.append(hidden.mean(dim=0))
            weights.append(float(len(window_ids) + 2))
    stacked = torch.stack(vectors)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    pooled = (stacked * weight_tensor[:, None]).sum(dim=0) / weight_tensor.sum()
    return pooled.cpu().numpy().astype(np.float64)


def _fit_embedding_reference(
    train: Sequence[TaskStartExample],
    query: Sequence[TaskStartExample],
    *,
    refinement: Mapping[str, object],
) -> np.ndarray:
    tokenizer, model, device = _load_text_encoder(
        str(refinement["embedding_model_id"]),
        str(refinement["embedding_model_revision"]),
    )
    for example in (*train, *query):
        if example.task_id not in _PROMPT_EMBEDDING_CACHE:
            _PROMPT_EMBEDDING_CACHE[example.task_id] = _prompt_embedding(
                _prompt_document(example.metadata), tokenizer, model, device
            )
    train_matrix = np.stack([_PROMPT_EMBEDDING_CACHE[example.task_id] for example in train])
    query_matrix = np.stack([_PROMPT_EMBEDDING_CACHE[example.task_id] for example in query])
    train_target = np.log1p([example.total_output_tokens for example in train])
    ridge = Ridge(alpha=float(refinement["embedding_ridge_alpha"]))
    ridge.fit(train_matrix, train_target)
    return np.maximum(1.0, np.expm1(ridge.predict(query_matrix)))


def _fit_proxy_lm_reference(
    train: Sequence[TaskStartExample],
    query: Sequence[TaskStartExample],
    *,
    refinement: Mapping[str, object],
) -> np.ndarray:
    import torch

    model_id = str(refinement["proxy_lm_model_id"])
    revision = str(refinement["proxy_lm_model_revision"])
    seed = int(refinement["proxy_lm_seed"])
    epochs = int(refinement["proxy_lm_epochs"])
    batch_size = int(refinement["proxy_lm_batch_size"])
    tokenizer = _load_pretrained_tokenizer(model_id, revision)
    device = _torch_device()
    autocast_enabled = device == "cuda"

    def encode(examples: Sequence[TaskStartExample]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokenizer(
            [_prompt_document(example.metadata) for example in examples],
            truncation=True,
            max_length=ENCODER_WINDOW_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        return batch["input_ids"], batch["attention_mask"]

    torch.manual_seed(seed)
    backbone = _load_proxy_backbone(model_id, revision).to(device)
    head = torch.nn.Linear(backbone.config.hidden_size, 1).to(device)

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = backbone(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).last_hidden_state[:, 0]
        return head(hidden).squeeze(-1)

    train_ids, train_mask = encode(train)
    train_target = torch.tensor(
        np.log1p([example.total_output_tokens for example in train]),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [*backbone.parameters(), *head.parameters()],
        lr=float(refinement["proxy_lm_learning_rate"]),
    )
    shuffle = torch.Generator().manual_seed(seed)
    backbone.train()
    for _ in range(epochs):
        order = torch.randperm(len(train), generator=shuffle)
        for start in range(0, len(train), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = torch.nn.functional.mse_loss(
                    forward(train_ids[index], train_mask[index]).float(),
                    train_target[index],
                )
            loss.backward()
            optimizer.step()

    backbone.eval()
    query_ids, query_mask = encode(query)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(query), batch_size):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = forward(
                    query_ids[start : start + batch_size],
                    query_mask[start : start + batch_size],
                )
            predictions.append(logits.float().cpu().numpy())
    return np.maximum(1.0, np.expm1(np.concatenate(predictions).astype(np.float64)))


def _fit_score_models(
    train: Sequence[TaskStartExample],
    query: Sequence[TaskStartExample],
    *,
    refinement: Mapping[str, object],
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    started = time.perf_counter()
    train_metadata = [example.metadata for example in train]
    query_metadata = [example.metadata for example in query]
    train_output = np.asarray([example.total_output_tokens for example in train], dtype=np.float64)
    train_target = np.log1p(train_output)

    train_lengths = np.log1p([len(_prompt_document(metadata)) for metadata in train_metadata])
    query_lengths = np.log1p([len(_prompt_document(metadata)) for metadata in query_metadata])
    length_design = np.column_stack((np.ones(len(train)), train_lengths))
    length_coefficients, *_ = np.linalg.lstsq(length_design, train_target, rcond=None)
    length_scores = np.maximum(
        1.0,
        np.expm1(np.column_stack((np.ones(len(query)), query_lengths)) @ length_coefficients),
    )

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=12_000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=12_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    text_train = vectorizer.fit_transform(
        [_prompt_document(metadata) for metadata in train_metadata]
    )
    text_query = vectorizer.transform([_prompt_document(metadata) for metadata in query_metadata])
    text_model = Ridge(alpha=float(refinement["text_ridge_alpha"]), solver="lsqr")
    text_model.fit(text_train, train_target)
    text_scores = np.maximum(1.0, np.expm1(text_model.predict(text_query)))
    residual_model = Ridge(alpha=float(refinement["residual_text_ridge_alpha"]), solver="lsqr")
    residual_model.fit(
        text_train,
        train_target - np.log1p(np.maximum(1.0, np.expm1(length_design @ length_coefficients))),
    )
    residual_scores = np.maximum(
        1.0,
        np.expm1(np.log1p(length_scores) + residual_model.predict(text_query)),
    )

    category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_categories = [[metadata.repo, metadata.repo_language] for metadata in train_metadata]
    query_categories = [[metadata.repo, metadata.repo_language] for metadata in query_metadata]
    category_encoder.fit(train_categories)
    structure_train = np.column_stack(
        (
            np.asarray([_structure_values(metadata) for metadata in train_metadata]),
            category_encoder.transform(train_categories),
        )
    )
    structure_query = np.column_stack(
        (
            np.asarray([_structure_values(metadata) for metadata in query_metadata]),
            category_encoder.transform(query_categories),
        )
    )
    tree_parameters = {
        "n_estimators": int(refinement["structure_estimators"]),
        "learning_rate": float(refinement["structure_learning_rate"]),
        "max_depth": int(refinement["structure_max_depth"]),
        "min_samples_leaf": int(refinement["structure_min_samples_leaf"]),
        "random_state": 1,
    }
    structure_model = GradientBoostingRegressor(
        loss=str(refinement["structure_loss"]),
        **tree_parameters,
    )
    structure_model.fit(structure_train, train_target)
    structure_scores = np.maximum(1.0, np.expm1(structure_model.predict(structure_query)))

    additional_scores: dict[str, np.ndarray] = {}
    if "group_reference" in refinement:
        repo_targets: dict[str, list[float]] = defaultdict(list)
        language_targets: dict[str, list[float]] = defaultdict(list)
        for metadata, target in zip(train_metadata, train_output):
            repo_targets[metadata.repo].append(float(target))
            language_targets[metadata.repo_language].append(float(target))
        repo_medians = {repo: float(np.median(values)) for repo, values in repo_targets.items()}
        language_medians = {
            language: float(np.median(values)) for language, values in language_targets.items()
        }
        global_median = float(np.median(train_output))
        additional_scores[str(refinement["group_reference"])] = np.asarray(
            [
                repo_medians.get(
                    metadata.repo,
                    language_medians.get(metadata.repo_language, global_median),
                )
                for metadata in query_metadata
            ],
            dtype=np.float64,
        )
    if "retrieval_reference" in refinement:
        retrieval_model = KNeighborsRegressor(
            n_neighbors=min(int(refinement["retrieval_neighbors"]), len(train)),
            weights="distance",
            algorithm="brute",
            metric="cosine",
        )
        retrieval_model.fit(text_train, train_target)
        additional_scores[str(refinement["retrieval_reference"])] = np.maximum(
            1.0,
            np.expm1(retrieval_model.predict(text_query)),
        )
    if "quantile_tree_reference" in refinement:
        quantile_tree = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.5,
            **tree_parameters,
        )
        quantile_tree.fit(structure_train, train_target)
        additional_scores[str(refinement["quantile_tree_reference"])] = np.maximum(
            1.0,
            np.expm1(quantile_tree.predict(structure_query)),
        )
    if "extra_trees_reference" in refinement:
        extra_trees = ExtraTreesRegressor(
            n_estimators=int(refinement["extra_trees_estimators"]),
            min_samples_leaf=int(refinement["extra_trees_min_samples_leaf"]),
            max_features=float(refinement["extra_trees_max_features"]),
            random_state=1,
            n_jobs=1,
        )
        extra_trees.fit(structure_train, train_target)
        additional_scores[str(refinement["extra_trees_reference"])] = np.maximum(
            1.0,
            np.expm1(extra_trees.predict(structure_query)),
        )
    if "embedding_reference" in refinement:
        additional_scores[str(refinement["embedding_reference"])] = _fit_embedding_reference(
            train,
            query,
            refinement=refinement,
        )
    if "proxy_lm_reference" in refinement:
        additional_scores[str(refinement["proxy_lm_reference"])] = _fit_proxy_lm_reference(
            train,
            query,
            refinement=refinement,
        )

    query_ids = [example.task_id for example in query]
    component_ids = (
        str(refinement["length_component"]),
        str(refinement["text_component"]),
        str(refinement["structure_component"]),
    )
    arrays = (length_scores, text_scores, structure_scores)
    score_maps = {
        candidate_id: {task_id: float(value) for task_id, value in zip(query_ids, values)}
        for candidate_id, values in zip(component_ids, arrays)
    }
    score_maps.update(
        {
            candidate_id: {task_id: float(value) for task_id, value in zip(query_ids, values)}
            for candidate_id, values in additional_scores.items()
        }
    )
    score_maps[str(refinement["candidate"])] = {
        task_id: float(
            math.exp(
                sum(math.log(score_maps[candidate_id][task_id]) for candidate_id in component_ids)
                / len(component_ids)
            )
        )
        for task_id in query_ids
    }
    score_maps[str(refinement["residual_candidate"])] = {
        task_id: float(value) for task_id, value in zip(query_ids, residual_scores)
    }
    return score_maps, {
        "fit_tasks": len(train),
        "query_tasks": len(query),
        "word_features": len(vectorizer.transformer_list[0][1].vocabulary_),
        "char_features": len(vectorizer.transformer_list[1][1].vocabulary_),
        "structure_features": int(structure_train.shape[1]),
        "additional_references": list(additional_scores),
        "fit_latency_seconds": time.perf_counter() - started,
    }


def _record(
    candidate_id: str,
    example: TaskStartExample,
    forecast: object,
) -> PredictionRecord:
    return PredictionRecord(
        candidate_id=candidate_id,
        point_id=example.point_id,
        task_id=example.task_id,
        trajectory_id=example.task_id,
        condition_id=CONDITION_ID,
        fold=example.fold,
        target=TARGET,
        forecast=forecast,
        sample_weight=1.0,
        token_unit_contract_id=TOKEN_UNIT_CONTRACT_ID,
        task_family_id=example.task_id,
        split_group_id=example.task_id,
        task_type=example.metadata.repo_language,
        actor_model_id=example.model,
        model_family_id=example.model,
    )


def _calibrate_and_score(
    candidate_id: str,
    calibration: Sequence[TaskStartExample],
    test: Sequence[TaskStartExample],
    calibration_forecasts: Mapping[str, object],
    test_forecasts: Mapping[str, object],
    *,
    alpha: float,
) -> tuple[list[PredictionRecord], list[ScoredForecast], dict[str, object]]:
    calibrator = fit_calibrator(
        "task_max_log_conformal",
        alpha=alpha,
        examples=[
            CalibrationExample(
                task_id=example.task_id,
                forecast=calibration_forecasts[example.point_id],
                target_value=example.total_output_tokens,
            )
            for example in calibration
        ],
    )
    records = []
    scored = []
    for example in test:
        forecast = calibrator.transform(test_forecasts[example.point_id])
        records.append(_record(candidate_id, example, forecast))
        scored.append(
            ScoredForecast(
                task_id=example.task_id,
                trajectory_id=example.task_id,
                forecast=forecast,
                target_value=example.total_output_tokens,
                sample_weight=1.0,
                task_family_id=example.task_id,
                split_group_id=example.task_id,
                task_type=example.metadata.repo_language,
                actor_model_id=example.model,
                model_family_id=example.model,
            )
        )
    return records, scored, {"calibrator": calibrator.to_dict()}


def _summarize(
    records: Mapping[str, Sequence[PredictionRecord]],
    scored: Mapping[str, Sequence[ScoredForecast]],
    truth: Mapping[str, float],
    *,
    candidate_id: str,
    references: Sequence[str],
    alpha: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    return {
        "metrics": {
            arm_id: evaluate_forecasts(rows, alpha=alpha) for arm_id, rows in scored.items()
        },
        "comparisons": {
            f"{candidate_id}_vs_{reference_id}": asdict(
                paired_task_bootstrap(
                    records[candidate_id],
                    records[reference_id],
                    truth,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
            )
            for reference_id in references
        },
    }


def _add_hybrid_arm(
    records: dict[str, list[PredictionRecord]],
    scored: dict[str, list[ScoredForecast]],
    *,
    point_arm_id: str,
    interval_arm_id: str,
    hybrid_id: str,
) -> None:
    interval_records = {record.point_id: record for record in records[interval_arm_id]}
    hybrid_records = []
    for point_record in records[point_arm_id]:
        interval = interval_records[point_record.point_id].forecast
        point = point_record.forecast.point
        raw_point = point_record.forecast.raw_point or point
        raw_lower = interval.raw_lower if interval.raw_lower is not None else interval.lower
        raw_upper = interval.raw_upper if interval.raw_upper is not None else interval.upper
        forecast = TokenForecast(
            point_id=point_record.point_id,
            target=point_record.target,
            lower=min(interval.lower, point),
            point=point,
            upper=max(interval.upper, point),
            latency_ms=max(point_record.forecast.latency_ms, interval.latency_ms),
            raw_lower=min(raw_lower, raw_point),
            raw_point=raw_point,
            raw_upper=max(raw_upper, raw_point),
        )
        hybrid_records.append(replace(point_record, candidate_id=hybrid_id, forecast=forecast))
    hybrid_by_task = {record.task_id: record.forecast for record in hybrid_records}
    records[hybrid_id] = hybrid_records
    scored[hybrid_id] = [
        replace(
            point_row,
            forecast=hybrid_by_task[point_row.task_id],
        )
        for point_row in scored[point_arm_id]
    ]


def _run_selection_screen(
    examples: Sequence[TaskStartExample],
    *,
    folds: int,
    alpha: float,
    refinement: Mapping[str, object],
    protocol: Mapping[str, object],
    candidate_id: str | None = None,
    learned_ids: Sequence[str] | None = None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, list[PredictionRecord]],
]:
    records: dict[str, list[PredictionRecord]] = defaultdict(list)
    scored: dict[str, list[ScoredForecast]] = defaultdict(list)
    fold_reports = []
    resolved_learned_ids = tuple(
        learned_ids
        or (
            str(refinement["length_component"]),
            str(refinement["text_component"]),
            str(refinement["structure_component"]),
            str(refinement["candidate"]),
        )
    )
    for fold in range(folds):
        train, validation, calibration, test = _fold_partition(examples, fold, folds)
        query = [*validation, *calibration, *test]
        score_maps, fit_report = _fit_score_models(
            train,
            query,
            refinement=refinement,
        )
        arm_reports = {}
        for arm_id in resolved_learned_ids:
            scale = _fit_scale(score_maps[arm_id], validation)
            residuals = _fit_log_residual_interval(
                score_maps[arm_id],
                scale,
                validation,
                alpha=alpha,
            )
            calibration_forecasts, _ = _score_forecasts(
                score_maps[arm_id],
                calibration,
                scale=scale,
                residual_interval=residuals,
            )
            test_forecasts, latency_ms = _score_forecasts(
                score_maps[arm_id],
                test,
                scale=scale,
                residual_interval=residuals,
            )
            arm_records, arm_scored, report = _calibrate_and_score(
                arm_id,
                calibration,
                test,
                calibration_forecasts,
                test_forecasts,
                alpha=alpha,
            )
            records[arm_id].extend(arm_records)
            scored[arm_id].extend(arm_scored)
            arm_reports[arm_id] = {
                "validation_mae_scale": scale,
                "validation_log_residual_interval": list(residuals),
                "prediction_latency_ms_per_task": latency_ms,
                **report,
            }
        empirical = _empirical_interval(train, alpha=alpha)
        calibration_forecasts, _ = _constant_forecasts(empirical, calibration)
        test_forecasts, _ = _constant_forecasts(empirical, test)
        arm_records, arm_scored, report = _calibrate_and_score(
            EMPIRICAL_ID,
            calibration,
            test,
            calibration_forecasts,
            test_forecasts,
            alpha=alpha,
        )
        records[EMPIRICAL_ID].extend(arm_records)
        scored[EMPIRICAL_ID].extend(arm_scored)
        arm_reports[EMPIRICAL_ID] = {"train_empirical_interval": list(empirical), **report}
        fold_reports.append(
            {
                "fold": fold,
                "task_counts": {
                    "train": len(train),
                    "validation": len(validation),
                    "calibration": len(calibration),
                    "test": len(test),
                },
                "model_fit": fit_report,
                "arms": arm_reports,
            }
        )
    truth = {example.point_id: example.total_output_tokens for example in examples}
    residual_id = str(refinement["residual_candidate"])
    fusion_id = str(refinement["candidate"])
    hybrid_id = str(refinement["hybrid_candidate"])
    if residual_id in resolved_learned_ids and fusion_id in resolved_learned_ids:
        _add_hybrid_arm(
            records,
            scored,
            point_arm_id=residual_id,
            interval_arm_id=fusion_id,
            hybrid_id=hybrid_id,
        )
    resolved_candidate_id = candidate_id or str(refinement["candidate"])
    return (
        _summarize(
            records,
            scored,
            truth,
            candidate_id=resolved_candidate_id,
            references=(
                *(arm_id for arm_id in resolved_learned_ids if arm_id != resolved_candidate_id),
                EMPIRICAL_ID,
            ),
            alpha=alpha,
            bootstrap_iterations=int(protocol["bootstrap_iterations"]),
            bootstrap_seed=int(protocol["bootstrap_seed"]),
        ),
        fold_reports,
        records,
    )


def _fit_oof_mappings(
    examples: Sequence[TaskStartExample],
    *,
    folds: int,
    alpha: float,
    refinement: Mapping[str, object],
) -> tuple[dict[str, tuple[float, tuple[float, float]]], list[dict[str, object]]]:
    learned_ids = (
        str(refinement["length_component"]),
        str(refinement["text_component"]),
        str(refinement["structure_component"]),
        str(refinement["candidate"]),
    )
    scores: dict[str, dict[str, float]] = {arm_id: {} for arm_id in learned_ids}
    reports = []
    for fold in range(folds):
        query = [example for example in examples if example.fold == fold]
        train = [example for example in examples if example.fold != fold]
        fold_scores, report = _fit_score_models(train, query, refinement=refinement)
        for arm_id in learned_ids:
            scores[arm_id].update(fold_scores[arm_id])
        reports.append({"fold": fold, "model_fit": report})
    if any(set(values) != {example.task_id for example in examples} for values in scores.values()):
        raise ValueError("OOF Task-start scores do not cover the pilot cohort exactly")
    return (
        {
            arm_id: (
                (scale := _fit_scale(arm_scores, examples)),
                _fit_log_residual_interval(
                    arm_scores,
                    scale,
                    examples,
                    alpha=alpha,
                ),
            )
            for arm_id, arm_scores in scores.items()
        },
        reports,
    )


def _run_confirmation(
    pilot: Sequence[TaskStartExample],
    reserve: Sequence[TaskStartExample],
    *,
    folds: int,
    alpha: float,
    refinement: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, list[PredictionRecord]], dict[str, object]]:
    calibration_fold = int(refinement["confirmation_calibration_fold"])
    calibration = [example for example in reserve if example.fold == calibration_fold]
    test = [example for example in reserve if example.fold != calibration_fold]
    mappings, oof_reports = _fit_oof_mappings(
        pilot,
        folds=folds,
        alpha=alpha,
        refinement=refinement,
    )
    score_maps, final_fit = _fit_score_models(pilot, reserve, refinement=refinement)
    learned_ids = tuple(mappings)
    records: dict[str, list[PredictionRecord]] = defaultdict(list)
    scored: dict[str, list[ScoredForecast]] = defaultdict(list)
    arm_reports = {}
    for arm_id in learned_ids:
        scale, residuals = mappings[arm_id]
        calibration_forecasts, _ = _score_forecasts(
            score_maps[arm_id],
            calibration,
            scale=scale,
            residual_interval=residuals,
        )
        test_forecasts, latency_ms = _score_forecasts(
            score_maps[arm_id],
            test,
            scale=scale,
            residual_interval=residuals,
        )
        arm_records, arm_scored, report = _calibrate_and_score(
            arm_id,
            calibration,
            test,
            calibration_forecasts,
            test_forecasts,
            alpha=alpha,
        )
        records[arm_id].extend(arm_records)
        scored[arm_id].extend(arm_scored)
        arm_reports[arm_id] = {
            "oof_mae_scale": scale,
            "oof_log_residual_interval": list(residuals),
            "prediction_latency_ms_per_task": latency_ms,
            **report,
        }
    empirical = _empirical_interval(pilot, alpha=alpha)
    calibration_forecasts, _ = _constant_forecasts(empirical, calibration)
    test_forecasts, _ = _constant_forecasts(empirical, test)
    arm_records, arm_scored, report = _calibrate_and_score(
        EMPIRICAL_ID,
        calibration,
        test,
        calibration_forecasts,
        test_forecasts,
        alpha=alpha,
    )
    records[EMPIRICAL_ID].extend(arm_records)
    scored[EMPIRICAL_ID].extend(arm_scored)
    arm_reports[EMPIRICAL_ID] = {"pilot_empirical_interval": list(empirical), **report}

    truth = {example.point_id: example.total_output_tokens for example in test}
    candidate_id = str(refinement["candidate"])
    result = _summarize(
        records,
        scored,
        truth,
        candidate_id=candidate_id,
        references=(*learned_ids[:-1], EMPIRICAL_ID),
        alpha=alpha,
        bootstrap_iterations=int(protocol["bootstrap_iterations"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
    )
    return (
        result,
        records,
        {
            "task_counts": {
                "pilot_fit_and_oof_mapping": len(pilot),
                "confirmation_calibration": len(calibration),
                "confirmation_test": len(test),
            },
            "oof_fit_reports": oof_reports,
            "final_model_fit": final_fit,
            "arms": arm_reports,
        },
    )


def _diagnostics(examples: Sequence[TaskStartExample]) -> dict[str, object]:
    target = np.asarray([example.total_output_tokens for example in examples])
    problem_length = np.asarray([len(example.metadata.problem_statement) for example in examples])
    prompt_length = np.asarray([len(_prompt_document(example.metadata)) for example in examples])
    call_count = np.asarray([example.call_count for example in examples])

    def correlation(values: np.ndarray) -> dict[str, float]:
        coefficient, p_value = spearmanr(values, target)
        return {"spearman": float(coefficient), "p_value": float(p_value)}

    return {
        "target_output_tokens": {
            "mean": float(np.mean(target)),
            "median": float(np.quantile(target, 0.5)),
            "p90": float(np.quantile(target, 0.9)),
            "p99": float(np.quantile(target, 0.99)),
            "max": float(np.max(target)),
        },
        "problem_length_vs_target": correlation(problem_length),
        "full_prompt_length_vs_target": correlation(prompt_length),
        "call_count_vs_target": correlation(call_count),
        "baseline_omitted_prompt_fields": ["requirements", "interface"],
    }


def run(
    config_path: Path,
    *,
    trace_override: Path | None = None,
    metadata_override: Path | None = None,
    output_override: Path | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    source = config["source"]
    task_metadata = config["task_metadata"]
    protocol = config["protocol"]
    refinement = config["task_initialization_refinement"]
    trace_path = (trace_override or root / source["local_path"]).resolve()
    metadata_path = (metadata_override or root / task_metadata["local_path"]).resolve()
    for label, path, expected_sha256 in (
        ("trajectory", trace_path, source["sha256"]),
        ("task metadata", metadata_path, task_metadata["sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} parquet is missing at {path}")
        if _sha256(path).lower() != str(expected_sha256).lower():
            raise ValueError(f"{label} parquet SHA-256 does not match the frozen config")

    trace_ids = _read_task_ids(trace_path)
    metadata_ids = _read_task_ids(metadata_path)
    if set(trace_ids) != set(metadata_ids):
        raise ValueError("trajectory and Task-start metadata instance_id sets differ")
    partition = _partition_task_ids(
        trace_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    pilot_ids = frozenset(partition.pilot_tasks)
    pilot_folds = _assign_folds(
        partition.pilot_tasks,
        folds=int(protocol["folds"]),
        seed=int(protocol["split_seed"]),
    )
    pilot = _build_examples(
        _load_selected_tasks(trace_path, pilot_ids),
        _load_metadata(metadata_path, pilot_ids),
        pilot_folds,
    )
    selection, selection_reports, _selection_records = _run_selection_screen(
        pilot,
        folds=int(protocol["folds"]),
        alpha=float(protocol["alpha"]),
        refinement=refinement,
        protocol=protocol,
    )

    reserve_ids = frozenset(partition.development_reserve_tasks)
    reserve_folds = _assign_folds(
        partition.development_reserve_tasks,
        folds=int(protocol["folds"]),
        seed=int(refinement["confirmation_seed"]),
    )
    reserve = _build_examples(
        _load_selected_tasks(trace_path, reserve_ids),
        _load_metadata(metadata_path, reserve_ids),
        reserve_folds,
    )
    confirmation, records, confirmation_report = _run_confirmation(
        pilot,
        reserve,
        folds=int(protocol["folds"]),
        alpha=float(protocol["alpha"]),
        refinement=refinement,
        protocol=protocol,
    )
    development = tuple(sorted((*pilot, *reserve), key=lambda example: example.task_id))
    development_folds = _assign_folds(
        [example.task_id for example in development],
        folds=int(protocol["folds"]),
        seed=int(protocol["split_seed"]),
    )
    development = tuple(
        TaskStartExample(
            task_id=example.task_id,
            metadata=example.metadata,
            total_output_tokens=example.total_output_tokens,
            call_count=example.call_count,
            mean_output_tokens=example.mean_output_tokens,
            fold=development_folds[example.task_id],
            model=example.model,
        )
        for example in development
    )
    residual_id = str(refinement["residual_candidate"])
    hybrid_id = str(refinement["hybrid_candidate"])
    residual_selection, residual_reports, residual_records = _run_selection_screen(
        development,
        folds=int(protocol["folds"]),
        alpha=float(protocol["alpha"]),
        refinement=refinement,
        protocol=protocol,
        candidate_id=hybrid_id,
        learned_ids=(
            str(refinement["length_component"]),
            str(refinement["text_component"]),
            str(refinement["structure_component"]),
            str(refinement["candidate"]),
            residual_id,
            *_additional_reference_ids(refinement),
        ),
    )

    result = {
        "config_id": config["config_id"],
        "experiment_id": str(
            refinement.get(
                "experiment_id",
                "public-task-initialization-refinement-v1",
            )
        ),
        "source": {
            **source,
            "local_path": str(trace_path),
            "actual_sha256": _sha256(trace_path),
            "source_rows": len(trace_ids),
        },
        "task_metadata": {
            **task_metadata,
            "local_path": str(metadata_path),
            "actual_sha256": _sha256(metadata_path),
            "source_rows": len(metadata_ids),
            "instance_id_join_complete": True,
        },
        "protocol": {
            **protocol,
            "selection_task_count": len(pilot),
            "confirmation_task_count": len(reserve),
            "confirmation_seed": refinement["confirmation_seed"],
            "confirmation_calibration_fold": refinement["confirmation_calibration_fold"],
            "candidate_frozen_before_confirmation_payload_read": True,
            "residual_selection_task_count": len(development),
            "residual_stage": "post-confirmation development selection",
            "additional_reference_ids": list(_additional_reference_ids(refinement)),
            "final_holdout_task_count": len(partition.final_holdout_tasks),
            "final_holdout_evaluated": False,
            "final_holdout_turn_payloads_selected": False,
            "final_holdout_metadata_payloads_selected": False,
        },
        "method": {
            **refinement,
            "fusion": "equal-weight geometric mean on component scores",
            "point_mapping": "pilot out-of-fold MAE scale",
            "raw_interval": "pilot out-of-fold log residual quantiles",
            "interval_calibrator": "task_max_log_conformal",
            "target": TARGET.value,
            "target_definition": "sum of all Task Call osl at Task launch",
            "target_unit": "provider output Tokens",
            "excluded_metadata_fields": [
                "patch",
                "test_patch",
                "fail_to_pass",
                "pass_to_pass",
                "before_repo_set_cmd",
                "selected_test_files_to_run",
            ],
        },
        "diagnostics": _diagnostics(pilot),
        "selection": selection,
        "selection_fold_reports": selection_reports,
        "confirmation": confirmation,
        "confirmation_fit_report": confirmation_report,
        "residual_selection": residual_selection,
        "residual_selection_fold_reports": residual_reports,
        "dependencies": {
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    output_dir = (output_override or root / config["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_start_refinement_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_records = [record for values in records.values() for record in values]
    with (output_dir / "task_start_refinement_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in sorted(all_records, key=lambda item: (item.candidate_id, item.point_id)):
            document = _prediction_document(record)
            document["stage"] = "development_confirmation"
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    residual_output = output_dir / "task_start_residual_predictions.jsonl"
    with residual_output.open("w", encoding="utf-8") as handle:
        residual_rows = [record for values in residual_records.values() for record in values]
        for record in sorted(
            residual_rows,
            key=lambda item: (item.candidate_id, item.point_id),
        ):
            document = _prediction_document(record)
            document["stage"] = "post_confirmation_development_selection"
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    args = _parser().parse_args()
    result = run(
        args.config,
        trace_override=args.trace_data,
        metadata_override=args.task_metadata,
        output_override=args.output_dir,
    )
    print(
        json.dumps(
            {
                "selection": result["selection"]["metrics"],
                "confirmation": result["confirmation"]["metrics"],
                "comparisons": result["confirmation"]["comparisons"],
                "residual_selection": result["residual_selection"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
