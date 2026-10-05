from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion

from scripts.experiments.run_public_task_initialization_refinement import (
    _load_metadata,
    _prompt_document,
)
from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.run_public_trajectory_validation import (
    _assign_folds,
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


ACTORS = ("opus", "gpt_oss")
SHARED_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_ALPHA = 3.0
INNER_SCALE_SEED = 20260903


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/public_trajectory_ood_transfer_v1.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/experiments/public_two_actor_task_start_dev_20260923"),
    )
    return parser


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2.0)]])


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
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


def _raw_predict(
    train_ids: list[str],
    query_ids: list[str],
    prompts: dict[str, str],
    truth: dict[str, dict[str, float]],
) -> dict[str, dict[str, np.ndarray]]:

    train_docs = [prompts[task_id] for task_id in train_ids]
    query_docs = [prompts[task_id] for task_id in query_ids]
    train_length = np.log1p([len(doc) for doc in train_docs])
    query_length = np.log1p([len(doc) for doc in query_docs])
    length_train_x = np.column_stack((np.ones(len(train_ids)), train_length))
    length_query_x = np.column_stack((np.ones(len(query_ids)), query_length))
    vectorizer = _vectorizer()
    train_text_x = vectorizer.fit_transform(train_docs)
    query_text_x = vectorizer.transform(query_docs)
    residual_train: dict[str, np.ndarray] = {}
    length_log_query: dict[str, np.ndarray] = {}
    result: dict[str, dict[str, np.ndarray]] = {}
    for actor in ACTORS:
        y_log = np.log1p([truth[actor][task_id] for task_id in train_ids])
        beta, *_ = np.linalg.lstsq(length_train_x, y_log, rcond=None)
        length_log_query[actor] = length_query_x @ beta
        residual_train[actor] = y_log - length_train_x @ beta
        result[actor] = {
            "actor_empirical_median": np.repeat(
                np.median([truth[actor][task_id] for task_id in train_ids]),
                len(query_ids),
            ),
            "actor_full_prompt_length": np.maximum(
                1.0, np.expm1(length_log_query[actor])
            ),
        }

    pooled_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    pooled_model.fit(
        vstack((train_text_x, train_text_x), format="csr"),
        np.concatenate([residual_train[actor] for actor in ACTORS]),
    )
    shared_query = pooled_model.predict(query_text_x)
    for actor in ACTORS:
        actor_model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
        actor_model.fit(train_text_x, residual_train[actor])
        actor_query = actor_model.predict(query_text_x)
        for weight in SHARED_WEIGHTS:
            name = f"shared_weight_{weight:.2f}"
            result[actor][name] = np.maximum(
                1.0,
                np.expm1(
                    length_log_query[actor]
                    + (1.0 - weight) * actor_query
                    + weight * shared_query
                ),
            )
    return result


def _load_development(
    config_path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, object]]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = project_root_for(config_path)
    opus_path = root / config["source"]["local_path"]
    metadata_path = root / config["task_metadata"]["local_path"]
    if _sha256(opus_path) != config["source"]["sha256"]:
        raise ValueError("Opus source hash mismatch")
    if _sha256(metadata_path) != config["task_metadata"]["sha256"]:
        raise ValueError("prompt metadata hash mismatch")
    opus_ids = _read_task_ids(opus_path)
    if set(opus_ids) != set(_read_task_ids(metadata_path)):
        raise ValueError("Opus and prompt metadata ID sets differ")
    protocol = config["protocol"]
    partition = _partition_task_ids(
        opus_ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    dev_ids = tuple(sorted((*partition.pilot_tasks, *partition.development_reserve_tasks)))
    dev_set = frozenset(dev_ids)
    holdout_set = set(partition.final_holdout_tasks)
    if len(dev_ids) != 585 or len(holdout_set) != 146 or dev_set & holdout_set:
        raise ValueError("permanent development/holdout partition changed")
    ood_source = config["ood_source"]
    ood_ids, ood_tasks = _load_ood_tasks(
        root / ood_source["local_directory"],
        ood_source["files"],
        ood_source["sha256"],
        dev_set,
    )
    if set(ood_ids) != set(opus_ids):
        raise ValueError("OOD and Opus source ID sets differ")
    opus_tasks = _load_selected_tasks(opus_path, dev_set)
    metadata = _load_metadata(metadata_path, dev_set)
    prompts = {task_id: _prompt_document(metadata[task_id]) for task_id in dev_ids}
    truth = {
        "opus": {task.task_id: float(sum(turn.output_tokens for turn in task.turns)) for task in opus_tasks},
        "gpt_oss": {task.task_id: float(sum(turn.output_tokens for turn in task.turns)) for task in ood_tasks},
    }
    if any(set(truth[actor]) != dev_set for actor in ACTORS):
        raise ValueError("development labels incomplete")
    receipt = {
        "source_sha256": config["source"]["sha256"],
        "metadata_sha256": config["task_metadata"]["sha256"],
        "ood_sha256": ood_source["sha256"],
        "split_seed": int(protocol["split_seed"]),
        "development_task_count": len(dev_ids),
        "final_holdout_task_count": len(holdout_set),
        "final_holdout_id_sha256": hashlib.sha256(
            "\n".join(partition.final_holdout_tasks).encode("utf-8")
        ).hexdigest(),
        "final_holdout_payloads_read": False,
    }
    return prompts, truth, receipt


def _bootstrap_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 20260902,
    draws: int = 5000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(candidate)
    differences = candidate - reference
    samples = np.empty(draws)
    for index in range(draws):
        selected = rng.integers(0, n, size=n)
        samples[index] = float(np.mean(differences[selected]))
    return {
        "delta": float(np.mean(differences)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    prompts, truth, receipt = _load_development(config_path)
    dev_ids = sorted(prompts)
    folds = _assign_folds(dev_ids, folds=5, seed=receipt["split_seed"])
    predictions: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    fold_reports = []
    for fold in range(5):
        train_ids = [task_id for task_id in dev_ids if folds[task_id] != fold]
        test_ids = [task_id for task_id in dev_ids if folds[task_id] == fold]
        inner = _assign_folds(train_ids, folds=5, seed=INNER_SCALE_SEED + fold)
        scale_fit_ids = [task_id for task_id in train_ids if inner[task_id] != 0]
        scale_val_ids = [task_id for task_id in train_ids if inner[task_id] == 0]
        validation_raw = _raw_predict(scale_fit_ids, scale_val_ids, prompts, truth)
        test_raw = _raw_predict(train_ids, test_ids, prompts, truth)
        for actor in ACTORS:
            validation_y = np.asarray([truth[actor][task_id] for task_id in scale_val_ids])
            for arm, raw_test in test_raw[actor].items():
                if arm == "actor_empirical_median":
                    scale = 1.0
                else:
                    raw_validation = validation_raw[actor][arm]
                    scale = _weighted_median(
                        validation_y / raw_validation, raw_validation
                    )
                for task_id, value in zip(test_ids, raw_test):
                    predictions[actor][arm][task_id] = max(1.0, float(scale * value))
        fold_reports.append(
            {
                "fold": fold,
                "fit_families": len(train_ids),
                "internal_scale_fit_families": len(scale_fit_ids),
                "internal_scale_validation_families": len(scale_val_ids),
                "test_families": len(test_ids),
            }
        )

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for actor in ACTORS:
        actual = np.asarray([truth[actor][task_id] for task_id in dev_ids])
        metrics[actor] = {}
        for arm, by_id in predictions[actor].items():
            if set(by_id) != set(dev_ids):
                raise ValueError(f"incomplete OOF predictions: {actor}/{arm}")
            forecast = np.asarray([by_id[task_id] for task_id in dev_ids])
            absolute = np.abs(forecast - actual)
            metrics[actor][arm] = {
                "mae": float(np.mean(absolute)),
                "wape": float(np.sum(absolute) / np.sum(actual)),
                "mean_target": float(np.mean(actual)),
            }
    arms = sorted(metrics[ACTORS[0]])
    macro_wape = {arm: float(np.mean([metrics[actor][arm]["wape"] for actor in ACTORS])) for arm in arms}
    blend_arms = [f"shared_weight_{weight:.2f}" for weight in SHARED_WEIGHTS]
    selected = min(blend_arms, key=lambda arm: (macro_wape[arm], float(arm.rsplit("_", 1)[-1])))
    paired = {}
    for actor in ACTORS:
        actual = np.asarray([truth[actor][task_id] for task_id in dev_ids])
        chosen_error = np.abs(
            np.asarray([predictions[actor][selected][task_id] for task_id in dev_ids]) - actual
        )
        paired[actor] = {}
        for reference in (
            "actor_empirical_median",
            "actor_full_prompt_length",
            "shared_weight_0.00",
        ):
            reference_error = np.abs(
                np.asarray([predictions[actor][reference][task_id] for task_id in dev_ids]) - actual
            )
            paired[actor][reference] = _bootstrap_delta(chosen_error, reference_error)
    report = {
        "status": "development-only two-actor Task Start output-token screen",
        "receipt": receipt,
        "method": {
            "target": "sum of all task turns' provider-reported osl output tokens",
            "prompt_fields": ["problem_statement", "requirements", "interface"],
            "ridge_alpha": RIDGE_ALPHA,
            "shared_weights": list(SHARED_WEIGHTS),
            "selection_metric": "macro WAPE across the two actors",
            "selected_arm": selected,
            "development_selection_only": True,
        },
        "folds": fold_reports,
        "metrics": metrics,
        "macro_wape": macro_wape,
        "paired_bootstrap": paired,
        "final_holdout_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "report.json").write_text(report_text, encoding="utf-8")
    (output_dir / "development_report.json").write_text(report_text, encoding="utf-8")
    with (output_dir / "development_oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for actor in ACTORS:
            for task_id in dev_ids:
                for arm in arms:
                    handle.write(
                        json.dumps(
                            {
                                "actor": actor,
                                "task_id": task_id,
                                "arm": arm,
                                "prediction": predictions[actor][arm][task_id],
                                "target": truth[actor][task_id],
                                "fold": folds[task_id],
                                "stage": "development_oof",
                            }
                        )
                        + "\n"
                    )
    return report


def main() -> int:
    args = _parser().parse_args()
    report = run(args.config, args.output_dir)
    print(json.dumps({"selected": report["method"]["selected_arm"], "macro_wape": report["macro_wape"], "actor_mae": {actor: {arm: value["mae"] for arm, value in arms.items()} for actor, arms in report["metrics"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
