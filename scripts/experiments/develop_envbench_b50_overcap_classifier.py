from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest, equal_task_weights,
)


OUT = ROOT / "workspace/experiments/envbench_b50_overcap_dev_20260923"
DIRECT_REPORT = BASE / "development_direct_ensemble_extension.json"
DIRECT_REPORT_SHA = "53ae61d80e7326b7a95e380b1a45df5ec869675b513f5a4e4375fdb04203bc35"
FIRST_REPORT = BASE / "development_budget_model_report.json"
FIRST_REPORT_SHA = "8f836ac410f628efd19299b8622698d45093eee8a46f59940426508fea6ddc29"
SEEDS = (20260923, 20270923)
ARM_NAMES = ("overcap_classifier_all", "overcap_classifier_complete_only",
             "direct_remainder_two_seed", "direct_next_input", "previous_input")


def train(seed: int, fit: list[dict], labels: np.ndarray) -> tuple[lgb.Booster, float]:
    matrix = np.asarray([point["features"] for point in fit], dtype=np.float64)
    data = lgb.Dataset(matrix, label=labels, weight=equal_task_weights(fit),
                       feature_name=list(FEATURES))
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 40,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1, "num_threads": 1, "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed,
    }
    started = time.perf_counter()
    booster = lgb.train(params, data, num_boost_round=180)
    return booster, time.perf_counter() - started


def load_existing_booster(path: Path) -> lgb.Booster:


    raw = path.read_bytes()
    return lgb.Booster(model_str=raw.replace(b"\r\n", b"\n").decode("utf-8"))


def threshold(cal: list[dict], tasks: dict[str, dict], arm: str,
              cap: int) -> dict:
    eligible = sorted(task_id for task_id, task in tasks.items()
                      if task["group"] == "calibration" and
                      task["terminal_class"] == "model_stop" and
                      task["nominal_total_tokens"] <= cap)
    by_id = defaultdict(list)
    for point in cal:
        if point["task_id"] in eligible and point["spent"] < cap:
            by_id[point["task_id"]].append(point)
    maxima = [max((point["score"][arm] for point in by_id[task_id]), default=-float("inf"))
              for task_id in eligible]
    rank = ceil((len(maxima) + 1) * 0.95)
    raw = sorted(maxima)[rank - 1] if rank <= len(maxima) else float("inf")
    final = max(0.0, raw)
    return {
        "calibration_reference_completers": len(eligible),
        "with_call_start": sum(bool(by_id[task_id]) for task_id in eligible),
        "rank": rank, "q_raw": raw, "q_final": final,
        "family_max_scores": dict(zip(eligible, maxima, strict=True)),
    }


def replay(cal: list[dict], tasks: dict[str, dict], arm: str, cap: int,
           q: float) -> list[dict]:
    by_id = defaultdict(list)
    for point in cal:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(task_id for task_id, task in tasks.items()
                          if task["group"] == "calibration"):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        terminal = task["terminal_class"]
        if terminal == "model_stop":
            stratum = "complete_undercap" if total <= cap else "complete_overcap"
        elif terminal == "step_limit_censored" and total >= cap:
            stratum = "censored_observed_overcap"
        else:
            stratum = "censored_below_cap_unknown"
        candidates = (point for point in by_id[task_id] if point["spent"] < cap)
        stop = next((point for point in candidates if point["score"][arm] > q), None)
        identifiable = terminal == "model_stop" or total >= cap
        reference = min(total, cap) if identifiable else None
        policy = stop["spent"] if stop is not None else reference
        saved = reference - policy if reference is not None and policy is not None else None
        if saved is not None and saved < 0:
            raise ValueError("Negative saving")
        rows.append({
            "task_id": task_id, "stratum": stratum, "observed_total": total,
            "reference_tokens": reference, "policy_tokens": policy,
            "saved_tokens": saved, "alert": stop is not None,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "false_abort": bool(stop is not None and stratum == "complete_undercap"),
            "reference_capture": int(stratum == "complete_undercap"),
            "policy_capture": int(stratum == "complete_undercap" and stop is None),
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    def summarise(part: list[dict]) -> dict:
        scored = [row for row in part if row["saved_tokens"] is not None]
        reference = sum(row["reference_tokens"] for row in scored)
        saved = sum(row["saved_tokens"] for row in scored)
        contributions = sorted((row["saved_tokens"] for row in scored), reverse=True)
        return {
            "tasks": len(part), "cost_identifiable": len(scored),
            "reference_tokens": reference, "saved_tokens": saved,
            "pooled_saving_ratio": saved / reference if reference else None,
            "alerts": sum(row["alert"] for row in part),
            "affected_tasks": sum(row["saved_tokens"] is not None and row["saved_tokens"] > 0
                                  for row in part),
            "false_aborts": sum(row["false_abort"] for row in part),
            "reference_capture": sum(row["reference_capture"] for row in part),
            "policy_capture": sum(row["policy_capture"] for row in part),
            "top_1_saved_share": contributions[0] / saved if saved else None,
            "top_5_saved_share": sum(contributions[:5]) / saved if saved else None,
        }
    result = {"all": summarise(rows)}
    for stratum in ("complete_undercap", "complete_overcap",
                    "censored_observed_overcap", "censored_below_cap_unknown"):
        result[stratum] = summarise([row for row in rows if row["stratum"] == stratum])
    return result


def json_finite(value):

    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValueError("NaN in report")
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
                           (FIRST_REPORT, FIRST_REPORT_SHA),
                           (DIRECT_REPORT, DIRECT_REPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development dependency changed: {path}")
    if OUT.exists():
        raise FileExistsError(f"Development output already exists: {OUT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078 or len(FEATURES) != 21:
        raise ValueError("B50 or causal feature contract drift")
    points, tasks = development_data(lock, v2)
    fit = [point for point in points["fit"] if point["spent"] < cap]
    cal = [point for point in points["calibration"] if point["spent"] < cap]
    fit_ids = set(lock["fit_ids"])
    cal_ids = set(lock["calibration_ids"])
    if fit_ids & cal_ids or set(tasks) != fit_ids | cal_ids or len(fit_ids) != 400 or len(cal_ids) != 132:
        raise ValueError("Fit/calibration task identity drift")
    fit_classes = {task_id: task["terminal_class"] for task_id, task in tasks.items()
                   if task_id in fit_ids}
    if list(fit_classes.values()).count("model_stop") != 370 or list(fit_classes.values()).count(
            "step_limit_censored") != 30:
        raise ValueError("Fit censor strata drift")
    for task_id in fit_ids:
        if fit_classes[task_id] == "step_limit_censored" and tasks[task_id][
                "nominal_total_tokens"] < cap:
            raise ValueError("Unidentifiable censored B50 class")
    def class_label(point: dict) -> int:
        task = tasks[point["task_id"]]
        if task["terminal_class"] == "model_stop":
            return int(task["nominal_total_tokens"] > cap)
        return int(task["nominal_total_tokens"] >= cap)
    subsets = {
        "overcap_classifier_all": fit,
        "overcap_classifier_complete_only": [point for point in fit if point["terminal_class"] == "model_stop"],
    }
    trained = {}
    train_seconds = {}
    for arm, rows in subsets.items():
        labels = np.asarray([class_label(point) for point in rows], dtype=np.float64)
        if set(labels) != {0.0, 1.0}:
            raise ValueError("Classifier target lacks a class")
        trained[arm] = []
        train_seconds[arm] = []
        for seed in SEEDS:
            model, seconds = train(seed, rows, labels)
            trained[arm].append(model)
            train_seconds[arm].append(seconds)
    direct_receipt = json.loads(DIRECT_REPORT.read_text(encoding="utf-8"))
    direct_models = []
    for seed in SEEDS:
        item = direct_receipt["models"][str(seed)]
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError("Direct model hash drift")
        direct_models.append(load_existing_booster(path))
    first_receipt = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    next_item = first_receipt["models"]["next_input"]
    next_path = ROOT / next_item["path"]
    if digest(next_path) != next_item["sha256"]:
        raise ValueError("Next-input model hash drift")
    next_model = load_existing_booster(next_path)
    matrix = np.asarray([point["features"] for point in cal], dtype=np.float64)
    if matrix.shape[1] != 21 or not np.all(np.isfinite(matrix)):
        raise ValueError("Calibration feature shape/value drift")
    for arm, models in trained.items():
        predictions = np.mean([model.predict(matrix, num_threads=1) for model in models], axis=0)
        for point, value in zip(cal, predictions, strict=True):
            point.setdefault("score", {})[arm] = float(value)
    remainder = np.mean([np.maximum(0.0, np.expm1(model.predict(matrix, num_threads=1)))
                         for model in direct_models], axis=0)
    next_input = np.maximum(0.0, np.expm1(next_model.predict(matrix, num_threads=1)))
    for index, point in enumerate(cal):
        for arm, predicted in (("direct_remainder_two_seed", remainder[index]),
                               ("direct_next_input", next_input[index]),
                               ("previous_input", point["previous_input"])):
            point["score"][arm] = float((point["spent"] + predicted - cap) / cap)
        if any(not np.isfinite(value) for value in point["score"].values()):
            raise ValueError("Nonfinite calibration score")
    settings = {arm: threshold(cal, tasks, arm, cap) for arm in ARM_NAMES}
    decisions = {arm: replay(cal, tasks, arm, cap, settings[arm]["q_final"]) for arm in ARM_NAMES}
    summaries = {arm: aggregate(decisions[arm]) for arm in ARM_NAMES}
    OUT.mkdir(parents=True)
    models_receipt = {}
    for arm, models in trained.items():
        models_receipt[arm] = {}
        for seed, model, seconds in zip(SEEDS, models, train_seconds[arm], strict=True):
            path = OUT / f"{arm}_seed_{seed}.txt"
            with path.open("xb") as stream:
                stream.write(model.model_to_string().encode("utf-8"))
            models_receipt[arm][str(seed)] = {
                "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                "train_seconds": seconds,
            }
    per_task = OUT / "calibration_per_task.jsonl"
    with per_task.open("x", encoding="utf-8") as stream:
        for task_id in sorted(cal_ids):
            stream.write(json.dumps({"task_id": task_id,
                                     "arms": {arm: next(row for row in decisions[arm]
                                                        if row["task_id"] == task_id)
                                              for arm in ARM_NAMES}},
                                    separators=(",", ":")) + "\n")
    report = {
        "status": "fit_calibration_development_only",
        "source_revision": lock["revision"],
        "identity_lock_sha256": LOCK_SHA, "development_resolution_sha256": V2_SHA,
        "cap_analysis_sha256": CAPS_SHA, "first_model_report_sha256": FIRST_REPORT_SHA,
        "direct_ensemble_report_sha256": DIRECT_REPORT_SHA,
        "existing_model_load": "verify original SHA; normalize CRLF to LF in memory only",
        "script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap, "feature_names": FEATURES,
        "label_contract": {
            "complete": "1 iff observed full T>B50; 0 iff T<=B50",
            "censored": "1 iff observed T_obs>=B50; known cap exhaustion, natural terminal unknown",
            "fit_censored_positive_tasks": 30,
            "fit_censored_positive_tasks_with_active_call_start": len({point["task_id"]
                for point in fit if point["terminal_class"] == "step_limit_censored"}),
            "fit_complete_tasks": 370,
            "training_points_with_spent_below_cap": {arm: len(rows) for arm, rows in subsets.items()},
            "calibration_points_with_spent_below_cap": len(cal),
        },
        "model_settings": {
            "binary_objective": "binary_logloss", "seeds": SEEDS,
            "num_boost_round_per_seed": 180, "learning_rate": 0.05,
            "num_leaves": 15, "min_data_in_leaf": 40,
            "feature_fraction": 0.9, "bagging_fraction": 0.9,
            "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 1,
            "task_weight": "Each fit task's active prefix weights sum to one",
        },
        "calibration_rule": {
            "risk_set": "complete model_stop tasks with T<=B50",
            "family_score": "max score over Call Start prefixes with settled C<B50; empty=-inf",
            "rank": "ceil((n+1)*.95), ascending; if rank>n q=+inf",
            "threshold": "max(0,q_raw), stop on strict score>threshold and C<B50",
            "evaluation_note": "Calibration replay uses the same tasks that set q; development only",
        },
        "models": models_receipt, "thresholds": settings, "summaries": summaries,
        "per_task_path": str(per_task.relative_to(ROOT)),
        "per_task_sha256": digest(per_task),
    }
    path = OUT / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_finite(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(json_finite({"report": str(path), "report_sha256": digest(path),
                      "B50": {arm: {"q": settings[arm]["q_final"],
                                      "summary": summaries[arm]["all"]} for arm in ARM_NAMES}}), indent=2))


if __name__ == "__main__":
    main()
