from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA, V2, V2_SHA,
    development_data, digest, equal_task_weights,
)


OUT = BASE / "development_additive_input_probe.json"
FEATURE_NAMES = (
    "previous_output", "tool_feedback_payload_bytes", "tool_messages_received",
    "previous_declared_tool_calls", "prior_input_change", "previous_cache_fraction",
)


def row(point: dict) -> list[float]:
    named = dict(zip(FEATURES, point["features"], strict=True))
    return [named[name] / 4 if name == "tool_feedback_payload_bytes" else named[name]
            for name in FEATURE_NAMES]


def replay(points: list[dict], tasks: dict[str, dict], cap: int, predictions: dict) -> dict:
    complete_under = sorted(task_id for task_id, task in tasks.items()
                            if task["terminal_class"] == "model_stop"
                            and task["nominal_total_tokens"] <= cap)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    maxima = []
    for task_id in complete_under:
        scores = [(point["spent"] + predictions[(task_id, point["call_index"])] - cap) / cap
                  for point in by_id[task_id] if point["spent"] < cap]
        maxima.append(max(scores, default=float("-inf")))
    rank = ceil((len(maxima) + 1) * .95)
    q = max(0.0, sorted(maxima)[rank - 1]) if rank <= len(maxima) else float("inf")
    details = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        observed = task["nominal_total_tokens"]
        ref = min(observed, cap)
        stop = next((point for point in by_id[task_id]
                     if point["spent"] < cap and
                     (point["spent"] + predictions[(task_id, point["call_index"])] - cap) / cap > q), None)
        saved = ref - (stop["spent"] if stop else ref)
        details.append({"task_id": task_id, "saved_tokens": saved,
                        "alert": stop is not None,
                        "false_abort_under_cap": bool(stop and task["terminal_class"] == "model_stop"
                                                      and observed <= cap),
                        "terminal_class": task["terminal_class"]})
    ref_total = sum(min(task["nominal_total_tokens"], cap) for task in tasks.values())
    return {"q": q, "calibration_undercap_complete": len(maxima),
            "reference_cost_tokens": ref_total,
            "saved_tokens": sum(item["saved_tokens"] for item in details),
            "pooled_fractional_saving": sum(item["saved_tokens"] for item in details) / ref_total,
            "alerts": sum(item["alert"] for item in details),
            "false_aborts_under_cap": sum(item["false_abort_under_cap"] for item in details),
            "details": details}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Development input changed: {path}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    points, tasks = development_data(lock, v2)
    fit = points["fit"]
    cal = points["calibration"]
    x_fit = np.asarray([row(point) for point in fit], dtype=float)
    y_fit = np.asarray([point["next_input_tokens"] - point["previous_input"]
                        for point in fit], dtype=float)
    scaler = StandardScaler().fit(x_fit)
    model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=300)
    model.fit(scaler.transform(x_fit), y_fit, sample_weight=equal_task_weights(fit))
    x_cal = np.asarray([row(point) for point in cal], dtype=float)
    raw = model.predict(scaler.transform(x_cal))
    estimate = np.maximum(0.0, np.asarray([point["previous_input"] for point in cal]) + raw)
    predictions = {(point["task_id"], point["call_index"]): float(estimate[index])
                   for index, point in enumerate(cal)}
    if len(predictions) != len(cal) or not np.all(np.isfinite(estimate)):
        raise ValueError("Prediction coverage or finiteness failed")
    report = {
        "status": "development_only_exploratory_additive_input_probe",
        "lock_sha256": LOCK_SHA, "development_resolution_sha256": V2_SHA,
        "script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False, "run_level_results_access": False,
        "fit_checkpoints": len(fit), "calibration_checkpoints": len(cal),
        "feature_names": FEATURE_NAMES,
        "mechanism": "I_next_hat=max(0,I_previous+Huber(previous settled output, pre-call tool feedback bytes/4 and counts, prior trend and cache fraction))",
        "coefficient": dict(zip(FEATURE_NAMES, [float(value) for value in model.coef_], strict=True)),
        "intercept": float(model.intercept_),
        "fit_delta_mae": float(np.mean(np.abs(model.predict(scaler.transform(x_fit)) - y_fit))),
        "cal_next_input_mae": float(np.mean(np.abs(estimate - np.asarray([p["next_input_tokens"] for p in cal])))),
        "cal_previous_input_mae": float(np.mean(np.abs(np.asarray([p["previous_input"]-p["next_input_tokens"] for p in cal])))),
        "B50_95pct_undercap_case_max": replay(
            cal, {task_id: tasks[task_id] for task_id in lock["calibration_ids"]},
            caps["0.5"], predictions),
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output_sha256": sha256(OUT.read_bytes()).hexdigest(),
                      "fit": len(fit), "cal": len(cal),
                      "cal_next_input_mae": report["cal_next_input_mae"],
                      "cal_previous_input_mae": report["cal_previous_input_mae"],
                      "B50": {key: value for key, value in report["B50_95pct_undercap_case_max"].items()
                              if key != "details"}}, indent=2))


if __name__ == "__main__":
    main()
