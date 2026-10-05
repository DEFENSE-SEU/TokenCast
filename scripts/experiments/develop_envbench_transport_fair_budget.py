from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import tiktoken

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest, equal_task_weights,
)
from .audit_envbench_transport_input_development import extract


TRANSPORT_REPORT = ROOT / "workspace/experiments/envbench_transport_input_dev_20260923/report.json"
TRANSPORT_SHA = "3801bc60101dbf0a30964c638f11f00a357e3dc9a51f07c5edc7eef09bfa2c2c"
OUT = ROOT / "workspace/experiments/envbench_transport_fair_budget_dev_20260923"
SEEDS = (20260923, 20270923)
FEATURE22 = FEATURES + ("transported_next_input",)
LEARNED = ("classifier_all", "classifier_complete_only", "remainder",
           "input_delta_previous", "input_delta_transport", "input_raw")
SINGLE_ARMS = ("classifier_all", "classifier_complete_only",
               "remainder_two_seed", "remainder_single_seed",
               "input_delta_previous", "input_delta_transport", "input_raw",
               "transport_guard", "previous_input")
LONG_ARMS = ("classifier_all", "classifier_complete_only",
             "remainder_two_seed", "remainder_single_seed")
GUARDS = ("transport_guard", "input_delta_previous", "input_delta_transport")


def prepare(lock: dict, v2: dict, fit_offset: int, cap: int):
    grouped, tasks = development_data(lock, v2)
    encoder = tiktoken.get_encoding("o200k_base")
    observed = {}
    for group, ids in (("fit", set(lock["fit_ids"])),
                       ("calibration", set(lock["calibration_ids"]))):
        for row in extract(group, ids, tasks, encoder):
            key = (row["task_id"], row["call_index"])
            if key in observed:
                raise ValueError("Duplicate transport point")
            observed[key] = row
    output = {}
    tool_index = FEATURES.index("tool_messages_received")
    for group in ("fit", "calibration"):
        output[group] = []
        for point in grouped[group]:
            row = observed[(point["task_id"], point["call_index"])]
            if (point["previous_input"] != row["previous_input"] or
                    point["next_input_tokens"] != row["observed_next_input"] or
                    point["spent"] != row["spent"] or
                    point["features"][tool_index] != row["tool_messages"]):
                raise ValueError("Transport and original causal feature mismatch")
            transported = float(row["transport_base"] + fit_offset)
            if transported < 0:
                raise ValueError("Negative transported input")
            new_point = dict(point)
            new_point["features"] = list(point["features"]) + [transported]
            new_point["transported_next_input"] = transported
            new_point["tool_messages"] = row["tool_messages"]
            if point["spent"] < cap:
                output[group].append(new_point)
    if len(grouped["fit"]) != 4759 or len(grouped["calibration"]) != 1502:
        raise ValueError("Full development point count drift")
    if set(observed) != {(point["task_id"], point["call_index"])
                        for group in grouped.values() for point in group}:
        raise ValueError("Transport point coverage drift")
    return output, tasks


def labels(name: str, fit: list[dict], tasks: dict[str, dict], cap: int):
    subset = fit if name != "classifier_complete_only" else [
        point for point in fit if point["terminal_class"] == "model_stop"]
    if name == "remainder":
        subset = [point for point in fit if point["terminal_class"] == "model_stop"]
    if name.startswith("classifier"):
        y = [int(tasks[point["task_id"]]["nominal_total_tokens"] > cap)
             if point["terminal_class"] == "model_stop" else
             int(tasks[point["task_id"]]["nominal_total_tokens"] >= cap)
             for point in subset]
        objective = "binary"
    elif name == "remainder":
        y = np.log1p([point["future_observed_tokens"] for point in subset])
        objective = "regression_l1"
    elif name == "input_delta_previous":
        y = [point["next_input_tokens"] - point["previous_input"] for point in subset]
        objective = "regression_l1"
    elif name == "input_delta_transport":
        y = [point["next_input_tokens"] - point["transported_next_input"] for point in subset]
        objective = "regression_l1"
    elif name == "input_raw":
        y = np.log1p([point["next_input_tokens"] for point in subset])
        objective = "regression_l1"
    else:
        raise ValueError(name)
    return subset, np.asarray(y, dtype=np.float64), objective


def train(name: str, seed: int, subset: list[dict], y: np.ndarray,
          objective: str) -> tuple[lgb.Booster, float]:
    x = np.asarray([point["features"] for point in subset], dtype=float)
    data = lgb.Dataset(x, label=y, weight=equal_task_weights(subset),
                       feature_name=list(FEATURE22))
    params = {
        "objective": objective, "learning_rate": 0.05, "num_leaves": 15,
        "min_data_in_leaf": 40, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1, "num_threads": 1,
        "seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed,
    }
    started = time.perf_counter()
    model = lgb.train(params, data, num_boost_round=180)
    return model, time.perf_counter() - started


def predict(cal: list[dict], models: dict[str, list[lgb.Booster]]) -> None:
    matrix = np.asarray([point["features"] for point in cal], dtype=float)
    if matrix.shape != (1180, 22) or not np.all(np.isfinite(matrix)):
        raise ValueError("Calibration feature shape/value drift")
    raw = {name: [model.predict(matrix, num_threads=1) for model in heads]
           for name, heads in models.items()}
    outputs = {name: np.mean(predictions, axis=0) for name, predictions in raw.items()}
    remainder_two = np.mean([np.maximum(0.0, np.expm1(values))
                             for values in raw["remainder"]], axis=0)
    remainder_one = np.maximum(0.0, np.expm1(raw["remainder"][0]))
    input_raw = np.mean([np.maximum(0.0, np.expm1(values))
                         for values in raw["input_raw"]], axis=0)
    for index, point in enumerate(cal):
        transport = point["transported_next_input"]
        previous = point["previous_input"]
        scores = {
            "classifier_all": float(outputs["classifier_all"][index]),
            "classifier_complete_only": float(outputs["classifier_complete_only"][index]),
            "remainder_two_seed": float(remainder_two[index]),
            "remainder_single_seed": float(remainder_one[index]),
            "input_delta_previous": float(max(0.0, previous + outputs["input_delta_previous"][index])),
            "input_delta_transport": float(max(0.0, transport + outputs["input_delta_transport"][index])),
            "input_raw": float(input_raw[index]),
            "transport_guard": transport,
            "previous_input": previous,
        }
        if any(not np.isfinite(value) or value < 0 for value in scores.values()):
            raise ValueError("Invalid calibration prediction")
        point["prediction"] = scores


def score(point: dict, arm: str, cap: int) -> float:
    if arm.startswith("classifier"):
        return point["prediction"][arm]
    return (point["spent"] + point["prediction"][arm] - cap) / cap


def qmax(cal: list[dict], tasks: dict[str, dict], arm: str, cap: int) -> dict:
    ids = sorted(task_id for task_id, task in tasks.items()
                 if task["group"] == "calibration" and
                 task["terminal_class"] == "model_stop" and
                 task["nominal_total_tokens"] <= cap)
    by_id = defaultdict(list)
    for point in cal:
        by_id[point["task_id"]].append(point)
    maxima = [max((score(point, arm, cap) for point in by_id[task_id]),
                  default=-float("inf")) for task_id in ids]
    if len(ids) != 64 or not all(np.isfinite(value) for value in maxima):
        raise ValueError("Undercap risk set drift")
    return {"q_raw": max(maxima), "q_final": max(0.0, max(maxima)),
            "risk_tasks": len(ids), "argmax_task": ids[int(np.argmax(maxima))]}


def replay(cal: list[dict], tasks: dict[str, dict], cap: int,
           long: str, guard: str | None, q: dict[str, dict]) -> list[dict]:
    by_id = defaultdict(list)
    for point in cal:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(task_id for task_id, task in tasks.items()
                          if task["group"] == "calibration"):
        task = tasks[task_id]
        total = task["nominal_total_tokens"]
        stop, reason = None, None
        for point in by_id[task_id]:
            if point["spent"] >= cap:
                continue
            long_hit = score(point, long, cap) > q[long]["q_final"]
            guard_hit = guard is not None and score(point, guard, cap) > q[guard]["q_final"]
            if long_hit or guard_hit:
                stop = point
                reason = "both" if long_hit and guard_hit else (
                    "long_only" if long_hit else "guard_only")
                break
        reference = min(total, cap)
        policy = stop["spent"] if stop is not None else reference
        if policy > reference:
            raise ValueError("Negative saving")
        terminal = task["terminal_class"]
        stratum = ("complete_undercap" if total <= cap else "complete_overcap") if (
            terminal == "model_stop") else "censored_observed_overcap"
        rows.append({
            "task_id": task_id, "stratum": stratum,
            "reference_tokens": reference, "policy_tokens": policy,
            "saved_tokens": reference - policy,
            "alert": stop is not None, "first_stop_reason": reason,
            "stop_call_index": stop["call_index"] if stop is not None else None,
            "false_abort": int(stop is not None and stratum == "complete_undercap"),
            "reference_capture": int(stratum == "complete_undercap"),
            "policy_capture": int(stratum == "complete_undercap" and stop is None),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    ref = sum(row["reference_tokens"] for row in rows)
    saved = sum(row["saved_tokens"] for row in rows)
    return {
        "tasks": len(rows), "reference_tokens": ref, "saved_tokens": saved,
        "pooled_saving_ratio": saved / ref if ref else None,
        "alerts": sum(row["alert"] for row in rows),
        "false_aborts": sum(row["false_abort"] for row in rows),
        "reference_capture": sum(row["reference_capture"] for row in rows),
        "policy_capture": sum(row["policy_capture"] for row in rows),
        "top_5_saved_share": sum(sorted((row["saved_tokens"] for row in rows),
                                        reverse=True)[:5]) / saved if saved else None,
        "first_stop_reasons": {key: sum(row["first_stop_reason"] == key for row in rows)
                               for key in ("long_only", "guard_only", "both")},
    }


def paired(rows: dict[str, list[dict]], left: str, right: str) -> dict:
    a, b = rows[left], rows[right]
    if [row["task_id"] for row in a] != [row["task_id"] for row in b]:
        raise ValueError("Paired task order drift")
    ref = np.asarray([row["reference_tokens"] for row in a], dtype=float)
    delta = np.asarray([x["saved_tokens"] - y["saved_tokens"]
                        for x, y in zip(a, b, strict=True)], dtype=float)
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0, len(ref), size=(10000, len(ref)))
    draw = delta[sample].sum(axis=1) / ref[sample].sum(axis=1)
    return {"pooled_delta": float(sum(delta) / sum(ref)),
            "ci_two_sided_95": [float(np.quantile(draw, .025)), float(np.quantile(draw, .975))],
            "one_sided_lower_95": float(np.quantile(draw, .05))}


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA),
                           (CAPS_FILE, CAPS_SHA), (TRANSPORT_REPORT, TRANSPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    transport = json.loads(TRANSPORT_REPORT.read_text(encoding="utf-8"))
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    offset = transport["fit_only_offset"]
    if (cap != 47078 or offset != 9 or len(FEATURE22) != 22 or
            tiktoken.__version__ != transport["tokenizer"]["version"]):
        raise ValueError("B50/transport/feature contract drift")
    grouped, tasks = prepare(lock, v2, offset, cap)
    fit, cal = grouped["fit"], grouped["calibration"]
    models, train_receipt = {}, {}
    for name in LEARNED:
        subset, y, objective = labels(name, fit, tasks, cap)
        if name.startswith("classifier") and set(y) != {0.0, 1.0}:
            raise ValueError("Missing classifier class")
        models[name] = []
        train_receipt[name] = {"points": len(subset), "tasks": len({row["task_id"] for row in subset}),
                               "objective": objective, "target": name, "seeds": {}}
        for seed in SEEDS:
            model, seconds = train(name, seed, subset, y, objective)
            models[name].append(model)
            train_receipt[name]["seeds"][str(seed)] = {"train_seconds": seconds}
    predict(cal, models)
    thresholds = {arm: qmax(cal, tasks, arm, cap) for arm in SINGLE_ARMS}
    names = list(SINGLE_ARMS) + [f"{long}_or_{guard}"
                                 for guard in GUARDS for long in LONG_ARMS]
    rows = {arm: replay(cal, tasks, cap, arm, None, thresholds) for arm in SINGLE_ARMS}
    for guard in GUARDS:
        for long in LONG_ARMS:
            rows[f"{long}_or_{guard}"] = replay(cal, tasks, cap, long, guard, thresholds)
    summaries = {}
    for arm in names:
        summaries[arm] = {"all": summarise(rows[arm])}
        for stratum in ("complete_undercap", "complete_overcap", "censored_observed_overcap"):
            summaries[arm][stratum] = summarise([row for row in rows[arm]
                                                  if row["stratum"] == stratum])
        if summaries[arm]["complete_undercap"]["false_aborts"]:
            raise ValueError("qmax arm has calibration false abort")
    comparisons = {}
    for guard in GUARDS:
        reference = f"remainder_two_seed_or_{guard}"
        comparisons[guard] = {
            long: paired(rows, f"{long}_or_{guard}", reference)
            for long in ("classifier_all", "classifier_complete_only", "remainder_single_seed")
        }
    OUT.mkdir(parents=True)
    for name, heads in models.items():
        for seed, model in zip(SEEDS, heads, strict=True):
            path = OUT / f"{name}_seed_{seed}.txt"
            with path.open("xb") as stream:
                stream.write(model.model_to_string().encode("utf-8"))
            train_receipt[name]["seeds"][str(seed)].update({
                "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    report = {
        "status": "fit_calibration_only_transport_feature_fair_budget_development",
        "chronology": "Proposed after standalone classifier and additive-input calibration outcomes; no test selection",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in (
            (LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
            (TRANSPORT_REPORT, TRANSPORT_SHA))},
        "imported_parser_sha256": {
            "develop_envbench_jvm_budget_models.py": digest(ROOT / "scripts/experiments/develop_envbench_jvm_budget_models.py"),
            "audit_envbench_transport_input_development.py": digest(ROOT / "scripts/experiments/audit_envbench_transport_input_development.py"),
        },
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap, "feature_names": FEATURE22,
        "transport_formula": transport["formula"],
        "fit_offset": offset,
        "active_prefixes": {"fit": len(fit), "calibration": len(cal)},
        "training": train_receipt,
        "calibration_rule": "q=max(0,max family-max score on 64 model_stop T<=B50 calibration tasks); strict >; each arm separately",
        "thresholds": thresholds, "summaries": summaries,
        "paired_vs_same_guard_direct_remainder_two_seed": comparisons,
        "per_task_rows": rows,
        "limits": [
            "Same 132 calibration tasks set q and evaluate policy; post-hoc development only.",
            "Transported input uses pre-current-call tool content but raw ATIF-like logs do not contain a live pre-dispatch hook.",
            "No verifier result labels or untouched test trajectories were accessed.",
        ],
    }
    path = OUT / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(path),
                      "B50": {arm: summaries[arm]["all"] for arm in names},
                      "paired": comparisons}, indent=2))


if __name__ == "__main__":
    main()
