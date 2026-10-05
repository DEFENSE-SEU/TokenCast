from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from develop_pagarsky_context_exposure import (
    ACTORS, FEATURES, HEADS, METHODS, PREREG, PREREG_SHA, SOURCE, SOURCE_SHA,
    SPLIT, SPLIT_SHA, _baselines, _dataset, _features, _params, _points, _sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEV_SCRIPT = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
DEV_REPORT = ROOT / "workspace/experiments/pagarsky_context_exposure_dev_20260923/report.json"
SCORER = ROOT / "scripts/experiments/score_pagarsky_context_exposure.py"
OUTPUT = ROOT / "workspace/experiments/pagarsky_context_exposure_test_20260923"
DEV_SCRIPT_SHA = "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc"
DEV_REPORT_SHA = "0a547a5816f56187f5e95fda9c8ec222a53f035b63c965af7816d2fd8ca86d4d"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _keys_and_report() -> tuple[set[tuple[str, int]], set[tuple[str, int]], dict]:
    required = ((SOURCE, SOURCE_SHA), (SPLIT, SPLIT_SHA),
                (PREREG, PREREG_SHA), (DEV_SCRIPT, DEV_SCRIPT_SHA),
                (DEV_REPORT, DEV_REPORT_SHA))
    for path, expected in required:
        if _sha256(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    dev = {tuple(item) for item in split["development_task_keys"]}
    test = {tuple(item) for item in split["holdout_task_keys"]}
    if len(dev) != 400 or len(test) != 300 or dev & test:
        raise ValueError("frozen identity split changed")
    report = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    if report["metrics"]["selected_by_prespecified_dev_wape"] != "context_exposure":
        raise ValueError("selected arm changed")
    if report["feature_names"] != list(FEATURES) or set(report["head_contract"]) != set(HEADS):
        raise ValueError("development model contract changed")
    return dev, test, report


def _filtered_rows(keys: set[tuple[str, int]]) -> list[dict]:
    filters = [[("dataset_name", "=", dataset),
                ("task_id", "in", sorted(key[1] for key in keys if key[0] == dataset))]
               for dataset in ("mbpp", "nl2bash")]
    return pq.read_table(
        SOURCE, columns=["dataset_name", "task_id", "model", "llm_steps_json"],
        filters=filters,
    ).to_pylist()


def _load_dev_traces(keys: set[tuple[str, int]]) -> tuple[dict, Counter]:
    rows = _filtered_rows(keys)
    if len(rows) != 800:
        raise ValueError("development trace count changed")
    traces = defaultdict(dict)
    reasons = Counter()
    for row in rows:
        key = (str(row["dataset_name"]), int(row["task_id"]))
        actor = str(row["model"])
        if key not in keys or actor not in ACTORS or actor in traces[key]:
            raise ValueError("development trace identity changed")
        try:
            steps = json.loads(row["llm_steps_json"])
        except (TypeError, ValueError):
            steps = None
            reasons["invalid_step_json"] += 1
        if steps is not None:
            if not isinstance(steps, list) or [s.get("step_number") if isinstance(s, dict) else None for s in steps] != list(range(1, len(steps) + 1)):
                steps = None
                reasons["invalid_step_order"] += 1
        if steps is None:
            traces[key][actor] = ([], [])
            continue
        inputs = [step.get("input_tokens") for step in steps]
        outputs = [step.get("output_tokens") for step in steps]
        if any(type(value) is not int or value <= 0 for value in (*inputs, *outputs)):
            traces[key][actor] = ([], [])
            reasons["invalid_usage"] += 1
            continue
        if len(steps) < 2:
            reasons["fewer_than_two_steps"] += 1
        traces[key][actor] = (inputs, outputs)
    if set(traces) != keys or any(set(pair) != set(ACTORS) for pair in traces.values()):
        raise ValueError("development actor pairing changed")
    return traces, reasons


def fit() -> None:
    if OUTPUT.exists():
        raise FileExistsError("fit output exists; refusing to overwrite")
    dev, _, report = _keys_and_report()
    traces, reasons = _load_dev_traces(dev)
    if dict(reasons) != report["structural_exclusions_by_trace"]:
        raise ValueError("development structural audit changed")
    points = _points(traces, {key: 0 for key in dev})
    for actor in ACTORS:
        if len(points[actor]) != report["nonterminal_points_by_actor"][actor]:
            raise ValueError("development point count changed")
    OUTPUT.mkdir(parents=True)
    model_dir = OUTPUT / "models"
    model_dir.mkdir()
    actor_receipts = {}
    for actor_index, actor in enumerate(ACTORS):
        rows = points[actor]
        clock, empirical = _baselines(rows)
        heads = {}
        for head, (label, max_rounds, seed, stochastic) in HEADS.items():
            rounds = int(statistics.median(
                fold["actors"][actor]["best_rounds"][head] for fold in report["folds"]
            ))
            if not 1 <= rounds <= max_rounds:
                raise ValueError(f"invalid fixed full-development rounds: {actor}, {head}")
            model = lgb.train(_params(seed, stochastic), _dataset(rows, label), num_boost_round=rounds)
            path = model_dir / f"actor{actor_index}_{head}.txt"
            model.save_model(str(path))
            heads[head] = {"file": path.name, "rounds": rounds, "sha256": _sha256(path)}
        actor_receipts[actor] = {
            "train_points": len(rows), "train_traces": sum(bool(traces[key][actor][0] and len(traces[key][actor][0]) >= 2) for key in dev),
            "heads": heads, "call_clock": {str(k): v for k, v in clock.items()},
            "empirical": empirical,
        }
        print(f"FULL_DEV_ACTOR_FIT {actor}", flush=True)
    receipt = {
        "status": "all 400 development keys fitted; test step payloads unread",
        "source_sha256": SOURCE_SHA, "split_sha256": SPLIT_SHA,
        "preregistration_sha256": PREREG_SHA, "dev_script_sha256": DEV_SCRIPT_SHA,
        "dev_report_sha256": DEV_REPORT_SHA, "predictor_sha256": _sha256(Path(__file__)),
        "scorer_sha256": _sha256(SCORER), "selected_arm": "context_exposure",
        "feature_names": list(FEATURES), "development_task_keys": len(dev),
        "holdout_task_keys_unopened": 300, "actors": actor_receipts,
        "test_steps_read": False, "test_future_labels_computed": False, "paid_api_calls": 0,
    }
    _write_json(OUTPUT / "fit_receipt.json", receipt)


def _verify_fit() -> tuple[set[tuple[str, int]], dict]:
    _, test, _ = _keys_and_report()
    receipt_path = OUTPUT / "fit_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt["predictor_sha256"] != _sha256(Path(__file__)) or
            receipt["scorer_sha256"] != _sha256(SCORER) or
            receipt["source_sha256"] != SOURCE_SHA or
            receipt["split_sha256"] != SPLIT_SHA or
            receipt["selected_arm"] != "context_exposure"):
        raise ValueError("fit receipt or code changed")
    for actor in ACTORS:
        for head in HEADS:
            entry = receipt["actors"][actor]["heads"][head]
            if _sha256(OUTPUT / "models" / entry["file"]) != entry["sha256"]:
                raise ValueError("frozen model changed")
    return test, receipt


def _stream_steps(payload: str):

    if not isinstance(payload, str):
        raise ValueError("invalid_step_json")
    decoder = json.JSONDecoder()
    position = 0
    while position < len(payload) and payload[position].isspace():
        position += 1
    if position >= len(payload) or payload[position] != "[":
        raise ValueError("invalid_step_json")
    position += 1
    count = 0
    while True:
        while position < len(payload) and payload[position].isspace():
            position += 1
        if position >= len(payload):
            raise ValueError("invalid_step_json")
        if payload[position] == "]":
            position += 1
            if payload[position:].strip():
                raise ValueError("invalid_step_json")
            return
        if count:
            if payload[position] != ",":
                raise ValueError("invalid_step_json")
            position += 1
            while position < len(payload) and payload[position].isspace():
                position += 1
        try:
            step, position = decoder.raw_decode(payload, position)
        except json.JSONDecodeError as error:
            raise ValueError("invalid_step_json") from error
        count += 1
        yield step


def _test_features(payload: str) -> tuple[list[tuple[float, ...]], int]:
    inputs, outputs, features = [], [], []
    for step in _stream_steps(payload):
        if not isinstance(step, dict) or type(step.get("step_number")) is not int or step["step_number"] != len(inputs) + 1:
            raise ValueError("invalid_step_order")
        input_value, output_value = step.get("input_tokens"), step.get("output_tokens")
        if type(input_value) is not int or input_value <= 0 or type(output_value) is not int or output_value <= 0:
            raise ValueError("invalid_usage")
        inputs.append(input_value)
        outputs.append(output_value)
        features.append(_features(inputs, outputs))
    return features, len(inputs)


def _forecasts(matrix: np.ndarray, ks: list[int], models: dict, baselines: dict) -> dict[str, np.ndarray]:
    heads = {head: model.predict(matrix) for head, model in models.items()}
    count = np.maximum(1.0, heads["future_calls"])
    current = matrix[:, FEATURES.index("current_input")]
    growth = heads["mean_growth"]
    output = np.maximum(0.0, heads["mean_output"])
    clock = {int(key): value for key, value in baselines["call_clock"].items()}
    nearest = np.asarray([clock[min(clock, key=lambda value: abs(value - k))] for k in ks])
    forecasts = {
        "context_exposure": np.maximum(0.0, count * (current + growth + output)),
        "count_mean": count * np.maximum(0.0, heads["mean_total"]),
        "direct_total": np.maximum(0.0, heads["direct_total"]),
        "direct_two_seed": np.maximum(0.0, (heads["direct_seed_a"] + heads["direct_seed_b"]) / 2.0),
        "call_clock": nearest,
        "empirical": np.full(len(ks), baselines["empirical"]),
        "count_current_input": count * current,
        "count_current_plus_growth": np.maximum(0.0, count * (current + growth)),
    }
    if set(forecasts) != set(METHODS) or any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in forecasts.values()):
        raise ValueError("nonfinite or negative forecast")
    return forecasts


def predict() -> None:
    test, receipt = _verify_fit()
    predictions_path = OUTPUT / "predictions.jsonl"
    manifest_path = OUTPUT / "prediction_manifest.json"
    if predictions_path.exists() or manifest_path.exists() or (OUTPUT / "score_started.json").exists():
        raise FileExistsError("prediction or scoring receipt exists; refusing overwrite")
    rows = _filtered_rows(test)
    if len(rows) != 600:
        raise ValueError("test trace count changed")
    models = {}
    for actor in ACTORS:
        models[actor] = {
            head: lgb.Booster(model_file=str(OUTPUT / "models" / receipt["actors"][actor]["heads"][head]["file"]))
            for head in HEADS
        }
    identities = set()
    statuses = []
    forecast_rows = []
    for row in rows:
        key = (str(row["dataset_name"]), int(row["task_id"]))
        actor = str(row["model"])
        trace_key = (key[0], key[1], actor)
        if key not in test or actor not in ACTORS or trace_key in identities:
            raise ValueError("test identity changed")
        identities.add(trace_key)
        try:
            feature_rows, count = _test_features(row["llm_steps_json"])
        except ValueError as error:
            if str(error) not in ("invalid_step_json", "invalid_step_order", "invalid_usage"):
                raise
            statuses.append({"dataset_name": key[0], "task_id": key[1], "actor": actor,
                             "status": str(error), "prefix_count": 0})
            continue
        statuses.append({"dataset_name": key[0], "task_id": key[1], "actor": actor,
                         "status": "valid" if count >= 2 else "fewer_than_two_steps",
                         "prefix_count": count})
        if count == 0:
            continue
        ks = list(range(1, count))
        if ks:
            matrix = np.asarray(feature_rows[:-1], dtype=np.float64)
            values = _forecasts(matrix, ks, models[actor], receipt["actors"][actor])
            for index, k in enumerate(ks):
                forecast_rows.append({
                    "dataset_name": key[0], "task_id": key[1], "actor": actor,
                    "k": k, "terminal": False,
                    "predictions": {method: float(values[method][index]) for method in METHODS},
                })
        forecast_rows.append({"dataset_name": key[0], "task_id": key[1], "actor": actor,
                              "k": count, "terminal": True,
                              "predictions": {method: 0.0 for method in METHODS}})
    if len(identities) != 600 or {item[:2] for item in identities} != test:
        raise ValueError("test identity coverage changed")
    with predictions_path.open("x", encoding="utf-8") as file:
        for row in forecast_rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "frozen predictions; test suffix target not computed or scored",
        "source_sha256": SOURCE_SHA, "split_sha256": SPLIT_SHA,
        "test_key_sha256": "629b248506924d9e7eefdac7fe194e06f0320694f7e3af6bc71b376eab22bdfa",
        "preregistration_sha256": PREREG_SHA, "dev_report_sha256": DEV_REPORT_SHA,
        "fit_receipt_sha256": _sha256(OUTPUT / "fit_receipt.json"),
        "predictor_sha256": _sha256(Path(__file__)), "scorer_sha256": _sha256(SCORER),
        "predictions_sha256": _sha256(predictions_path),
        "selected_arm": "context_exposure", "method_names": list(METHODS),
        "task_keys": len(test), "trace_identities": len(identities),
        "forecast_rows_including_terminal": len(forecast_rows),
        "trace_statuses": sorted(statuses, key=lambda item: (item["dataset_name"], item["task_id"], item["actor"])),
        "test_future_labels_computed": False, "test_scored": False, "paid_api_calls": 0,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"forecast_rows": len(forecast_rows), "trace_status_counts": dict(Counter(item["status"] for item in statuses)),
                      "predictions_sha256": manifest["predictions_sha256"], "manifest_sha256": _sha256(manifest_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "predict"))
    args = parser.parse_args()
    {"fit": fit, "predict": predict}[args.mode]()
