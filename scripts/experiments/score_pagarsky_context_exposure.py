from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/external/pagarsky_agent_trace_4b05b2f/data/agenttrace.parquet"
SPLIT = ROOT / "workspace/experiments/pagarsky_agent_trace_feasibility_20260923/identity_split_v2.json"
PREREG = ROOT / "docs/local/pagarsky-full-cost-preregistration-2026-09-23.md"
DEV_SCRIPT = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
DEV_REPORT = ROOT / "workspace/experiments/pagarsky_context_exposure_dev_20260923/report.json"
PREDICTOR = ROOT / "scripts/experiments/predict_pagarsky_context_exposure.py"
OUTPUT = ROOT / "workspace/experiments/pagarsky_context_exposure_test_20260923"
SOURCE_SHA = "11c04875f9b6e91b117b5739d748e6bd4f8fa621e95faf430fa40d5d4d223c97"
SPLIT_SHA = "14b9fb4ad1319a9fe288952b62c72db3edbefd55d0e4a350d231a0784af993ea"
PREREG_SHA = "c73e2f8686c3219d2f388d1169493e489af2d1fe4d6d45a55816588967b390b0"
DEV_SCRIPT_SHA = "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc"
DEV_REPORT_SHA = "0a547a5816f56187f5e95fda9c8ec222a53f035b63c965af7816d2fd8ca86d4d"
TEST_KEY_SHA = "629b248506924d9e7eefdac7fe194e06f0320694f7e3af6bc71b376eab22bdfa"
ACTORS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
METHODS = (
    "context_exposure", "count_mean", "direct_total", "direct_two_seed",
    "call_clock", "empirical", "count_current_input", "count_current_plus_growth",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _identities() -> set[tuple[str, int, str]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test = {tuple(item) for item in split["holdout_task_keys"]}
    if len(test) != 300 or split["holdout_key_sha256"] != TEST_KEY_SHA:
        raise ValueError("test task-key roster changed")
    filters = [[("dataset_name", "=", dataset),
                ("task_id", "in", sorted(key[1] for key in test if key[0] == dataset))]
               for dataset in ("mbpp", "nl2bash")]
    rows = pq.read_table(SOURCE, columns=["dataset_name", "task_id", "model"], filters=filters).to_pylist()
    identities = {(str(row["dataset_name"]), int(row["task_id"]), str(row["model"])) for row in rows}
    if len(rows) != 600 or len(identities) != 600 or {(a, b) for a, b, _ in identities} != test:
        raise ValueError("test identity coverage changed")
    if any(actor not in ACTORS for _, _, actor in identities):
        raise ValueError("test actor changed")
    return identities


def verify() -> tuple[dict, dict, dict]:
    for path, expected in ((SOURCE, SOURCE_SHA), (SPLIT, SPLIT_SHA),
                           (PREREG, PREREG_SHA), (DEV_SCRIPT, DEV_SCRIPT_SHA),
                           (DEV_REPORT, DEV_REPORT_SHA)):
        if _sha256(path) != expected:
            raise ValueError(f"frozen source or development input changed: {path}")
    fit_path = OUTPUT / "fit_receipt.json"
    manifest_path = OUTPUT / "prediction_manifest.json"
    predictions_path = OUTPUT / "predictions.jsonl"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fit["predictor_sha256"] != _sha256(PREDICTOR) or fit["scorer_sha256"] != _sha256(Path(__file__)):
        raise ValueError("frozen predictor or scorer changed after fit")
    if manifest["fit_receipt_sha256"] != _sha256(fit_path) or manifest["predictions_sha256"] != _sha256(predictions_path):
        raise ValueError("frozen fit or predictions changed")
    if manifest["predictor_sha256"] != _sha256(PREDICTOR) or manifest["scorer_sha256"] != _sha256(Path(__file__)):
        raise ValueError("prediction code changed")
    if manifest["source_sha256"] != SOURCE_SHA or manifest["split_sha256"] != SPLIT_SHA or manifest["test_key_sha256"] != TEST_KEY_SHA:
        raise ValueError("prediction source or roster changed")
    if manifest["selected_arm"] != "context_exposure" or manifest["method_names"] != list(METHODS):
        raise ValueError("prediction method changed")
    for actor in ACTORS:
        for head, entry in fit["actors"][actor]["heads"].items():
            if _sha256(OUTPUT / "models" / entry["file"]) != entry["sha256"]:
                raise ValueError(f"frozen model changed: {actor}/{head}")
    identities = _identities()
    statuses = {}
    for item in manifest["trace_statuses"]:
        identity = (item["dataset_name"], item["task_id"], item["actor"])
        if identity in statuses or identity not in identities:
            raise ValueError("duplicate or unknown trace status")
        if item["status"] not in ("valid", "fewer_than_two_steps", "invalid_step_json", "invalid_step_order", "invalid_usage"):
            raise ValueError("unknown trace status")
        statuses[identity] = item
    if set(statuses) != identities or manifest["task_keys"] != 300 or manifest["trace_identities"] != 600:
        raise ValueError("prediction status identity coverage changed")
    forecasts = defaultdict(dict)
    with predictions_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            identity = (row["dataset_name"], row["task_id"], row["actor"])
            k = row["k"]
            if identity not in statuses or type(k) is not int or k < 1 or k in forecasts[identity]:
                raise ValueError("invalid or duplicate prefix forecast")
            values = row["predictions"]
            if set(values) != set(METHODS) or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("invalid forecast values")
            forecasts[identity][k] = row
    if sum(len(rows) for rows in forecasts.values()) != manifest["forecast_rows_including_terminal"]:
        raise ValueError("prediction row count changed")
    for identity, item in statuses.items():
        count = item["prefix_count"]
        status = item["status"]
        if type(count) is not int or count < 0:
            raise ValueError("invalid prefix count")
        if status == "valid" and count < 2:
            raise ValueError("valid trace has no nonterminal prefix")
        if status == "fewer_than_two_steps" and count >= 2:
            raise ValueError("short trace status changed")
        if status.startswith("invalid_") and count != 0:
            raise ValueError("malformed trace has forecasts")
        if set(forecasts[identity]) != set(range(1, count + 1)):
            raise ValueError("missing or surplus prefix forecast")
        for k, row in forecasts[identity].items():
            if row["terminal"] is not (k == count):
                raise ValueError("terminal sentinel location changed")
            if row["terminal"] and any(value != 0 for value in row["predictions"].values()):
                raise ValueError("terminal sentinel must be zero")
    return manifest, statuses, forecasts


def _test_rows() -> list[dict]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test = {tuple(item) for item in split["holdout_task_keys"]}
    filters = [[("dataset_name", "=", dataset),
                ("task_id", "in", sorted(key[1] for key in test if key[0] == dataset))]
               for dataset in ("mbpp", "nl2bash")]
    return pq.read_table(
        SOURCE, columns=["dataset_name", "task_id", "model", "llm_steps_json"],
        filters=filters,
    ).to_pylist()


def _trace_steps(payload: str) -> tuple[str, list[int], list[int]]:
    try:
        steps = json.loads(payload)
    except (TypeError, ValueError):
        return "invalid_step_json", [], []
    if not isinstance(steps, list) or [step.get("step_number") if isinstance(step, dict) else None for step in steps] != list(range(1, len(steps) + 1)):
        return "invalid_step_order", [], []
    inputs = [step.get("input_tokens") for step in steps]
    outputs = [step.get("output_tokens") for step in steps]
    if any(type(value) is not int or value <= 0 for value in (*inputs, *outputs)):
        return "invalid_usage", [], []
    if len(steps) < 2:
        return "fewer_than_two_steps", inputs, outputs
    return "valid", inputs, outputs


def _metrics(by_actor: dict, keys: list[tuple[str, int]]) -> dict:
    if not keys:
        return {"paired_task_count": 0}
    per_actor = {}
    for actor in ACTORS:
        rows = [by_actor[actor][key] for key in keys]
        denominator = sum(row["mean_target"] for row in rows)
        per_actor[actor] = {
            method: {
                "wape": sum(row["mae"][method] for row in rows) / denominator,
                "task_equal_mae": float(np.mean([row["mae"][method] for row in rows])),
                "mean_task_wape": float(np.mean([row["mae"][method] / row["mean_target"] for row in rows])),
            }
            for method in METHODS
        }
    macro = {method: float(np.mean([per_actor[actor][method]["wape"] for actor in ACTORS])) for method in METHODS}
    rng = np.random.default_rng(20260924)
    draws = rng.integers(0, len(keys), size=(5000, len(keys)))
    boot = {}
    for method in METHODS:
        actor_draws = []
        for actor in ACTORS:
            rows = [by_actor[actor][key] for key in keys]
            targets = np.asarray([row["mean_target"] for row in rows])
            errors = np.asarray([row["mae"][method] for row in rows])
            actor_draws.append(errors[draws].sum(axis=1) / targets[draws].sum(axis=1))
        boot[method] = np.mean(actor_draws, axis=0)
    comparisons = {}
    for control in ("direct_two_seed", "count_mean", "call_clock"):
        delta = boot["context_exposure"] - boot[control]
        comparisons[control] = {
            "delta": macro["context_exposure"] - macro[control],
            "paired_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
        }
    return {"paired_task_count": len(keys), "macro_wape": macro,
            "per_actor": per_actor, "comparisons": comparisons,
            "bootstrap_draws": 5000, "bootstrap_seed": 20260924}


def score() -> None:
    manifest, statuses, forecasts = verify()
    for path in (OUTPUT / "score_started.json", OUTPUT / "per_task_scored.jsonl", OUTPUT / "score.json"):
        if path.exists():
            raise FileExistsError("scorer already started or completed; refusing second score")
    _write_json(OUTPUT / "score_started.json", {
        "status": "exclusive one-shot score started before any test suffix labels were read",
        "manifest_sha256": _sha256(OUTPUT / "prediction_manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "scorer_sha256": _sha256(Path(__file__)),
    })
    by_actor = {actor: {} for actor in ACTORS}
    exclusions = Counter()
    seen = set()
    for row in _test_rows():
        key = (str(row["dataset_name"]), int(row["task_id"]))
        actor = str(row["model"])
        identity = (key[0], key[1], actor)
        if identity in seen or identity not in statuses:
            raise ValueError("test trace identity changed during score")
        seen.add(identity)
        status, inputs, outputs = _trace_steps(row["llm_steps_json"])
        prediction_status = statuses[identity]
        if prediction_status["status"] != status or prediction_status["prefix_count"] != len(inputs):
            raise ValueError("prediction-time test structure changed")
        if status != "valid":
            exclusions[status] += 1
            continue
        targets = [float(sum(inputs[k:]) + sum(outputs[k:])) for k in range(1, len(inputs))]
        if any(target <= 0 for target in targets):
            raise ValueError("nonpositive target")
        mae = {
            method: float(np.mean([abs(forecasts[identity][k]["predictions"][method] - target)
                                   for k, target in enumerate(targets, start=1)]))
            for method in METHODS
        }
        by_actor[actor][key] = {
            "prefix_count": len(targets), "mean_target": float(np.mean(targets)), "mae": mae,
        }
    if seen != set(statuses):
        raise ValueError("test trace coverage changed during score")
    paired = sorted(set.intersection(*(set(by_actor[actor]) for actor in ACTORS)))
    all_metrics = _metrics(by_actor, paired)
    by_benchmark = {
        dataset: _metrics(by_actor, [key for key in paired if key[0] == dataset])
        for dataset in ("mbpp", "nl2bash")
    }
    with (OUTPUT / "per_task_scored.jsonl").open("x", encoding="utf-8") as file:
        for key in paired:
            file.write(json.dumps({"dataset_name": key[0], "task_id": key[1],
                                   "actors": {actor: by_actor[actor][key] for actor in ACTORS}},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")
    invalid_count = sum(exclusions[name] for name in ("invalid_step_json", "invalid_step_order", "invalid_usage"))
    gate = len(paired) >= 100 and invalid_count / 600 <= 0.05
    comparisons = all_metrics.get("comparisons", {})
    mechanistic = gate and all(comparisons[control]["paired_ci95"][1] < 0 for control in ("direct_two_seed", "count_mean"))
    result = {
        "status": "one-shot test scored; no test-driven model selection or refit",
        "source_sha256": SOURCE_SHA, "split_sha256": SPLIT_SHA,
        "prediction_manifest_sha256": _sha256(OUTPUT / "prediction_manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "scorer_sha256": _sha256(Path(__file__)),
        "score_start_receipt_sha256": _sha256(OUTPUT / "score_started.json"),
        "per_task_scored_sha256": _sha256(OUTPUT / "per_task_scored.jsonl"),
        "task_keys": 300, "trace_identities": 600,
        "excluded_traces_by_reason": dict(exclusions), "invalid_trace_fraction": invalid_count / 600,
        "jointly_eligible_task_keys": len(paired), "inferential_gate_pass": gate,
        "primary": all_metrics, "by_benchmark": by_benchmark,
        "mechanistic_gain_vs_both_strong_controls": mechanistic,
        "paid_api_calls": 0,
    }
    _write_json(OUTPUT / "score.json", result)
    print(json.dumps({"paired_keys": len(paired), "invalid_traces": invalid_count,
                      "macro_wape": all_metrics.get("macro_wape"),
                      "mechanistic_gain": mechanistic}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "score"))
    args = parser.parse_args()
    if args.mode == "verify":
        manifest, statuses, forecasts = verify()
        print(json.dumps({"status": "verified without reading test step payloads or targets",
                          "task_keys": manifest["task_keys"], "trace_identities": len(statuses),
                          "forecast_rows_including_terminal": sum(len(rows) for rows in forecasts.values()),
                          "predictions_sha256": manifest["predictions_sha256"]}, ensure_ascii=False))
    else:
        score()
