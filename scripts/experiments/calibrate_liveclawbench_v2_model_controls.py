from __future__ import annotations

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, EXPECTED, SOURCE, digest, parse_primary_eligible,
)
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features


ROOT = Path(__file__).resolve().parents[2]
CORE = BASE / "v2_core_ab_calibration.json"
OUT = BASE / "v2_model_controls_ab_calibration.json"
EXTERNAL = ROOT / "workspace/experiments/exgentic_next_input_baselines_20260923_v2"
DOMAIN = ROOT / "workspace/experiments/liveclawbench_next_input_control_20260923"
EXPECTED_HASHES = {
    "core": "d9187f020356ccf1c85279c260ec912059c9748817b7ec76a2dbaee7ee1d2e5c",
    "external_receipt": "ae9a89da25d868342b0f14a0f25a1a23aad83005edc32327972533d76adf84b4",
    "external_a": "fd7d6c0303f5aa2f322bc9d5c504618e0ae99b77486ec707ba3f8d0f826d1cb0",
    "external_b": "a9943a7c7ca430df643dcd3c4ba02c14ab60f258a229908bb732f4763fecf3ab",
    "external_clock": "ee87a18e2a3e492fb31c8861c778d828e1aa9e6d803dcd6286ea3f27be965d6e",
    "domain_lock": "2bcb08849acfde0c933329f1002d3f195aa02d8288abf1c0b312af5bfbea3e9f",
    "domain_model": "e11b9d50b5e11a2343bf85834c70580c7ac6b0af8940eaba03bc61271213ddfe",
    "feature": "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc",
}
ARMS = ("historical", "external_direct", "external_clock", "domain_direct")
GRID = (0.25, 0.5, 0.75)


def q21(values: dict[int, float], cases: set[int]) -> float:
    ordered = sorted(values.get(case_id, -math.inf) for case_id in cases)
    return ordered[20]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {
        "core": CORE,
        "external_receipt": EXTERNAL / "fit_receipt.json",
        "external_a": EXTERNAL / "next_input_seed_1.txt",
        "external_b": EXTERNAL / "next_input_seed_2.txt",
        "external_clock": EXTERNAL / "call_clock_input.json",
        "domain_lock": DOMAIN / "primary_Afit_Bcal_lock.json",
        "domain_model": DOMAIN / "model_fit_A.txt",
        "feature": ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py",
    }
    for name, path in paths.items():
        if digest(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"Pinned control changed: {name}")
    if digest(SOURCE) != EXPECTED["source"]:
        raise ValueError("Pinned source changed")
    core = json.loads(CORE.read_text(encoding="utf-8"))
    domain_lock = json.loads(paths["domain_lock"].read_text(encoding="utf-8"))
    B = set(core["B_case_ids"])
    if len(B) != 22 or set(domain_lock["calibration_case_ids"]) != B:
        raise ValueError("B roster mismatch")
    if domain_lock["actor_median_caps"] != {
        actor: values["0.5"] for actor, values in core["actor_caps_from_A"].items()
    }:
        raise ValueError("A actor caps mismatch")
    if domain_lock["feature_names"] != [*FEATURES, "actor_code"]:
        raise ValueError("Domain feature contract mismatch")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(B)),
    ).to_pylist()
    if len(rows) != 1122 or any(row["case_id"] not in B for row in rows):
        raise ValueError("B-only projection mismatch")
    active = set(core["active_actors"])
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 820:
        raise ValueError("B strict denominator changed")
    external_a = lgb.Booster(model_file=str(paths["external_a"]))
    external_b = lgb.Booster(model_file=str(paths["external_b"]))
    domain = lgb.Booster(model_file=str(paths["domain_model"]))
    if external_a.feature_name() != list(FEATURES) or external_b.feature_name() != list(FEATURES):
        raise ValueError("External feature names changed")
    if domain.feature_name() != [*FEATURES, "actor_code"]:
        raise ValueError("Domain feature names changed")
    clock = {int(k): int(v) for k, v in json.loads(paths["external_clock"].read_text()).items()}
    clock_ks = sorted(clock)
    metadata, x, x_domain = [], [], []
    for trace_index, trace in enumerate(traces):
        inputs, outputs, spent = trace["inputs"], trace["outputs"], 0
        actor_code = domain_lock["actor_codes"][trace["actor"]]
        for k in range(1, len(inputs)):
            spent += inputs[k - 1] + outputs[k - 1]
            feat = _features(inputs[:k], outputs[:k])
            x.append(feat)
            x_domain.append((*feat, float(actor_code)))
            metadata.append((trace_index, trace["case_id"], trace["actor"], k, spent, inputs[k - 1]))
    matrix = np.asarray(x, dtype=np.float64)
    domain_matrix = np.asarray(x_domain, dtype=np.float64)
    if matrix.shape != (len(metadata), len(FEATURES)) or domain_matrix.shape != (len(metadata), len(FEATURES) + 1):
        raise ValueError("Prefix matrix changed")
    pred_a = external_a.predict(matrix, num_threads=4)
    pred_b = external_b.predict(matrix, num_threads=4)
    pred_domain = domain.predict(domain_matrix, num_threads=4)
    points = []
    for row, pa, pb, pd in zip(metadata, pred_a, pred_b, pred_domain):
        trace_index, case_id, actor, k, spent, previous_input = row
        nearest_k = min(clock_ks, key=lambda value: (abs(value - k), value))
        values = {
            "historical": float(previous_input),
            "external_direct": max(0.0, float((pa + pb) / 2)),
            "external_clock": float(clock[nearest_k]),
            "domain_direct": max(0.0, float(np.expm1(pd))),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite control point")
        points.append((trace_index, case_id, actor, spent, values))
    result = {
        "status": "model controls calibrated on B only; test unopened",
        "source_sha256": EXPECTED["source"],
        "core_ab_sha256": EXPECTED_HASHES["core"],
        "code_sha256": digest(Path(__file__)),
        "pinned_inputs_sha256": EXPECTED_HASHES,
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "B_strict_traces": len(traces),
        "B_call_starts": len(points),
        "q_raw_and_final": {},
    }
    for p in GRID:
        label = str(p)
        caps = {actor: values[label] for actor, values in core["actor_caps_from_A"].items()}
        case_max = {arm: {case_id: -math.inf for case_id in B} for arm in ARMS}
        for trace_index, case_id, actor, spent, values in points:
            cap = caps[actor]
            if spent >= cap or traces[trace_index]["total"] > cap:
                continue
            for arm in ARMS:
                score = (spent + values[arm] - cap) / cap
                case_max[arm][case_id] = max(case_max[arm][case_id], score)
        calibrated = {}
        for arm in ARMS:
            q_raw = q21(case_max[arm], B)
            calibrated[arm] = {
                "q_raw": q_raw, "q_final": max(0.0, q_raw),
                "B_complete_candidate_cases": sum(math.isfinite(v) for v in case_max[arm].values()),
            }
        if calibrated["historical"]["q_raw"] != core["primary_decision_risk_from_B"][label]["q"]:
            raise ValueError("Historical q disagrees with core calibration")
        if p == .5 and calibrated["domain_direct"]["q_raw"] != domain_lock["case_max_direct_decision_q"]["regression"]:
            raise ValueError("Domain median q disagrees with independent lock")
        result["q_raw_and_final"][label] = calibrated
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "sha256": digest(OUT), "B_call_starts": len(points),
        "q_median": result["q_raw_and_final"]["0.5"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
