from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    EXPECTED, MANIFEST, SOURCE, digest, parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_next_input_control import (
    BASE, LOCK, MODEL_A, MODEL_B, REPORT, exclusive_json,
    points_from_traces, predict_points, replay,
)


NOTE = BASE / "QCLIP_NOTE.md"
OUT = BASE / "qclip_development_diagnostic.json"
EXPECTED_NOTE = "3e4f7b853186ef270437c180605938b3c4f814a0858e2440ec5ea2cbdf44e245"
EXPECTED_LOCK = "2bcb08849acfde0c933329f1002d3f195aa02d8288abf1c0b312af5bfbea3e9f"
EXPECTED_REPORT = "e631dc184b864d82c51b6590de39353b2116b02e4ed846826700c7039e922591"
EXPECTED_MODELS = {
    "A_fit_B_cal": "e11b9d50b5e11a2343bf85834c70580c7ac6b0af8940eaba03bc61271213ddfe",
    "B_fit_A_cal": "c963625782555b13572a1f05c69780e02e788be910d67571450647912ad32541",
}


def clip(q: dict[str, float]) -> dict[str, float]:
    return {name: max(0.0, value) for name, value in q.items()}


def metrics(replay_result: dict) -> dict:
    return {name: result["overall"] for name, result in replay_result.items()}


def check_raw(recomputed: dict, frozen: dict) -> None:
    for name, stats in recomputed.items():
        previous = frozen[name]["overall"]
        for key, value in stats["overall"].items():
            old = previous.get(key, 0)
            if abs(value - old) > 1e-6:
                raise AssertionError(f"Raw replay changed: {name}/{key}: {value} != {old}")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    checks = {
        NOTE: EXPECTED_NOTE, LOCK: EXPECTED_LOCK, REPORT: EXPECTED_REPORT,
        MODEL_A: EXPECTED_MODELS["A_fit_B_cal"],
        MODEL_B: EXPECTED_MODELS["B_fit_A_cal"],
        SOURCE: EXPECTED["source"], MANIFEST: EXPECTED["manifest"],
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    primary = json.loads(LOCK.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    reserved = set(manifest["test_case_ids"])
    if len(cases) != 44 or len(reserved) != 90 or cases & reserved:
        raise ValueError("Identity split changed")
    actors = sorted(primary["actor_codes"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("Calibration-only projection changed")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in actors:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")
    points = points_from_traces(traces, primary["actor_codes"])
    point_indexes = defaultdict(list)
    for index, point in enumerate(points):
        point_indexes[point["trace_index"]].append(index)

    directions = {}
    for label, model_path in (("A_fit_B_cal", MODEL_A), ("B_fit_A_cal", MODEL_B)):
        spec = frozen["directions"][label]
        all_eval = set(spec["calibration_case_ids"])
        model = lgb.Booster(model_file=str(model_path))
        pred = predict_points(model, points, all_eval)
        caps = spec["actor_median_caps"]
        q_raw = spec["calibration"]["q"]
        full_raw = replay(traces, points, point_indexes, all_eval, caps, pred, q_raw)
        check_raw(full_raw, spec["calibration_resubstitution"])
        full_clip = replay(traces, points, point_indexes, all_eval, caps, pred, clip(q_raw))
        nested = []
        for item in spec["nested_11_cal_11_eval"]:
            eval_cases = set(item["evaluation_case_ids"])
            raw = replay(traces, points, point_indexes, eval_cases, caps, pred, item["thresholds"])
            check_raw(raw, item["evaluation"])
            clipped = replay(traces, points, point_indexes, eval_cases, caps, pred, clip(item["thresholds"]))
            nested.append({
                "calibration_case_ids": item["calibration_case_ids"],
                "evaluation_case_ids": item["evaluation_case_ids"],
                "raw_q": item["thresholds"], "clipped_q": clip(item["thresholds"]),
                "raw_metrics": metrics(raw), "clipped_metrics": metrics(clipped),
                "raw_by_case": {name: result["case"] for name, result in raw.items()},
                "clipped_by_case": {name: result["case"] for name, result in clipped.items()},
            })
        directions[label] = {
            "raw_q": q_raw, "clipped_q": clip(q_raw),
            "raw_metrics": metrics(full_raw), "clipped_metrics": metrics(full_clip),
            "nested_11_cal_11_eval": nested,
        }
    receipt = {
        "status": "44-case development only; q clipping before reserved-case score",
        "rule_note_sha256": EXPECTED_NOTE, "primary_lock_sha256": EXPECTED_LOCK,
        "frozen_development_report_sha256": EXPECTED_REPORT,
        "model_sha256": EXPECTED_MODELS,
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "directions": directions,
    }
    exclusive_json(OUT, receipt)
    print(json.dumps({
        "report": str(OUT), "sha256": digest(OUT),
        "A_fit_B_cal": {key: value for key, value in directions["A_fit_B_cal"].items()
                        if key in ("raw_q", "clipped_q", "raw_metrics", "clipped_metrics")},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
