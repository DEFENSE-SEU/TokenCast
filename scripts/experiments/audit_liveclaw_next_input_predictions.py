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
    points_from_traces, predict_points,
)


OUT = BASE / "next_input_predictive_diagnostic.json"
EXPECTED_LOCK = "2bcb08849acfde0c933329f1002d3f195aa02d8288abf1c0b312af5bfbea3e9f"
EXPECTED_REPORT = "e631dc184b864d82c51b6590de39353b2116b02e4ed846826700c7039e922591"
EXPECTED_MODELS = {
    "A_fit_B_eval": "e11b9d50b5e11a2343bf85834c70580c7ac6b0af8940eaba03bc61271213ddfe",
    "B_fit_A_eval": "c963625782555b13572a1f05c69780e02e788be910d67571450647912ad32541",
}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for path, expected in (
        (LOCK, EXPECTED_LOCK), (REPORT, EXPECTED_REPORT),
        (MODEL_A, EXPECTED_MODELS["A_fit_B_eval"]),
        (MODEL_B, EXPECTED_MODELS["B_fit_A_eval"]),
        (SOURCE, EXPECTED["source"]), (MANIFEST, EXPECTED["manifest"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    reserved = set(manifest["test_case_ids"])
    if len(cases) != 44 or len(reserved) != 90 or cases & reserved:
        raise ValueError("Identity split changed")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("Calibration-only projection changed")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in lock["actor_codes"]:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")
    points = points_from_traces(traces, lock["actor_codes"])
    fit_a = set(lock["fit_case_ids"])
    fit_b = set(lock["calibration_case_ids"])
    model_a = lgb.Booster(model_file=str(MODEL_A))
    model_b = lgb.Booster(model_file=str(MODEL_B))
    predictions = {
        **predict_points(model_a, points, fit_b),
        **predict_points(model_b, points, fit_a),
    }
    if len(predictions) != len(points):
        raise AssertionError("Every development prefix needs one out-of-fit prediction")
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"absolute_error": 0.0, "target_sum": 0.0, "prefixes": 0})))
    for index, point in enumerate(points):
        case_id, actor = point["case_id"], point["actor"]
        target = point["next_input"]
        for method, estimate in (("regression", predictions[index]),
                                 ("prior_input", float(point["prior_input"]))):
            error = abs(estimate - target)
            for scope, key in (("all", "all"), ("actor", actor), ("case", str(case_id))):
                bucket = totals[scope][key][method]
                bucket["absolute_error"] += error
                bucket["target_sum"] += target
                bucket["prefixes"] += 1
    result = {}
    for scope, named in totals.items():
        result[scope] = {}
        for key, method_values in named.items():
            result[scope][key] = {
                method: {**value, "wape": value["absolute_error"] / max(value["target_sum"], 1)}
                for method, value in method_values.items()
            }
    report = {
        "status": "44-case development-only out-of-fit next-input accuracy; no model tuning",
        "primary_lock_sha256": EXPECTED_LOCK,
        "development_report_sha256": EXPECTED_REPORT,
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "model_sha256": EXPECTED_MODELS,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "out_of_fit_prefixes": len(points),
        "metrics": result,
    }
    exclusive_json(OUT, report)
    print(json.dumps({"report": str(OUT), "sha256": digest(OUT),
                      "overall": result["all"]["all"]}, sort_keys=True))


if __name__ == "__main__":
    main()
