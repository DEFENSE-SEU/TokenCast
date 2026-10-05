from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import floor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_four_actor_predictor_dev_20260923"
PREDICTIONS = BASE / "calibration_four_actor_25_fields_v2.jsonl"
RECEIPT = BASE / "calibration_four_actor_25_fields_v2.receipt.json"
DEVELOPMENT = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/learned_cross_actor_delta_control_tool_token_v1.json"
DEVELOPMENT_SHA = "d9caa600fb1d356b8af6e4964549eae05b3e968df4736031ff8329c8d5d30830"
RESIDUAL = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/residual_on_transport_v1/report.json"
RESIDUAL_SHA = "f07e672a1ace2b51701a28deb419c539ef001dde33934f344c09da695bdd5032"
OUT = BASE / "calibration_stream_25_field_audit_v2.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(DEVELOPMENT) != DEVELOPMENT_SHA:
        raise ValueError("Development model/report changed")
    if digest(RESIDUAL) != RESIDUAL_SHA:
        raise ValueError("Residual development report changed")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["predictions_sha256"] != digest(PREDICTIONS) or receipt["group"] != "calibration":
        raise ValueError("Streamed development prediction receipt changed")
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    residual_report = json.loads(RESIDUAL.read_text(encoding="utf-8"))
    residual_points_path = ROOT / residual_report["calibration_points"]["path"]
    if digest(residual_points_path) != residual_report["calibration_points"]["sha256"]:
        raise ValueError("Residual point audit source changed")
    residual_points = {(row["actor"], row["task_id"], row["call_index"]): row
                       for row in (json.loads(line) for line in
                                   residual_points_path.open(encoding="utf-8"))}
    if len(residual_points) != residual_report["calibration_points"]["rows"]:
        raise ValueError("Residual point denominator changed")
    rows = [json.loads(line) for line in PREDICTIONS.open(encoding="utf-8")]
    if len(rows) != 11390:
        raise ValueError("Four-actor calibration AI-row denominator changed")
    by_actor: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_actor[row["actor"]][row["task_id"]].append(row)
    report = {"status": "fit_calibration_only_streamed_learned_replay_audit",
              "script_sha256": digest(Path(__file__)),
              "development_report_sha256": DEVELOPMENT_SHA,
              "residual_report_sha256": RESIDUAL_SHA,
              "residual_points_sha256": digest(residual_points_path),
              "prediction_receipt_sha256": digest(RECEIPT),
              "predictions_sha256": digest(PREDICTIONS),
              "test_trajectory_or_results_opened": False,
              "actors": {}}
    for run, evidence in development["runs"].items():
        slug = run.replace("/", "_")
        group = by_actor[slug]
        expected = {(row["task_id"], row["call_index"]): row
                    for row in evidence["calibration_point_predictions"]}
        eligible = [row for task in group.values() for row in task if row["eligible"]]
        errors = []
        residual_errors = []
        for row in eligible:
            source = expected[(row["task_id"], row["call_index"])]
            integer = floor(max(0.0, source["learned_input"]))
            if row["learned_next_input"] != integer:
                raise ValueError(f"Learned inference mismatch: {run} {row['task_id']}")
            errors.append(abs(integer - source["actual_input"]))
            residual_source = residual_points[(run, row["task_id"], row["call_index"])]
            if row["residual_next_input"] != residual_source["residual_next_input"]:
                raise ValueError(f"Residual inference mismatch: {run} {row['task_id']}")
            residual_errors.append(abs(row["residual_next_input"] -
                                       residual_source["observed_input"]))
        if not eligible:
            raise ValueError("No active learned points")
        raw_saved = minus_saved = raw_alerts = minus_alerts = raw_disagree = 0
        for source_task in evidence["B50_per_repo_rows"]:
            task_id = source_task["task_id"]
            observed = group[task_id]
            first_raw = next((row for row in observed if row["learned_hit_raw"]), None)
            first_minus = next((row for row in observed if row["learned_hit_minus64"]), None)
            raw_alerts += first_raw is not None
            minus_alerts += first_minus is not None
            if (first_raw["call_index"] if first_raw else None) != source_task["stop_call_index"]:
                raw_disagree += 1
            if source_task["cost_identified"]:
                reference = source_task["reference_tokens"]
                raw_saved += (reference - first_raw["spent_before"] if first_raw else 0)
                minus_saved += (reference - first_minus["spent_before"] if first_minus else 0)
        if (raw_disagree or raw_alerts != evidence["B50_policy"]["alerts"] or
                raw_saved != evidence["B50_policy"]["saved_tokens"]):
            raise ValueError(f"First-stop/cost reproduction mismatch: {run} "
                             f"actual={(raw_disagree,raw_alerts,raw_saved,minus_alerts,minus_saved)} "
                             f"expected={(evidence['B50_policy']['alerts'],evidence['B50_policy']['saved_tokens'])}")
        statuses = Counter(row["abstain_reason"] for task in group.values() for row in task)
        statuses["step_limit_no_usage"] += statuses.pop("synthetic_step_limit_no_usage", 0)
        report["actors"][run] = {
            "calibration_tasks": len(group), "ai_rows": sum(len(v) for v in group.values()),
            "active_eligible_points": len(eligible),
            "learned_integer_equals_floor_frozen_float_points": len(eligible),
            "residual_integer_equals_frozen_development_points": len(eligible),
            "active_integer_next_input_mae_tokens": sum(errors) / len(errors),
            "active_residual_next_input_mae_tokens": sum(residual_errors) / len(residual_errors),
            "development_float_full_gated_mae_tokens": evidence["calibration_input_mae"],
            "B50_raw_alerts": raw_alerts, "B50_raw_saved_tokens": raw_saved,
            "B50_raw_first_stop_disagreements_vs_frozen_float": raw_disagree,
            "B50_minus64_alerts": minus_alerts,
            "B50_minus64_saved_tokens": minus_saved,
            "B50_minus64_interpretation": "new learned minus-64 sensitivity; development report's B50_minus_64_policy is the separate transport certificate",
            "step_limit_no_usage_rows": statuses["step_limit_no_usage"],
        }
    if sum(item["ai_rows"] for item in report["actors"].values()) != 11390:
        raise ValueError("Actor row denominator drift")
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT), "actors": report["actors"]}, indent=2))


if __name__ == "__main__":
    main()
