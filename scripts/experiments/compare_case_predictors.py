from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

from token_prediction.case_study.evidence import load_run_evidence, read_seed_files
from token_prediction.case_study.forecast import PointCalibration, summarize
from token_prediction.case_study.gateway import BoundedGateway
from token_prediction.case_study.self_prediction import parse_estimate, prediction_record
from token_prediction.collection.dsh.env_file import parse_environment_file
from token_prediction.collection.dsh.provider_environment import resolve_hosted_environment
from token_prediction.estimators.composable_segment import ComposableSegmentModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--forecasts-only", action="store_true",
                        help="Score the frozen predictor locally without baseline API calls")
    parser.add_argument("--reuse-baseline-from", type=Path,
                        help="Reuse recorded responses after verifying identical cutoff prompts (offline only)")
    parser.add_argument("--no-token-call-budget", action="store_true",
                        help="Remove cumulative baseline request/token limits; retain accounting")
    parser.add_argument("--direct-connection", action="store_true",
                        help="Connect to the configured gateway without environment proxy settings")
    parser.add_argument("--allow-ineligible-terminal", action="store_true")
    args = parser.parse_args()
    if args.execute and args.forecasts_only:
        parser.error("--execute and --forecasts-only are mutually exclusive")
    if args.reuse_baseline_from and not args.forecasts_only:
        parser.error("--reuse-baseline-from requires --forecasts-only")
    args.output.mkdir(parents=True, exist_ok=True)
    model = ComposableSegmentModel.load(args.model)
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    provenance = json.loads((args.model.parent / "provenance.json").read_text(encoding="utf-8"))
    excluded = set(provenance["task_family_partition"])
    calibrators = {}
    for point, value in training["calibrators"].items():
        calibrators[point] = PointCalibration(**{
            field.name: tuple(value[field.name]) if field.name == "task_ids" else value[field.name]
            for field in fields(PointCalibration)})
    prepared = []
    for directory in args.run_dir:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        task = run["task"]
        family = task["task_family_id"]
        if family in excluded:
            raise ValueError("case task appeared in training, validation or calibration")
        model_id = run["condition"]["actor_model_id"]
        root = Path(__file__).resolve().parents[2]
        seed = (root / task["pack_path"] / task["seed_workspace"]).resolve()
        if not seed.is_relative_to((root / "benchmarks/dsh").resolve()):
            raise ValueError("this small pilot only accepts the preselected authored P0 seeds")
        seed_files = sorted(path for path in seed.rglob("*") if path.is_file())
        if len(seed_files) > 12 or sum(path.stat().st_size for path in seed_files) > 32768:
            raise ValueError("initial seed exceeds the small-case evidence budget")
        hashes = {path.relative_to(seed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in seed_files}
        initial_environment = read_seed_files(seed, hashes)
        horizon_options = ({"pilot_max_calls": model.config.max_horizon}
                           if run.get("cap_profile", {}).get("max_logical_calls") is None else {})
        for item in load_run_evidence(directory, initial_environment=initial_environment,
                                     allow_ineligible_terminal=args.allow_ineligible_terminal,
                                     request_surface_features=training.get("feature_schema") in
                                     {"causal_request_surface_v2", "causal_request_surface_history_v3"},
                                     history_cost_features=training.get("feature_schema") ==
                                     "causal_request_surface_history_v3",
                                     **horizon_options):
            prepared.append({"run_directory": str(directory), "task_id": family,
                             "actor_model": model_id, **item})

    for i, item in enumerate(prepared):
        (args.output / f"prompt-{i:02}.txt").write_text(item["prompt"], encoding="utf-8")
    protocol = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "selection": "first recorded cutoff at each of four prediction points",
                "baseline_max_calls": None if args.no_token_call_budget else 8,
                "baseline_max_output_tokens": 768,
                "baseline_local_token_budget": None if args.no_token_call_budget else 150000,
                "initial_inspection": "complete immutable seed snapshot; no baseline tool exploration",
                "points": [{"point_id": item["point_id"], "actor_model": item["actor_model"],
                            "prediction_point": item["prediction_point"],
                            "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                            "prompt_utf8_bytes": len(item["prompt"].encode())} for item in prepared]}
    protocol_path = args.output / "protocol.json"
    if args.direct_connection:
        protocol["baseline_connection_mode"] = "direct"
    reused = {}
    if args.reuse_baseline_from:
        source = args.reuse_baseline_from.resolve()
        prior_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
        if prior_protocol["points"] != protocol["points"]:
            raise ValueError("baseline reuse requires identical ordered cutoff prompts and actor models")
        prior_rows = json.loads((source / "paired-results.json").read_text(encoding="utf-8"))
        reused = {row["point_id"]: row["self_prediction"] for row in prior_rows}
        if set(reused) != {item["point_id"] for item in prepared}:
            raise ValueError("baseline source does not cover all requested points")
        ledger_bytes = (source / "baseline-ledger.json").read_bytes()
        protocol["baseline_reuse"] = {"source": str(source), "new_api_calls": 0,
                                     "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest()}
    if (args.output / "baseline-ledger.json").exists() and protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("cannot change model or evidence after baseline requests were sent")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.reuse_baseline_from:
        (args.output / "baseline-ledger.json").write_bytes(ledger_bytes)
    if not args.execute and not args.forecasts_only:
        print(json.dumps({"prepared_points": len(prepared), "paid_calls": 0}))
        return
    gateway = None
    if args.execute:
        values = resolve_hosted_environment({}, file_environment=parse_environment_file(
            args.env_file.read_text(encoding="utf-8"))).environment
        gateway = BoundedGateway(base_url=values["DSH_BASE_URL"],
            api_key=values[values["DSH_API_KEY_ENV"]], ledger_path=args.output / "baseline-ledger.json",
            models=("claude-4.5-haiku", "deepseek/deepseek-v4-flash"),
            max_prompt_bytes=60000, max_calls=protocol["baseline_max_calls"],
            max_accounted_tokens=protocol["baseline_local_token_budget"],
            direct_connection=args.direct_connection)
    result_path = args.output / "paired-results.json"
    previous_results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
    previous_by_id = {row["point_id"]: row for row in previous_results}
    requested = {row["request_id"] for row in gateway.ledger["requests"]} if gateway else set()
    results = []
    for i, item in enumerate(prepared):
        if item["point_id"] in requested:
            if item["point_id"] not in previous_by_id:
                raise ValueError("request already sent but result missing; reconstruct locally before resume")
            results.append(previous_by_id[item["point_id"]])
            continue
        point, query = item["prediction_point"], item["core_query"]
        started = time.perf_counter()
        forecast = model.forecast(query, samples=256, seed=20260919+i)
        is_call = point in {"call_start", "in_call_update"}
        if not is_call and "horizon_exhausted" in forecast and forecast["horizon_exhausted"].any():
            raise ValueError("forecast horizon exhausted before sampled task termination; "
                             "a full-task forecast is unavailable")
        sample_key = "call_samples" if is_call else "task_remaining_samples"
        ours = summarize(forecast[sample_key][0], point=point, task_id=item["task_id"],
                         confirmed_in_scope=float(query["confirmed_call"][0]) if is_call else 0,
                         spent=float(query["spent"][0]), calibration=calibrators[point],
                         terminal_actual=0 if point == "task_update" and
                         bool(query["is_terminal"][0]) else None)
        ours["latency_ms"] = (time.perf_counter()-started)*1000
        if ours["status"] == "calibrated":
            ours["status"] = "source_calibration_transfer"
            ours["target_coverage_validated"] = False
            ours["calibration_scope"] = "historical source tasks; different actor and call caps"
        result = {"task_id": item["task_id"], "actor_model": item["actor_model"],
                  "prediction_point": point, "point_id": item["point_id"],
                  "ours": ours, "labels": item["labels"],
                  "unknown_fields": item["unknown_fields"],
                  "run_metadata": item["run_metadata"],
                  "initial_inspection": item["initial_inspection"]}
        if point == "task_update" and bool(query["is_terminal"][0]):
            result["self_prediction"] = {"status": "observed_terminal", "total_tokens": 0,
                                         "paid_calls": 0}
        elif item["point_id"] in reused:
            result["self_prediction"] = reused[item["point_id"]]
            result["baseline_reused_from"] = str(args.reuse_baseline_from.resolve())
        elif gateway is None:
            result["self_prediction"] = {"status": "not_requested_offline", "paid_calls": 0}
        else:
            try:
                response = gateway.complete(model=item["actor_model"], prompt=item["prompt"],
                                            request_id=item["point_id"])
                bounds = {"total_tokens": int(query["confirmed_call"][0])} if is_call else None
                estimate = parse_estimate(response["content"], prediction_point=point,
                                          confirmed_usage_in_scope=bounds)
                result["self_prediction"] = prediction_record(estimate,
                    prediction_point=point, usage=response["usage"],
                    elapsed_seconds=response["wall_seconds"])
            except Exception as exc:

                reason = str(exc).replace(values[values["DSH_API_KEY_ENV"]], "[REDACTED]")
                result["self_prediction"] = {"status": "failed", "error_type": type(exc).__name__,
                                             "reason": reason[:500]}
        results.append(result)
        (args.output / "paired-results.json").write_text(json.dumps(results, indent=2),
                                                         encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
