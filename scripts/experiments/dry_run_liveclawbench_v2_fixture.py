from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.experiments import run_liveclawbench_budget_v2 as runner


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "workspace/experiments/liveclawbench_budget_v2_fixture_20260923_v2"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "fixture.parquet"
    if source.exists():
        raise FileExistsError(source)
    rows = []
    actors = [f"model{x:02d}" for x in range(17)]
    active = actors[:14]
    for case_id in range(90):
        for actor in actors:
            for repeat in range(3):
                first_input, next_input = (80, 70) if repeat == 0 else (40, 30)
                steps = [
                    {"step_id": 1, "source": "agent", "message": "a", "tool_calls": [],
                     "metrics": {"prompt_tokens": first_input, "completion_tokens": 1, "cached_tokens": 0}},
                    {"step_id": 2, "source": "agent", "message": "b", "tool_calls": [],
                     "metrics": {"prompt_tokens": next_input, "completion_tokens": 1, "cached_tokens": 0}},
                ]
                if actor == "model00" and repeat == 2:
                    steps[-1]["tool_calls"] = [{"name": "continuation"}]
                if actor == "model02" and repeat == 2:
                    steps[0]["metrics"]["cached_tokens"] = first_input + 1
                trajectory = {
                    "schema_version": "ATIF-v1.2", "steps": steps, "extra": {},
                    "final_metrics": {
                        "total_prompt_tokens": first_input + next_input,
                        "total_completion_tokens": 2,
                        "total_cached_tokens": sum(s["metrics"]["cached_tokens"] for s in steps),
                        "total_steps": 2,
                    },
                }
                rows.append({
                    "case_id": case_id, "sample_id": f"{case_id}-{actor}-{repeat}",
                    "model_name": actor, "trajectory": json.dumps(trajectory),
                    "score": None if actor == "model01" and repeat == 2 else 1.0,
                })
    pq.write_table(pa.Table.from_pylist(rows), source)
    lock = {
        "test_case_ids": list(range(90)), "active_actors": active,
        "source_sha256": hash_file(source), "identity_sha256": "fixture-identity-only",
        "actor_caps_A": {actor: {"0.25": 60, "0.5": 100, "0.75": 160} for actor in active},
        "model_decision_q_final": {
            cap: {name: 0.0 for name in runner.MODEL_FIELD.values()} for cap in runner.GRID
        },
        "actor_input_drop_q_B": {actor: 0.0 for actor in active},
        "joint_input_drop_q_B": 0.0, "global_input_drop_q_B": 0.0,
        "full_remaining_q_B": 1000000.0, "full_remaining_q_old": 1000000.0,
    }
    lock_path = OUT / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    predictions_path = OUT / "predictions.jsonl"
    statuses = Counter()
    with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: (item["case_id"], item["model_name"], item["sample_id"])):
            status, detail, points, _ = runner.prefix_points(row, set(active))
            statuses[status] += 1
            for point in points:
                point["g"] = {name: float(point["prior_input"]) for name in runner.MODEL_FIELD.values()}
                point["full_remaining_pred"] = 0.0
            stops = {}
            if row["model_name"] in active:
                for cap_label in runner.GRID:
                    cap = lock["actor_caps_A"][row["model_name"]][cap_label]
                    stops[cap_label] = {arm: None for arm in runner.DECISION_ARMS}
                    for point in points:
                        for arm in runner.DECISION_ARMS:
                            if (stops[cap_label][arm] is None and
                                    runner.stop_condition(arm, point, cap, cap_label, row["model_name"], lock)):
                                stops[cap_label][arm] = {"k": point["k"], "spent": point["spent"]}
            record = {
                "case_id": row["case_id"], "sample_id": row["sample_id"],
                "actor": row["model_name"], "prefix_status": status,
                "prefix_detail": detail, "points": points, "stops": stops,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest_path = OUT / "prediction_manifest.json"
    manifest_path.write_text(json.dumps({
        "lock_sha256": hash_file(lock_path), "predictions_sha256": hash_file(predictions_path),
        "slot_count": 4590, "test_case_count": 90, "score_column_projected": False,
    }), encoding="utf-8")
    runner.SOURCE = source
    runner.LOCK = lock_path
    runner.PREDICTIONS = predictions_path
    runner.PREDICTION_MANIFEST = manifest_path
    runner.SCORE_STARTED = OUT / "score_started.json"
    runner.PER_SLOT = OUT / "per_slot_scored.jsonl"
    runner.PER_CASE = OUT / "per_case_scored.jsonl"
    runner.SCORE_REPORT = OUT / "score_report.json"
    runner.load_lock = lambda: lock
    runner.score()
    report = json.loads(runner.SCORE_REPORT.read_text(encoding="utf-8"))
    assert report["planned_slot_count"] == 4590
    assert report["test_case_count"] == 90
    assert report["strict_active_traces"] == 14 * 3 * 90 - 3 * 90
    assert report["bootstrap"]["quality_case_macro_difference"]["verifier_reward"]["point"] == 0.0
    assert report["cap_grid"]["0.5"]["arms"]["primary_historical"]["saved_tokens"] > 0
    print(json.dumps({
        "fixture_score_report": str(runner.SCORE_REPORT),
        "fixture_report_sha256": hash_file(runner.SCORE_REPORT),
        "strict": report["strict_active_traces"], "prefix_statuses": dict(statuses),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
