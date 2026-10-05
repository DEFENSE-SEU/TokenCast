from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from scripts.experiments import run_liveclawbench_budget_v2 as runner
from scripts.experiments.calibrate_liveclawbench_budget import BASE, MANIFEST, SOURCE, digest, parse_primary_eligible
from scripts.experiments.develop_pagarsky_context_exposure import FEATURES


ROOT = Path(__file__).resolve().parents[2]
CORE = BASE / "v2_core_ab_calibration.json"
MODELS = BASE / "v2_model_controls_ab_calibration.json"
OUT = BASE / "v2_real_model_dev_dry_run.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    core = json.loads(CORE.read_text(encoding="utf-8"))
    calibrated = json.loads(MODELS.read_text(encoding="utf-8"))
    roster = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dev = set(roster["calibration_case_ids"])
    active = set(core["active_actors"])
    lock = {
        "model_paths": {
            "external_a": "workspace/experiments/exgentic_next_input_baselines_20260923_v2/next_input_seed_1.txt",
            "external_b": "workspace/experiments/exgentic_next_input_baselines_20260923_v2/next_input_seed_2.txt",
            "remaining_a": "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/models/direct_seed_a.txt",
            "remaining_b": "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/models/direct_seed_b.txt",
            "domain": "workspace/experiments/liveclawbench_next_input_control_20260923/model_fit_A.txt",
        },
        "clock_path": "workspace/experiments/exgentic_next_input_baselines_20260923_v2/call_clock_input.json",
        "actor_caps_A": core["actor_caps_from_A"],
        "active_actors": core["active_actors"],
        "domain_actor_codes": json.loads((ROOT / "workspace/experiments/liveclawbench_next_input_control_20260923/primary_Afit_Bcal_lock.json").read_text())["actor_codes"],
        "model_decision_q_final": {
            cap: {name: values["q_final"] for name, values in arms.items()}
            for cap, arms in calibrated["q_raw_and_final"].items()
        },
        "actor_input_drop_q_B": core["actor_input_drop_q_from_B"],
        "joint_input_drop_q_B": core["joint_input_drop_q_from_B"],
        "global_input_drop_q_B": core["raw_global_input_drop_q_from_B"],
        "full_remaining_q_B": core["full_remaining_q_from_B"],
        "full_remaining_q_old": core["full_remaining_q_from_first_44_lock"],
    }
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(dev)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in dev for row in rows):
        raise ValueError("Development-only projection changed")
    points, x, xd, actors = [], [], [], []
    statuses = Counter()
    strict_count = strict_points = 0
    for row in rows:
        status, _, slot_points, features = runner.prefix_points(row, active)
        statuses[status] += 1
        strict, _ = parse_primary_eligible(row)
        if strict is not None and row["model_name"] in active:
            strict_count += 1
            strict_points += len(slot_points)
            if len(slot_points) != len(strict["inputs"]) - 1:
                raise ValueError("Strict development prefix length changed")
        if row["model_name"] not in active:
            continue
        actor_code = lock["domain_actor_codes"][row["model_name"]]
        for point, feature in zip(slot_points, features):
            points.append(point)
            x.append(feature)
            xd.append((*feature, float(actor_code)))
            actors.append(row["model_name"])
    if strict_count != 1646 or strict_points != 51705:
        raise ValueError("Strict development counts changed")
    models, clock = runner.load_models(lock)
    matrix, domain_matrix = np.asarray(x, dtype=np.float64), np.asarray(xd, dtype=np.float64)
    if matrix.shape != (len(points), len(FEATURES)) or domain_matrix.shape != (len(points), len(FEATURES) + 1):
        raise ValueError("Development feature shape wrong")
    ea = models["external_a"].predict(matrix, num_threads=4)
    eb = models["external_b"].predict(matrix, num_threads=4)
    ra = models["remaining_a"].predict(matrix, num_threads=4)
    rb = models["remaining_b"].predict(matrix, num_threads=4)
    dd = models["domain"].predict(domain_matrix, num_threads=4)
    clock_ks = sorted(clock)
    triggers = Counter()
    for point, actor, pa, pb, pr, ps, pd in zip(points, actors, ea, eb, ra, rb, dd):
        k = point["k"]
        ck = min(clock_ks, key=lambda value: (abs(value - k), value))
        point["g"] = {
            "historical": float(point["prior_input"]),
            "external_direct": max(0.0, float((pa + pb) / 2)),
            "external_clock": float(clock[ck]),
            "domain_direct": max(0.0, float(np.expm1(pd))),
        }
        point["full_remaining_pred"] = max(0.0, float((pr + ps) / 2))
        if not all(math.isfinite(value) for value in (*point["g"].values(), point["full_remaining_pred"])):
            raise ValueError("Nonfinite development model output")
        for cap_label in runner.GRID:
            cap = lock["actor_caps_A"][actor][cap_label]
            for arm in runner.DECISION_ARMS:
                triggers[(cap_label, arm)] += int(runner.stop_condition(arm, point, cap, cap_label, actor, lock))
    result = {
        "status": "real-model inference and stop-condition dry run on 44 development cases only",
        "source_sha256": digest(SOURCE), "identity_sha256": digest(MANIFEST),
        "core_sha256": digest(CORE), "model_calibration_sha256": digest(MODELS),
        "runner_sha256": digest(ROOT / "scripts/experiments/run_liveclawbench_budget_v2.py"),
        "code_sha256": digest(Path(__file__)),
        "development_case_count": len(dev), "development_slots": len(rows),
        "strict_active_traces": strict_count, "strict_call_starts": strict_points,
        "all_active_parseable_call_starts": len(points),
        "prefix_statuses": dict(sorted(statuses.items())),
        "actor_count": len(active), "all_outputs_finite": True,
        "triggered_prefix_conditions": {f"{cap}/{arm}": value for (cap, arm), value in sorted(triggers.items())},
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"receipt": str(OUT), "sha256": digest(OUT),
                      "active_call_starts": len(points), "strict_call_starts": strict_points}, sort_keys=True))


if __name__ == "__main__":
    main()
