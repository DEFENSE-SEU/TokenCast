from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, EXPECTED, MANIFEST, SOURCE, digest, nearest_rank,
    parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_call_start import family_q, half_rank


OUT = BASE / "call_start_joint_development.json"
FIRST_LOCK = BASE / "budget_calibration_lock.json"
CALL_START = BASE / "call_start_development.json"
EXPECTED_FIRST_LOCK_SHA256 = "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d"
EXPECTED_CALL_START_SHA256 = "ea181932000282f30335e63b479e1e55d2a3ca7025902e994b5cddde0d87a60f"
CAP_GRID = (0.25, 0.5, 0.75)


def joint_q(maxima: dict[str, dict[int, float]], cases: set[int], scale: dict[str, int]) -> float:
    case_max = {
        case_id: max(maxima[actor].get(case_id, 0.0) / max(scale[actor], 1) for actor in scale)
        for case_id in cases
    }
    return family_q(case_max, cases)


def replay(traces: list[dict], cases: set[int], caps: dict[str, int], q_actor: dict[str, float],
           q_joint: float, scale: dict[str, int]) -> dict[str, Counter]:
    stats: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        actor = trace["actor"]
        if trace["case_id"] not in cases or actor not in caps:
            continue
        cap, total = caps[actor], trace["total"]
        ref = min(total, cap)
        s = stats[actor]
        s["eligible_traces"] += 1
        s["reference_cost_tokens"] += ref
        s["reference_completed"] += int(total <= cap)
        s["reference_reward"] += trace["score"] if total <= cap else 0.0
        spent = 0
        stops = {"actor": False, "joint": False, "one_step_input_oracle": False}
        for j in range(1, len(trace["inputs"])):
            spent += trace["inputs"][j - 1] + trace["outputs"][j - 1]
            if spent >= cap:
                continue
            current_input = trace["inputs"][j]
            prior_input = trace["inputs"][j - 1]
            lowers = {
                "actor": max(0.0, prior_input - q_actor[actor]),
                "joint": max(0.0, prior_input - q_joint * scale[actor]),
                "one_step_input_oracle": current_input,
            }
            for name, lower in lowers.items():
                if stops[name] or spent + lower <= cap:
                    continue
                stops[name] = True
                s[f"{name}_alerts"] += 1
                s[f"{name}_saved_tokens"] += ref - spent
                s[f"{name}_fractional_saving_sum"] += (ref - spent) / max(ref, 1)
                if total <= cap:
                    s[f"{name}_false_aborts"] += 1
                    s[f"{name}_lost_reward"] += trace["score"]
                if name != "one_step_input_oracle" and lower > current_input:
                    s[f"{name}_trigger_lower_violations"] += 1
    return stats


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(FIRST_LOCK) != EXPECTED_FIRST_LOCK_SHA256 or digest(CALL_START) != EXPECTED_CALL_START_SHA256:
        raise ValueError("Earlier reports changed")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or roster changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    first = json.loads(FIRST_LOCK.read_text(encoding="utf-8"))
    cases = set(manifest["calibration_case_ids"])
    active = set(first["active_actors"])
    if len(cases) != 44 or len(active) != 14:
        raise ValueError("Development roster changed")
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(cases)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in cases for row in rows):
        raise ValueError("Calibration-only projection mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")
    maxima: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for trace in traces:
        actor, case_id = trace["actor"], trace["case_id"]
        for j in range(1, len(trace["inputs"])):
            maxima[actor][case_id] = max(
                maxima[actor][case_id], trace["inputs"][j - 1] - trace["inputs"][j], 0.0,
            )

    q_actor = {actor: family_q(maxima[actor], cases) for actor in sorted(active)}
    full_scale = {actor: first["actor_caps_nearest_rank"][actor]["0.5"] for actor in sorted(active)}
    q_full_joint = joint_q(maxima, cases, full_scale)
    full_grid = {}
    for p in CAP_GRID:
        caps = {actor: first["actor_caps_nearest_rank"][actor][str(p)] for actor in sorted(active)}
        values = replay(traces, cases, caps, q_actor, q_full_joint, full_scale)
        full_grid[str(p)] = {actor: dict(v) for actor, v in sorted(values.items())}

    ranked = sorted(cases, key=lambda case_id: (half_rank(case_id), case_id))
    halves = [set(ranked[:22]), set(ranked[22:])]
    cross_grid: dict[str, dict[str, Counter]] = {
        str(p): defaultdict(Counter) for p in CAP_GRID
    }
    folds = []
    for train_cases, eval_cases in ((halves[0], halves[1]), (halves[1], halves[0])):
        totals: dict[str, list[int]] = defaultdict(list)
        for trace in traces:
            if trace["case_id"] in train_cases:
                totals[trace["actor"]].append(trace["total"])
        scale = {actor: nearest_rank(totals[actor], .5) for actor in sorted(active)}
        qa = {actor: family_q(maxima[actor], train_cases) for actor in sorted(active)}
        qj = joint_q(maxima, train_cases, scale)
        joint_covered = sum(
            max(maxima[actor].get(case_id, 0.0) / max(scale[actor], 1) for actor in scale) <= qj
            for case_id in eval_cases
        )
        fold_entry = {
            "train_case_ids": sorted(train_cases), "eval_case_ids": sorted(eval_cases),
            "actor_scale_median_total": scale, "actor_q": qa,
            "joint_normalized_q": qj, "joint_case_covered_of_22": joint_covered,
            "caps": {},
        }
        for p in CAP_GRID:
            caps = {actor: nearest_rank(totals[actor], p) for actor in sorted(active)}
            values = replay(traces, eval_cases, caps, qa, qj, scale)
            for actor, val in values.items():
                cross_grid[str(p)][actor].update(val)
            fold_entry["caps"][str(p)] = caps
        folds.append(fold_entry)

    report = {
        "status": "calibration-only development; test trajectories/scores unopened",
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "first_lock_sha256": EXPECTED_FIRST_LOCK_SHA256,
        "call_start_report_sha256": EXPECTED_CALL_START_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "joint_scale_rule": "actor median strict trace total from calibration cases; fixed across cap grid",
        "joint_case_residual": "max over actor, trace repeat and next-call j>=2 of positive (I_previous-I_current)/actor_scale",
        "joint_quantile_rank": "ceil((n_cases+1)*0.9): 41/44 or 21/22",
        "full44_joint_normalized_q": q_full_joint,
        "full44_grid": full_grid,
        "twofold_22_22": folds,
        "twofold_out_of_half_grid": {
            p: {actor: dict(v) for actor, v in sorted(actors.items())}
            for p, actors in sorted(cross_grid.items())
        },
        "eligible_traces": len(traces),
    }
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "sha256": digest(OUT), "full44_joint_q": q_full_joint,
        "crossfold_joint_covered_cases_of_44": sum(fold["joint_case_covered_of_22"] for fold in folds),
        "median_grid": {
            arm: {
                key: sum(values.get(f"{arm}_{key}", 0) for values in cross_grid["0.5"].values())
                for key in ("alerts", "saved_tokens", "false_aborts", "lost_reward")
            } for arm in ("actor", "joint", "one_step_input_oracle")
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
