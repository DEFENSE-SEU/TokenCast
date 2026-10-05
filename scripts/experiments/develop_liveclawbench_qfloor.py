from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from scripts.experiments.calibrate_liveclawbench_budget import (
    BASE, EXPECTED, MANIFEST, SOURCE, digest, parse_primary_eligible,
)
from scripts.experiments.develop_liveclawbench_decision_risk import replay


OUT = BASE / "decision_risk_qfloor_development.json"
CORE = BASE / "v2_core_ab_calibration.json"
EXPECTED_CORE_SHA256 = "d9187f020356ccf1c85279c260ec912059c9748817b7ec76a2dbaee7ee1d2e5c"
GRID = (0.25, 0.5, 0.75)


def oracle(traces: list[dict], cases: set[int], caps: dict[str, int]) -> dict[int, Counter]:
    result: dict[int, Counter] = defaultdict(Counter)
    for trace in traces:
        if trace["case_id"] not in cases or trace["actor"] not in caps:
            continue
        cap = caps[trace["actor"]]
        ref = min(trace["total"], cap)
        spent = 0
        stats = result[trace["case_id"]]
        stats["eligible_traces"] += 1
        stats["reference_cost_tokens"] += ref
        stats["reference_completed"] += int(trace["total"] <= cap)
        stats["reference_reward"] += trace["score"] if trace["total"] <= cap else 0.0
        for j in range(1, len(trace["inputs"])):
            spent += trace["inputs"][j - 1] + trace["outputs"][j - 1]
            if spent < cap and spent + trace["inputs"][j] > cap:
                stats["alerts"] += 1
                stats["saved_tokens"] += ref - spent
                break
    return result


def summarize(by_case: dict[int, Counter], cases: set[int]) -> dict:
    totals = Counter()
    for case_id in cases:
        totals.update(by_case.get(case_id, Counter()))
    result = dict(totals)
    result["families_without_false_abort"] = sum(
        by_case.get(case_id, Counter())["false_aborts"] == 0 for case_id in cases
    )
    result["case_count"] = len(cases)
    result["pooled_saving_fraction"] = (
        totals["saved_tokens"] / totals["reference_cost_tokens"] if totals["reference_cost_tokens"] else None
    )
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(CORE) != EXPECTED_CORE_SHA256:
        raise ValueError("A/B core calibration changed")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["manifest"]:
        raise ValueError("Pinned source or identity roster changed")
    core = json.loads(CORE.read_text(encoding="utf-8"))
    A, B = set(core["A_case_ids"]), set(core["B_case_ids"])
    if len(A) != 22 or len(B) != 22 or A & B:
        raise ValueError("A/B split changed")
    active = set(core["active_actors"])
    rows = ds.dataset(SOURCE, format="parquet").to_table(
        columns=["case_id", "sample_id", "model_name", "score", "trajectory"],
        filter=ds.field("case_id").isin(sorted(A | B)),
    ).to_pylist()
    if len(rows) != 2244 or any(row["case_id"] not in A | B for row in rows):
        raise ValueError("Development-only projection mismatch")
    traces = []
    for row in rows:
        trace, _ = parse_primary_eligible(row)
        if trace is not None and trace["actor"] in active:
            traces.append(trace)
    if len(traces) != 1646:
        raise ValueError("Strict denominator changed")
    result = {
        "status": "44-case development only; 90-case test unopened",
        "source_sha256": EXPECTED["source"],
        "identity_manifest_sha256": EXPECTED["manifest"],
        "core_ab_sha256": EXPECTED_CORE_SHA256,
        "code_sha256": digest(Path(__file__)),
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "fixed_rule": "q_final=max(0,q_raw); stop iff C<B and (C+I_prev-B)/B>q_final",
        "grid": {},
    }
    for p in GRID:
        label = str(p)
        caps = {actor: values[label] for actor, values in core["actor_caps_from_A"].items()}
        raw_q = core["primary_decision_risk_from_B"][label]["q"]
        arm = {"raw_q_from_B": raw_q, "q_final": max(0.0, raw_q), "caps_from_A": caps}
        for half_name, cases in (("A", A), ("B", B), ("A_plus_B", A | B)):
            relevant = [trace for trace in traces if trace["case_id"] in cases]
            _, raw_case = replay(relevant, cases, caps, raw_q)
            _, floor_case = replay(relevant, cases, caps, max(0.0, raw_q))
            oracle_case = oracle(relevant, cases, caps)
            arm[half_name] = {
                "raw": summarize(raw_case, cases),
                "floor": summarize(floor_case, cases),
                "one_step_input_oracle": summarize(oracle_case, cases),
                "floor_by_case": {str(case_id): dict(floor_case.get(case_id, Counter()))
                                  for case_id in sorted(cases)},
            }
        result["grid"][label] = arm
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "sha256": digest(OUT),
        "median": {half: {
            arm: {k: result["grid"]["0.5"][half][arm].get(k, 0)
                  for k in ("alerts", "false_aborts", "saved_tokens", "lost_reward", "pooled_saving_fraction")}
            for arm in ("raw", "floor", "one_step_input_oracle")
        } for half in ("A", "B", "A_plus_B")},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
