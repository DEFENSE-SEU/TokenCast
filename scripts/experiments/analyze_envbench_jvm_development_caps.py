from __future__ import annotations

from hashlib import sha256
import json
from math import ceil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923"
LOCK = BASE / "jvm_bash_agent_4o_cohort.lock.json"
LOCK_SHA = "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"
V2 = BASE / "development_sentinel_resolution_v2.json"
V2_SHA = "061722185a8741e27341593f67060d3dbf8c6b70b1fab99323aa22873998cfac"
OUT = BASE / "development_cap_candidates.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # Keep archived task exports readable under the current labels.
    for row in rows:
        if row.get("terminal_class") == "synthetic_step_limit_censored":
            row["terminal_class"] = "step_limit_censored"
    return rows


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[ceil(fraction * len(ordered)) - 1]


def describe(rows: list[dict], cap: int) -> dict:
    observed = [row["nominal_total_tokens"] for row in rows]
    return {
        "tasks": len(rows),
        "observed_total_tokens": sum(observed),
        "observed_total_ge_cap": sum(value >= cap for value in observed),
        "observed_total_lt_cap": sum(value < cap for value in observed),
        "observed_trace_reference_tokens": sum(min(value, cap) for value in observed),
    }


def main() -> None:
    if digest(LOCK) != LOCK_SHA or digest(V2) != V2_SHA:
        raise ValueError("Frozen identity or development resolution changed")
    if OUT.exists():
        raise FileExistsError(OUT)
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    rows = {}
    for group in ("fit", "calibration"):
        path = BASE / f"development_{group}_tasks_v2.jsonl"
        if digest(path) != v2["feature_exports"][f"{group}_tasks_v2"]["sha256"]:
            raise ValueError("Development task export changed")
        rows[group] = read_jsonl(path)
    if len(rows["fit"]) != 400 or len(rows["calibration"]) != 132:
        raise ValueError("Development split changed")
    fit_completed = [row["nominal_total_tokens"] for row in rows["fit"]
                     if row["terminal_class"] == "model_stop"]
    if len(fit_completed) != 370:
        raise ValueError("Fit natural-stop denominator changed")
    caps = {str(fraction): nearest_rank(fit_completed, fraction)
            for fraction in (.25, .5, .75)}
    all_rows = rows["fit"] + rows["calibration"]
    payload = {
        "status": "development_only_cap_candidate_analysis",
        "identity_lock_sha256": LOCK_SHA,
        "development_resolution_sha256": V2_SHA,
        "analysis_code_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "cap_population": "370 fit tasks with final real AI finish_reason=stop",
        "cap_quantile_rule": "nearest rank: sorted_total[ceil(fraction*n)-1]",
        "caps": caps,
        "total_development_observed_tokens": sum(row["nominal_total_tokens"] for row in all_rows),
        "sentinel_observed_tokens": sum(row["nominal_total_tokens"] for row in all_rows
                                        if row["terminal_class"] == "step_limit_censored"),
        "by_cap": {},
        "interpretation": {
            "model_stop": "Observed end after model stop; still an agent completion proxy",
            "step_limit_censored": "Observed trace stops at step limit; T_obs is a lower bound on natural total",
            "observed_trace_reference_tokens": "sum min(T_obs, cap), not a natural-completion reference for censored tasks",
        },
    }
    for fraction, cap in caps.items():
        entry = {}
        for group, group_rows in (("fit", rows["fit"]), ("calibration", rows["calibration"]),
                                  ("development_all", all_rows)):
            entry[group] = {
                terminal_class: describe([row for row in group_rows
                                          if row["terminal_class"] == terminal_class], cap)
                for terminal_class in ("model_stop", "step_limit_censored")
            }
        payload["by_cap"][fraction] = entry
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"caps": caps, "development_all": {
        fraction: value["development_all"] for fraction, value in payload["by_cap"].items()
    }, "sha256": digest(OUT)}, indent=2))


if __name__ == "__main__":
    main()
