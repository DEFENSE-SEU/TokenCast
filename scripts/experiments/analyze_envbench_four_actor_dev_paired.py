from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/four_actor_budget_certificate_development.json"
SOURCE_SHA = "26b46f36eac596efd5412ab446188bd1a633be936869927290497283fdf3da1f"
DEST = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/four_actor_paired_development_summary.json"
ARMS = ("minus_64_engineering", "previous_input_same_gate", "current_input_same_gate_oracle")


def main() -> None:
    if sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Source development report drift")
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    actor_order = list(report["actors"])
    actor_rows = {}
    common = set(report["cross_actor_paired_132_repo"]["common_identified_repo_ids"])
    if len(actor_order) != 4 or len(common) != 129:
        raise ValueError("Development actor or common repo count drift")
    for actor in actor_order:
        actor_rows[actor] = {}
        for arm in ARMS:
            rows = report["actors"][actor]["per_repo_rows"][arm]
            by_id = {row["task_id"]: row for row in rows}
            if len(by_id) != len(rows) or not common <= by_id.keys():
                raise ValueError(f"Task coverage drift: {actor}/{arm}")
            actor_rows[actor][arm] = by_id
    ids = sorted(common)
    ref = np.asarray([[actor_rows[actor][ARMS[0]][task_id]["reference_tokens"]
                       for task_id in ids] for actor in actor_order], dtype=float)
    if not np.all(np.isfinite(ref)) or np.any(ref <= 0):
        raise ValueError("Nonpositive reference")
    saved = {arm: np.asarray([[actor_rows[actor][arm][task_id]["saved_tokens"]
                               for task_id in ids] for actor in actor_order], dtype=float)
             for arm in ARMS}
    for arm in ARMS:
        if np.any(saved[arm] < 0):
            raise ValueError("Negative saving")
        for ai, actor in enumerate(actor_order):
            for ti, task_id in enumerate(ids):
                if actor_rows[actor][arm][task_id]["reference_tokens"] != ref[ai, ti]:
                    raise ValueError("Arm reference differs")

    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(ids), size=(10000, len(ids)))

    def equal_actor_ratio(matrix: np.ndarray, index: np.ndarray | None = None) -> np.ndarray | float:
        if index is None:
            return float(np.mean(matrix.sum(axis=1) / ref.sum(axis=1)))
        per_actor = matrix[:, index].sum(axis=2) / ref[:, index].sum(axis=2)
        return per_actor.mean(axis=0)

    summaries = {}
    for arm in ARMS:
        bootstrap = equal_actor_ratio(saved[arm], draws)
        summaries[arm] = {
            "mean_of_four_actor_pooled_saving": equal_actor_ratio(saved[arm]),
            "ci_95_repo_cluster": [float(np.quantile(bootstrap, q)) for q in (.025, .975)],
            "per_actor": {actor: float(saved[arm][ai].sum() / ref[ai].sum())
                          for ai, actor in enumerate(actor_order)},
        }
    comparisons = {}
    candidate = saved[ARMS[0]]
    for other in ARMS[1:]:
        delta = candidate - saved[other]
        bootstrap = equal_actor_ratio(delta, draws)
        comparisons[f"{ARMS[0]}_minus_{other}"] = {
            "mean_paired_delta": equal_actor_ratio(delta),
            "ci_95_repo_cluster": [float(np.quantile(bootstrap, q)) for q in (.025, .975)],
            "one_sided_lower_95": float(np.quantile(bootstrap, .05)),
        }
    output = {
        "status": "four_actor_paired_calibration_development_only",
        "source_sha256": SOURCE_SHA,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "repo_clusters": len(ids),
        "actor_order": actor_order,
        "arms": summaries,
        "comparisons": comparisons,
    }
    with DEST.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report_sha256": sha256(DEST.read_bytes()).hexdigest(),
                      "arms": summaries, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
