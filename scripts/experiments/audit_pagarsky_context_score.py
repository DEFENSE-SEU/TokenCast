from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "workspace/experiments/pagarsky_context_exposure_test_20260923"
ACTORS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
METHODS = ("context_exposure", "direct_two_seed", "count_mean", "call_clock")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    score_path = OUTPUT / "score.json"
    task_path = OUTPUT / "per_task_scored.jsonl"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == score["jointly_eligible_task_keys"] == 150
    assert _sha(task_path) == score["per_task_scored_sha256"]
    deviations = []
    actor_wape = {}
    for actor in ACTORS:
        targets = np.asarray([row["actors"][actor]["mean_target"] for row in rows])
        assert np.all(targets > 0)
        actor_wape[actor] = {}
        for method in METHODS:
            errors = np.asarray([row["actors"][actor]["mae"][method] for row in rows])
            actor_wape[actor][method] = float(errors.sum() / targets.sum())
            deviations.append(abs(actor_wape[actor][method] - score["primary"]["per_actor"][actor][method]["wape"]))
    macro = {method: float(np.mean([actor_wape[actor][method] for actor in ACTORS])) for method in METHODS}
    for method in METHODS:
        deviations.append(abs(macro[method] - score["primary"]["macro_wape"][method]))
    draws = np.random.default_rng(20260924).integers(0, len(rows), size=(5000, len(rows)))
    intervals = {}
    for method in METHODS:
        actor_draws = []
        for actor in ACTORS:
            targets = np.asarray([row["actors"][actor]["mean_target"] for row in rows])
            errors = np.asarray([row["actors"][actor]["mae"][method] for row in rows])
            actor_draws.append(errors[draws].sum(axis=1) / targets[draws].sum(axis=1))
        intervals[method] = np.mean(actor_draws, axis=0)
    for control in ("direct_two_seed", "count_mean", "call_clock"):
        comparison = score["primary"]["comparisons"][control]
        delta = macro["context_exposure"] - macro[control]
        ci = np.quantile(intervals["context_exposure"] - intervals[control], [0.025, 0.975])
        deviations.append(abs(delta - comparison["delta"]))
        deviations.append(float(np.max(np.abs(ci - comparison["paired_ci95"]))))
    assert max(deviations) < 1e-12
    print(json.dumps({
        "status": "independent arithmetic audit passed; frozen scorer not rerun",
        "score_sha256": _sha(score_path), "per_task_sha256": _sha(task_path),
        "paired_task_keys": len(rows), "macro_wape": macro,
        "max_absolute_discrepancy": max(deviations),
    }, indent=2))


if __name__ == "__main__":
    main()
