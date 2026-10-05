from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from scripts.experiments.develop_composable_balanced_point_loss import (
    FROZEN,
    POSITIONS,
    SWE,
    SWE_FROZEN,
    load_source,
)
from scripts.experiments.diagnose_composable_bias import select
from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.case_study.segments import with_history_cost_observations
from token_prediction.estimators.composable_segment import ComposableSegmentModel


CANDIDATE = Path("workspace/experiments/composable_balanced_point_loss_one_epoch_20260923/candidate.pt")
OUTPUT = Path("workspace/experiments/composable_balanced_point_loss_audit_20260923")


def audit_one(model, data: dict[str, np.ndarray], frozen_path: Path) -> dict:
    eligible = data["whole_suffix"] & (data["n"] > 0) & (data["state"][:, 7] == 0)
    cohort = select(data, eligible)
    started = time.perf_counter()
    _, candidate = four_point_scores(model, cohort, 128)
    inference_seconds = time.perf_counter() - started
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    by_candidate = {str(row["point_id"]): row for row in candidate}
    by_frozen = {str(row["point_id"]): row for row in frozen}
    if (len(by_candidate) != len(candidate) or len(by_frozen) != len(frozen)
            or set(by_candidate) != set(by_frozen)
            or set(by_candidate) != set(map(str, cohort["point_id"]))):
        raise ValueError("coverage roster mismatch")
    run_by_id = {str(point): str(run) for point, run in zip(cohort["point_id"], cohort["trajectory_id"])}
    result = {"inference_seconds": inference_seconds,
              "points": len(candidate), "families": len(set(map(str, cohort["task_family_id"]))),
              "samples_per_point": 128, "position": {}}
    for position in POSITIONS:
        rows = [row for row in candidate if row["point"] == position]
        sub = {"points": len(rows), "families": len(set(row["task_id"] for row in rows))}
        for name, by_id in (("candidate", by_candidate), ("frozen", by_frozen)):
            by_run: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                record = by_id[row["point_id"]]
                if not record["prediction_available"] or record["actual"] != row["actual"]:
                    raise ValueError("unavailable point or mismatched target")
                covered = float(record["lower"] <= record["actual"] <= record["upper"])
                width = float(record["upper"] - record["lower"])
                by_run[(str(record["task_id"]), run_by_id[row["point_id"]])].append((covered, width))
            by_family: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for (family, _run), values in by_run.items():
                by_family[family].append(tuple(np.mean(values, axis=0)))
            family_summary = [np.mean(values, axis=0) for values in by_family.values()]
            sub[name] = {
                "task_run_point_90_coverage": float(np.mean([value[0] for value in family_summary])),
                "task_run_point_interval_width": float(np.mean([value[1] for value in family_summary])),
                "unavailable_points": 0,
            }
        result["position"][position] = sub
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = ComposableSegmentModel.load(CANDIDATE)
    _, _, _, validation = load_source()
    with np.load(SWE, allow_pickle=False) as archive:
        swe = {name: archive[name] for name in archive.files}
    swe = with_history_cost_observations(swe)
    result = {
        "source_validation": audit_one(model, validation, FROZEN / "candidate-predictions.json"),
        "swe_exposed_diagnostic": audit_one(model, swe, SWE_FROZEN),
        "device": "CPU",
        "paid_api_calls": 0,
        "new_qwen25_dsh_holdout_read": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
