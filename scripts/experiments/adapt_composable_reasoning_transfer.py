from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/runs/composable-multidomain-history-v1"
ZERO = ROOT / "workspace/runs/appendix-f5-composable-zero-shot-20260923-v2"
OUTPUT = ROOT / "workspace/runs/appendix-f5-composable-off-adaptation-20260923-v1"
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
SCALES = {
    "call": ("call_start", "in_call_update"),
    "task": ("task_start", "task_update"),
}
TASK_ID = re.compile(r'"task_id":\s*"([^"\\]+)"')
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260923
SPLIT_SALT = "f5-off-adaptation-20260923-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_audit(provenance: dict, paired: set[str]) -> dict:

    source_runs = [Path(path) for path in provenance["sources"]]
    if not source_runs:
        raise ValueError("source run list is empty")
    low_tasks = set()
    conditions = set()
    for directory in source_runs:
        manifest = read_json(directory / "run.json")
        condition = manifest["condition"]
        conditions.add((condition["actor_model_id"], condition["reasoning_effort"]))
        if condition["actor_model_id"] != "gpt-5.4" or condition["reasoning_effort"] != "low":
            raise ValueError(f"unexpected source condition: {directory}")
        low_tasks.add(manifest["task"]["task_family_id"])
    overlap = sorted(low_tasks & paired)
    source_partition = provenance["task_family_partition"]
    if set(overlap) != (paired & set(source_partition)):
        raise ValueError("source-run and partition overlap disagree")
    if len(overlap) != 42:
        raise ValueError(f"expected 42 low-source-overlap tasks, got {len(overlap)}")
    qwen_provenance = read_json(Path(provenance["additional_source_provenance"]))
    qwen_overlap = sorted(paired & {run["task_family_id"] for run in qwen_provenance["accepted"]})
    if not set(qwen_overlap) <= set(overlap):
        raise ValueError("Qwen accepted runs include a source-unseen paired task")
    if len(qwen_overlap) != 6:
        raise ValueError(f"expected 6 Qwen-overlap task families, got {len(qwen_overlap)}")
    return {
        "gpt_source_runs": len(source_runs),
        "gpt_source_conditions": sorted([list(value) for value in conditions]),
        "paired_source_low_overlap": overlap,
        "paired_qwen_source_overlap": qwen_overlap,
        "source_partition_counts": {
            name: sum(source_partition[task] == name for task in overlap)
            for name in ("train", "validation", "calibration", "test")
        },
    }


def freeze() -> None:
    declaration_path = OUTPUT / "predeclared.json"
    if declaration_path.exists():
        raise FileExistsError(declaration_path)
    prior_path = ZERO / "predeclared.json"
    provenance_path = SOURCE / "provenance.json"
    checkpoint = SOURCE / "model.pt"
    predictions_path = ZERO / "predictions.jsonl"
    prior = read_json(prior_path)
    provenance = read_json(provenance_path)
    paired = set(prior["all_paired_tasks_descriptive"])
    unseen = set(prior["source_unseen_tasks_primary"])
    if len(paired) != 69 or len(unseen) != 27:
        raise ValueError("paired/source-unseen roster changed")
    if sha256(checkpoint) != prior["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from zero-shot declaration")
    if sha256(provenance_path) != prior["source_provenance_sha256"]:
        raise ValueError("source provenance differs from zero-shot declaration")
    audit = source_audit(provenance, paired)
    exposed = set(audit["paired_source_low_overlap"])
    if paired - exposed != unseen:
        raise ValueError("source-unseen roster disagrees with source-run audit")
    ranked = sorted(exposed, key=lambda task: (hashlib.sha256(
        f"{SPLIT_SALT}:{task}".encode()).hexdigest(), task))
    adaptation = sorted(ranked[:20])
    unused = sorted(ranked[20:])
    if set(adaptation) & unseen or len(adaptation) != 20 or len(unused) != 22:
        raise ValueError("invalid family split")
    declaration = {
        "status": "frozen before this post-calibration fit and score",
        "interpretation": "exploratory retrospective analysis; prior v2 exposed aggregate test labels",
        "prior_predeclared": str(prior_path),
        "prior_predeclared_sha256": sha256(prior_path),
        "zero_shot_predictions": str(predictions_path),
        "zero_shot_predictions_sha256": sha256(predictions_path),
        "source_provenance": str(provenance_path),
        "source_provenance_sha256": sha256(provenance_path),
        "frozen_checkpoint": str(checkpoint),
        "frozen_checkpoint_sha256": sha256(checkpoint),
        "script_sha256": sha256(Path(__file__)),
        "split_rule": f"lowest SHA256('{SPLIT_SALT}:' + task_family_id) among 42 source-exposed pairs",
        "off_adaptation_tasks": adaptation,
        "source_unseen_test_tasks": sorted(unseen),
        "unused_source_exposed_tasks": unused,
        "source_audit": audit,
        "target": "complete logical-call provider input plus output tokens, including attempts",
        "features": "frozen causal_request_surface_history_v3 predictions from zero-shot v2",
        "fit_condition": "off_v21 only; no low-v18 labels used in calibration",
        "fit_rule": "for each call/task scale, median over adaptation tasks of the mean of the two position-wise median log((actual+1)/(prediction+1)) residuals; omit rows with both values zero; multiply each frozen point prediction by exp(offset)",
        "fit_parameters": "two positive multipliers, one shared by call positions and one by task positions; no grid, clipping, or test-label tuning",
        "aggregation": "mean absolute error within task/position, equal mean across test task families",
        "uncertainty": "paired task bootstrap of adapted-minus-zero-shot MAE and adapted/zero-shot macro MAE",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "historical_test_label_exposure": "v2 predictions.jsonl and v2 summary were available before this split; score phase does not feed test labels to fit",
        "pdf_table_15_eligible": False,
        "paid_provider_calls": 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(declaration_path, declaration)
    print(f"frozen 20 off adaptation and 27 source-unseen test tasks: {declaration_path}")


def verified_declaration() -> dict:
    declaration = read_json(OUTPUT / "predeclared.json")
    paths = {
        "script_sha256": Path(__file__),
        "prior_predeclared_sha256": Path(declaration["prior_predeclared"]),
        "zero_shot_predictions_sha256": Path(declaration["zero_shot_predictions"]),
        "source_provenance_sha256": Path(declaration["source_provenance"]),
        "frozen_checkpoint_sha256": Path(declaration["frozen_checkpoint"]),
    }
    for key, path in paths.items():
        if sha256(path) != declaration[key]:
            raise ValueError(f"input changed after freeze: {key}")
    return declaration


def selected_rows(path: Path, tasks: set[str], conditions: set[str]) -> list[dict]:

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = TASK_ID.search(line)
            if match is None:
                raise ValueError("prediction row lacks task_id")
            if match.group(1) not in tasks:
                continue
            row = json.loads(line)
            if row["condition"] in conditions:
                if not row["prediction_available"]:
                    raise ValueError("unavailable forecast in selected cohort")
                rows.append(row)
    return rows


def fit() -> None:
    declaration = verified_declaration()
    output_path = OUTPUT / "calibration.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    adaptation = set(declaration["off_adaptation_tasks"])
    rows = selected_rows(Path(declaration["zero_shot_predictions"]), adaptation, {"off_v21"})
    seen = {row["task_id"] for row in rows}
    if seen != adaptation:
        raise ValueError(f"missing off adaptation tasks: {sorted(adaptation - seen)}")
    by_task_position = defaultdict(list)
    for row in rows:
        if row["point"] not in POSITIONS:
            raise ValueError(f"unexpected position: {row['point']}")
        actual, prediction = float(row["actual"]), float(row["prediction"])
        if not all(math.isfinite(v) and v >= 0 for v in (actual, prediction)):
            raise ValueError("nonfinite or negative prediction/target")
        if actual or prediction:
            by_task_position[(row["task_id"], row["point"])].append(
                math.log1p(actual) - math.log1p(prediction))
    fitted = {}
    for scale, positions in SCALES.items():
        task_residuals = []
        for task in sorted(adaptation):
            medians = []
            for point in positions:
                values = by_task_position[(task, point)]
                if not values:
                    raise ValueError(f"no nonzero adaptation rows: {task} {point}")
                medians.append(float(np.median(values)))
            task_residuals.append(float(np.mean(medians)))
        offset = float(np.median(task_residuals))
        fitted[scale] = {
            "log_offset": offset,
            "multiplier": math.exp(offset),
            "adaptation_tasks": len(task_residuals),
            "per_task_residuals": dict(zip(sorted(adaptation), task_residuals)),
        }
    calibration = {
        "status": "fit from 20 declared off adaptation tasks only",
        "predeclared_sha256": sha256(OUTPUT / "predeclared.json"),
        "source_predictions_sha256": declaration["zero_shot_predictions_sha256"],
        "parsed_conditions": ["off_v21"],
        "parsed_task_ids": sorted(seen),
        "parsed_rows": len(rows),
        "per_position_rows": {point: sum(row["point"] == point for row in rows)
                              for point in POSITIONS},
        "scales": fitted,
        "paid_provider_calls": 0,
    }
    write_json(output_path, calibration)
    print(f"fitted two off-only multipliers: {output_path}")


def summarize(rows: list[dict], factor: float) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tasks = sorted(by_task)
    if len(tasks) != 27:
        raise ValueError(f"test position has {len(tasks)} tasks, expected 27")
    before = np.array([np.mean([abs(row["prediction"] - row["actual"])
                                for row in by_task[task]]) for task in tasks])
    after = np.array([np.mean([abs(factor * row["prediction"] - row["actual"])
                               for row in by_task[task]]) for task in tasks])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    before_draws = before[indices].mean(axis=1)
    after_draws = after[indices].mean(axis=1)
    delta_draws = after_draws - before_draws
    ratio_draws = after_draws / np.maximum(before_draws, 1e-9)
    return {
        "tasks": len(tasks),
        "rows": len(rows),
        "zero_shot_task_macro_mae": float(before.mean()),
        "adapted_task_macro_mae": float(after.mean()),
        "adapted_minus_zero_shot": float(after.mean() - before.mean()),
        "adapted_minus_zero_shot_ci95": np.quantile(delta_draws, [.025, .975]).tolist(),
        "adapted_over_zero_shot": float(after.mean() / before.mean()),
        "adapted_over_zero_shot_ci95": np.quantile(ratio_draws, [.025, .975]).tolist(),
        "per_task_mae": {task: {"zero_shot": float(before[i]), "adapted": float(after[i])}
                         for i, task in enumerate(tasks)},
    }


def score() -> None:
    declaration = verified_declaration()
    calibration_path = OUTPUT / "calibration.json"
    calibration = read_json(calibration_path)
    if calibration["predeclared_sha256"] != sha256(OUTPUT / "predeclared.json"):
        raise ValueError("calibration refers to another declaration")
    if calibration["parsed_task_ids"] != declaration["off_adaptation_tasks"]:
        raise ValueError("calibration used tasks outside the adaptation partition")
    if calibration["parsed_conditions"] != ["off_v21"]:
        raise ValueError("calibration used another condition")
    test = set(declaration["source_unseen_test_tasks"])
    if test & set(calibration["parsed_task_ids"]):
        raise ValueError("adaptation/test task-family leakage")
    source_partition = read_json(Path(declaration["source_provenance"]))["task_family_partition"]
    if test & set(source_partition):
        raise ValueError("source checkpoint has seen a test task family")
    rows = selected_rows(Path(declaration["zero_shot_predictions"]), test,
                         {"low_v18", "off_v21"})
    if len({(row["condition"], row["point_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate test forecast point ID")
    factors = {
        position: calibration["scales"][scale]["multiplier"]
        for scale, positions in SCALES.items() for position in positions
    }
    results = {}
    for point in POSITIONS:
        off = [row for row in rows if row["condition"] == "off_v21" and row["point"] == point]
        low = [row for row in rows if row["condition"] == "low_v18" and row["point"] == point]
        results[point] = summarize(off, factors[point])
        low_by_task = defaultdict(list)
        for row in low:
            low_by_task[row["task_id"]].append(abs(row["prediction"] - row["actual"]))
        if set(low_by_task) != test:
            raise ValueError(f"low condition missing test tasks at {point}")
        results[point]["low_zero_shot_task_macro_mae"] = float(np.mean(
            [np.mean(low_by_task[task]) for task in sorted(test)]))
        results[point]["low_rows"] = len(low)
        results[point]["off_multiplier"] = factors[point]
    report = {
        "status": "exploratory retrospective F.5 off-only post-calibration diagnostic",
        "predeclared_sha256": sha256(OUTPUT / "predeclared.json"),
        "calibration_sha256": sha256(calibration_path),
        "test_task_count": len(test),
        "adaptation_task_count": len(declaration["off_adaptation_tasks"]),
        "test_rows": len(rows),
        "positions": results,
        "leakage_audit": {
            "adaptation_test_intersection": sorted(test & set(calibration["parsed_task_ids"])),
            "test_source_partition_intersection": sorted(test & set(source_partition)),
            "fitted_conditions": calibration["parsed_conditions"],
            "prior_test_label_exposure": declaration["historical_test_label_exposure"],
            "off_test_labels_used_in_fit": False,
        },
        "pdf_table_15_eligible": False,
        "paid_provider_calls": 0,
    }
    write_json(OUTPUT / "report.json", report)
    lines = [
        "# F.5 off-condition post-calibration: exploratory diagnostic",
        "",
        "The checkpoint and causal full-call forecasts are frozen. Twenty hash-selected",
        "source-exposed task families provide off-condition calibration labels. All 27",
        "source-unseen paired families are task-disjoint tests. The two correction factors",
        "share information within call and task positions, respectively.",
        "",
        "| Position | Test tasks | Low points | Off points | Low zero-shot MAE | Off zero-shot MAE | Off adapted MAE | Adapted−zero [95% CI] | Adapted/zero [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for point in POSITIONS:
        result = results[point]
        ci = result["adapted_minus_zero_shot_ci95"]
        ratio_ci = result["adapted_over_zero_shot_ci95"]
        lines.append(
            f"| {point} | {result['tasks']} | {result['low_rows']} | {result['rows']} | "
            f"{result['low_zero_shot_task_macro_mae']:,.1f} | "
            f"{result['zero_shot_task_macro_mae']:,.1f} | "
            f"{result['adapted_task_macro_mae']:,.1f} | "
            f"{result['adapted_minus_zero_shot']:+,.1f} "
            f"[{ci[0]:+,.1f}, {ci[1]:+,.1f}] | "
            f"{result['adapted_over_zero_shot']:.3f} "
            f"[{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}] |"
        )
    lines += [
        "",
        f"Fitted multipliers: call {calibration['scales']['call']['multiplier']:.4f}; "
        f"task {calibration['scales']['task']['multiplier']:.4f}.",
        "",
        "The prior v2 zero-shot report exposed aggregate off test outcomes before this",
        "split. No off test label enters the fit phase, but this is not a fresh blind",
        "holdout. Only off post-calibration is measured; no off-trained shared model",
        "or same-target Self-Prediction is available. Do not fill PDF Table 15 from this run.",
        "Paid provider calls: 0; additional API cost: CNY 0.",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scored 27 source-unseen test tasks: {OUTPUT / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score"))
    args = parser.parse_args()
    {"freeze": freeze, "fit": fit, "score": score}[args.phase]()


if __name__ == "__main__":
    main()
