from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.repair_composable_length import four_point_scores
from token_prediction.case_study.segments import load_run
from token_prediction.estimators.composable_segment import ComposableSegmentModel


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOTS = {
    "low_v18": ROOT / "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v18/runs/runs",
    "off_v21": ROOT / "workspace/runs/reasoning-off-v21-main/runs",
}
SOURCE = ROOT / "workspace/runs/composable-multidomain-history-v1"
MODEL = SOURCE / "model.pt"
OUTPUT = ROOT / "workspace/runs/appendix-f5-composable-zero-shot-20260923-v2"
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
SAMPLES = 128
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260923


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> dict[str, dict[str, Path]]:
    runs = {}
    for condition, root in RUN_ROOTS.items():
        found = {}
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "run.json").is_file():
                continue
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            if state.get("capture_status") not in {"complete", "validated"} or not state.get("supervision_eligible"):
                raise ValueError(f"ineligible {condition} run: {directory}")
            manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            task = str(manifest["task"]["task_family_id"])
            if task in found:
                raise ValueError(f"duplicate {condition} task: {task}")
            found[task] = directory
        runs[condition] = found
    if len(runs["low_v18"]) != 69 or set(runs["low_v18"]) != set(runs["off_v21"]):
        raise ValueError("expected the same 69 eligible task families in both cohorts")
    return runs


def frozen_manifest(runs: dict[str, dict[str, Path]]) -> dict:
    partition_path = SOURCE / "provenance.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))["task_family_partition"]
    tasks = sorted(runs["low_v18"])
    unseen = sorted(set(tasks) - partition.keys())
    if len(unseen) != 27:
        raise ValueError("the source-unseen paired cohort changed")
    capture_hashes = {}
    for condition, by_task in runs.items():
        capture_hashes[condition] = {}
        for task, directory in sorted(by_task.items()):
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            capture = (directory / state["capture_path"]).resolve()
            if not capture.is_relative_to(directory.resolve()):
                raise ValueError(f"capture path escapes run: {directory}")
            files = [directory / "run.json", directory / "state.json"]
            files.extend(path for path in capture.rglob("*") if path.is_file())
            capture_hashes[condition][task] = {
                str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
                for path in sorted(files)
            }
    return {
        "status": "frozen before transfer scoring",
        "protocol": "shared composable checkpoint, zero-shot reasoning low/off comparison",
        "checkpoint": str(MODEL),
        "checkpoint_sha256": sha256(MODEL),
        "source_provenance": str(partition_path),
        "source_provenance_sha256": sha256(partition_path),
        "scoring_script_sha256": sha256(Path(__file__)),
        "conditions": {key: str(value) for key, value in RUN_ROOTS.items()},
        "all_paired_tasks_descriptive": tasks,
        "source_unseen_tasks_primary": unseen,
        "source_partition_overlap": {
            part: sorted(task for task in tasks if partition.get(task) == part)
            for part in ("train", "validation", "calibration", "test")
        },
        "capture_hashes": capture_hashes,
        "features": "causal_request_surface_history_v3 via load_run(request_surface_features=True, history_cost_features=True)",
        "target": "complete provider-accounted input + output tokens of each logical call, including attempts",
        "positions": POSITIONS,
        "rows": "one full-suffix forecast per observed cutoff; exclude terminal n=0 rows",
        "aggregation": "absolute error averaged within each task, then equal-weighted across paired tasks",
        "missing_points": "pair tasks within each position; disclose any absent or unavailable forecasts",
        "uncertainty": "paired task bootstrap of off-minus-low task MAE and ratio of macro MAEs",
        "samples": SAMPLES,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "no_off_label_fitting_or_selection": True,
        "paid_provider_calls": 0,
        "limitation": "source training includes GPT-5.4 low v19 and Qwen; only 27 paired task families are absent from every source partition; this is not PDF Table 15",
    }


def freeze() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "predeclared.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = frozen_manifest(inventory())
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"frozen {len(manifest['all_paired_tasks_descriptive'])} paired tasks and "
          f"{len(manifest['source_unseen_tasks_primary'])} source-unseen tasks: {path}", flush=True)


def summarize_rows(rows: list[dict], cohort: set[str]) -> dict:
    by_condition = {}
    for condition in RUN_ROOTS:
        by_task = {}
        for row in rows:
            if row["condition"] == condition and row["task_id"] in cohort:
                by_task.setdefault(row["task_id"], []).append(row)
        by_condition[condition] = by_task
    paired = sorted(set.intersection(*(set(group) for group in by_condition.values())))
    if not paired:
        raise ValueError("no paired tasks with a forecast")
    per_task = {
        condition: {
            task: float(np.mean([abs(row["prediction"] - row["actual"])
                                 for row in by_condition[condition][task]]))
            for task in paired
        }
        for condition in RUN_ROOTS
    }
    low = np.array([per_task["low_v18"][task] for task in paired])
    off = np.array([per_task["off_v21"][task] for task in paired])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)))
    low_draws = low[indices].mean(axis=1)
    off_draws = off[indices].mean(axis=1)
    difference = off_draws - low_draws
    ratios = off_draws / np.maximum(low_draws, 1e-9)
    return {
        "tasks": len(paired),
        "paired_task_ids": paired,
        "missing_tasks": sorted(cohort - set(paired)),
        "points": {condition: sum(len(by_condition[condition][task]) for task in paired)
                   for condition in RUN_ROOTS},
        "task_macro_mae": {"low_v18": float(low.mean()), "off_v21": float(off.mean())},
        "off_minus_low": float(off.mean() - low.mean()),
        "off_minus_low_ci95": np.quantile(difference, [.025, .975]).tolist(),
        "off_over_low": float(off.mean() / low.mean()),
        "off_over_low_ci95": np.quantile(ratios, [.025, .975]).tolist(),
        "per_task_mae": per_task,
    }


def score() -> None:
    declaration = json.loads((OUTPUT / "predeclared.json").read_text(encoding="utf-8"))
    if sha256(MODEL) != declaration["checkpoint_sha256"]:
        raise ValueError("shared checkpoint changed after freeze")
    if sha256(Path(__file__)) != declaration["scoring_script_sha256"]:
        raise ValueError("scoring script changed after freeze")
    if sha256(SOURCE / "provenance.json") != declaration["source_provenance_sha256"]:
        raise ValueError("source provenance changed after freeze")
    runs = inventory()
    for condition, by_task in runs.items():
        for task, directory in by_task.items():
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            capture = (directory / state["capture_path"]).resolve()
            paths = [directory / name for name in declaration["capture_hashes"][condition][task]]
            actual_names = {str(path.relative_to(directory)).replace("\\", "/")
                            for path in (directory / "run.json", directory / "state.json", *capture.rglob("*"))
                            if path.is_file()}
            if actual_names != set(declaration["capture_hashes"][condition][task]):
                raise ValueError(f"capture files changed after freeze: {condition} {task}")
            for path in paths:
                if sha256(path) != declaration["capture_hashes"][condition][task][str(path.relative_to(directory)).replace("\\", "/")]:
                    raise ValueError(f"capture content changed after freeze: {condition} {task}")
    model = ComposableSegmentModel.load(MODEL)
    output_rows = []
    for condition, by_task in runs.items():
        for index, (task, directory) in enumerate(sorted(by_task.items()), 1):
            batch = load_run(directory, request_surface_features=True, history_cost_features=True)
            if set(batch["request_feature_version"]) != {"causal_request_surface_v2"}:
                raise ValueError(f"request feature contract changed: {condition} {task}")
            _, predictions = four_point_scores(model, batch, SAMPLES)
            output_rows.extend({**row, "condition": condition} for row in predictions)
            if index % 10 == 0 or index == len(by_task):
                print(f"{condition}: {index}/{len(by_task)} tasks scored", flush=True)
    if len({(row["condition"], row["point_id"]) for row in output_rows}) != len(output_rows):
        raise ValueError("duplicate scored point ID")
    full = set(declaration["all_paired_tasks_descriptive"])
    unseen = set(declaration["source_unseen_tasks_primary"])
    report = {"status": "offline predeclared zero-shot transfer diagnostic",
              "predeclared": str(OUTPUT / "predeclared.json"),
              "predeclared_sha256": sha256(OUTPUT / "predeclared.json"),
              "cohorts": {}, "forecast_rows": len(output_rows),
              "unavailable_rows": {
                  condition: sum(row["condition"] == condition and not row["prediction_available"]
                                 for row in output_rows)
                  for condition in RUN_ROOTS},
              "paid_provider_calls": 0}
    for name, tasks in (("source_unseen_primary", unseen), ("all_paired_descriptive", full)):
        report["cohorts"][name] = {}
        for position in POSITIONS:
            selected = [row for row in output_rows if row["point"] == position
                        and row["prediction_available"]]
            report["cohorts"][name][position] = summarize_rows(selected, tasks)
    with (OUTPUT / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    lines = ["# Shared composable model: GPT-5.4 reasoning low → off",
             "", "Frozen shared checkpoint, no off-label fitting or provider calls. The 27 source-unseen",
             "paired task families are primary; all 69 paired families are descriptive because 42",
             "appear in source training, validation, or calibration. Scores are task-equal MAE of",
             "full logical-call provider tokens, including input and output.", "",
             "| Cohort | Position | Paired tasks | Low points | Off points | Low MAE | Off MAE | Off−low [95% CI] | Off/low [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name in ("source_unseen_primary", "all_paired_descriptive"):
        for position in POSITIONS:
            row = report["cohorts"][name][position]
            a, b = row["task_macro_mae"]["low_v18"], row["task_macro_mae"]["off_v21"]
            ci, ratio_ci = row["off_minus_low_ci95"], row["off_over_low_ci95"]
            lines.append(f"| {name} | {position} | {row['tasks']} | {row['points']['low_v18']} | "
                         f"{row['points']['off_v21']} | {a:,.1f} | {b:,.1f} | "
                         f"{row['off_minus_low']:+,.1f} [{ci[0]:+,.1f}, {ci[1]:+,.1f}] | "
                         f"{row['off_over_low']:.3f} [{ratio_ci[0]:.3f}, {ratio_ci[1]:.3f}] |")
    lines += ["", "This is a condition-transfer diagnostic for a v19-low/Qwen-trained shared model.",
              "It does not supply the PDF Table 15 within-condition, adapted, or Self-Prediction cells.",
              f"Unavailable forecast rows: {report['unavailable_rows']}."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT / 'summary.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else score)()


if __name__ == "__main__":
    main()
