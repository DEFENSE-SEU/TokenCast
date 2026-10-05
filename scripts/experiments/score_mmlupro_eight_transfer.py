from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.experiments.freeze_mmlupro_eight_transfer import (
    ANSWER_AUDIT, EXPECTED_IDS, MANIFEST, MODEL, OUTPUT, POSITIONS, ROOT, sha256,
)
from token_prediction.case_study.segments import load_run, query_from_row
from token_prediction.estimators.composable_segment import ComposableSegmentModel


SAMPLES = 128
BOOTSTRAP_DRAWS = 10_000
SEED = 20260923


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_freeze() -> dict:
    frozen = read_json(OUTPUT / "predeclared.json")
    if (frozen["status"] != "frozen_before_eight_paid_task_calls"
            or tuple(frozen["task_order"]) != EXPECTED_IDS
            or tuple(frozen["prediction_policy"]["positions"]) != POSITIONS):
        raise ValueError("predeclaration roster or policy differs")
    for name, item in frozen["paths_sha256"].items():
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}: {path}")
    lock_path = OUTPUT / "implementation_lock.json"
    if lock_path.is_file():
        lock = read_json(lock_path)
        if (lock["predeclared_sha256"] != sha256(OUTPUT / "predeclared.json")
                or lock["score_script_sha256"] != sha256(Path(__file__))):
            raise ValueError("scoring implementation changed after its hash lock")
    manifest = read_json(MANIFEST)
    for task in manifest["tasks"]:
        evidence = frozen["packs"][task["task_id"]]
        pack = ROOT / evidence["pack_path"]
        for filename, key in (("pack.json", "pack_json_sha256"),
                              ("task.md", "task_prompt_sha256")):
            if sha256(pack / filename) != evidence[key]:
                raise ValueError(f"frozen pack changed: {task['task_id']} {filename}")
    return frozen


def run_inventory(runs_root: Path, frozen: dict) -> dict[str, Path]:
    runs_dir = runs_root / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(runs_dir)
    found = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "run.json").is_file():
            continue
        run = read_json(run_dir / "run.json")
        task = run["task"]["task_id"]
        if task not in EXPECTED_IDS or task in found:
            raise ValueError(f"unexpected or duplicate task in pilot runs: {task}")
        if (run["manifest_id"] != frozen["manifest_id"]
                or run["condition"]["condition_id"] != frozen["condition_id"]
                or run["repeat_id"] != frozen["repeat_id"]
                or run["cap_profile"]["max_logical_calls"] != 12
                or run["condition"]["reasoning_effort"] != "low"
                or run["condition"]["configured_model"] != "openai/gpt-5.4"):
            raise ValueError(f"run protocol differs: {run_dir}")
        found[task] = run_dir
    return found


def observed_cutoffs(batch: dict) -> np.ndarray:


    return np.flatnonzero(batch["whole_suffix"] & (batch["state"][:, 7] < .5))


def source_hashes(run_dir: Path, state: dict) -> dict[str, str]:
    files = [run_dir / "run.json", run_dir / "state.json"]
    if state.get("capture_path"):
        capture = (run_dir / state["capture_path"]).resolve()
        if not capture.is_relative_to(run_dir.resolve()):
            raise ValueError(f"capture escapes run directory: {run_dir}")
        files.extend(path for path in capture.rglob("*") if path.is_file())
    for launch in state.get("launches", []):
        if not launch.get("workspace_path"):
            continue
        workspace = (run_dir / launch["workspace_path"]).resolve()
        if not workspace.is_relative_to(run_dir.resolve()):
            raise ValueError(f"actor workspace escapes run directory: {run_dir}")
        files.extend(path for name in ("answer.md", "submission.patch")
                     if (path := workspace / name).is_file())
    return {str(path.relative_to(run_dir)).replace("\\", "/"): sha256(path)
            for path in sorted(set(files))}


def point_seed(point_id: str) -> int:
    return SEED + int(hashlib.sha256(point_id.encode("utf-8")).hexdigest()[:8], 16)


def predict(runs_root: Path) -> None:
    frozen = check_freeze()
    if (OUTPUT / "predictions.jsonl").exists() or (OUTPUT / "prediction_receipt.json").exists():
        raise FileExistsError("prediction export already exists; preserve its original hash")
    found = run_inventory(runs_root, frozen)
    model = ComposableSegmentModel.load(MODEL)
    records = []
    inventory = {}
    hashes = {}
    for task in EXPECTED_IDS:
        run_dir = found.get(task)
        if run_dir is None:
            inventory[task] = {"status": "not_started", "prediction_rows": 0}
            continue
        state = read_json(run_dir / "state.json")
        hashes[task] = source_hashes(run_dir, state)
        complete = (state.get("terminal_status") == "completed"
                    and state.get("supervision_eligible") is True
                    and state.get("capture_status") in {"complete", "validated"})
        if not complete:
            inventory[task] = {
                "status": "censored_or_ineligible", "terminal_status": state.get("terminal_status"),
                "capture_status": state.get("capture_status"),
                "supervision_eligible": state.get("supervision_eligible"),
                "prediction_rows": 0,
            }
            continue
        batch = load_run(run_dir, request_surface_features=True,
                         history_cost_features=True, checkpoint_policy="first_per_call")
        if set(batch["request_feature_version"]) != {"causal_request_surface_v2"}:
            raise ValueError(f"request surface contract differs: {task}")
        indices = observed_cutoffs(batch)
        counts = Counter()
        for i in indices:
            position = str(batch["position"][i])
            if position not in POSITIONS:
                raise ValueError(f"unexpected prediction point: {position}")
            point_id = str(batch["point_id"][i])
            forecast = model.forecast(query_from_row(batch, int(i)),
                                      samples=SAMPLES, seed=point_seed(point_id))
            values = forecast["call_samples" if position in ("call_start", "in_call_update")
                              else "task_remaining_samples"][0]
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite forecast: {task} {point_id}")
            horizon_exhausted = int(np.count_nonzero(forecast["horizon_exhausted"][0]))
            available = position in ("call_start", "in_call_update") or horizon_exhausted == 0
            lo, median, hi = np.quantile(values, [.05, .5, .95])
            records.append({
                "task_id": task, "run_id": str(batch["run_id"][i]),
                "point_id": point_id, "position": position,
                "cutoff_event_seq": int(batch["cutoff_event_seq"][i]),
                "completed_calls": int(batch["start_call"][i]),
                "prediction_available": bool(available),
                "horizon_exhausted_samples": horizon_exhausted,
                "prediction": float(median), "lower": float(lo), "upper": float(hi),
                "history_median": float(frozen["baseline_constants"][position]),
            })
            counts[position] += 1
        inventory[task] = {"status": "predicted", "prediction_rows": len(indices),
                           "points_by_position": dict(counts)}
        print(f"predicted {task}: {len(indices)} nonterminal points", flush=True)
    if len({row["point_id"] for row in records}) != len(records):
        raise ValueError("duplicate point_id in prediction export")
    path = OUTPUT / "predictions.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "status": "label_free_predictions_frozen_before_score",
        "predeclared_sha256": sha256(OUTPUT / "predeclared.json"),
        "predictions_sha256": sha256(path),
        "runs_root": str(runs_root.resolve()),
        "run_source_hashes": hashes,
        "roster_status": inventory,
        "rows": len(records),
        "predictor_script_sha256": sha256(Path(__file__)),
        "warning": "capture parsing reads usage metadata; forecast queries and this export exclude future usage labels",
    }
    with (OUTPUT / "prediction_receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"predictions": str(path), "sha256": receipt["predictions_sha256"],
                      "rows": len(records)}, ensure_ascii=False), flush=True)


def key_map() -> dict[str, str]:
    answers = defaultdict(set)
    with ANSWER_AUDIT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("benchmark") == "mmlupro" and row.get("task_id") in EXPECTED_IDS:
                answers[row["task_id"]].add(row["reference_answer"])
    if set(answers) != set(EXPECTED_IDS) or any(len(values) != 1 for values in answers.values()):
        raise ValueError("prior original-answer audit lacks unique keys for eight tasks")
    return {task: next(iter(values)) for task, values in answers.items()}


def answer_letter(text: str) -> str | None:
    if not text:
        return None
    first = text.splitlines()[0].strip()
    return first if re.fullmatch("[A-J]", first) else None


def patch_letter(text: str) -> str | None:
    added = [line[1:].strip() for line in text.splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    return added[0] if added and re.fullmatch("[A-J]", added[0]) else None


def correctness(run_dir: Path, state: dict, reference: str) -> dict:
    if state.get("terminal_status") != "completed":
        return {"status": "censored", "correct": None}
    launches = state.get("launches", [])
    if not launches or not launches[-1].get("workspace_path"):
        return {"status": "missing_workspace", "correct": False}
    workspace = (run_dir / launches[-1]["workspace_path"]).resolve()
    if not workspace.is_relative_to(run_dir.resolve()):
        raise ValueError(f"actor workspace escapes run directory: {run_dir}")
    answer_path, patch_path = workspace / "answer.md", workspace / "submission.patch"
    answer = answer_letter(answer_path.read_text(encoding="utf-8") if answer_path.is_file() else "")
    patch = patch_letter(patch_path.read_text(encoding="utf-8") if patch_path.is_file() else "")
    status = "strict_scorable" if answer and answer == patch else "invalid_answer_or_patch"
    return {"status": status, "answer": answer, "patch_answer": patch,
            "correct": answer == reference if status == "strict_scorable" else False}


def usage_summary(state: dict) -> dict:
    attempts = [attempt for call in state.get("calls", []) for attempt in call.get("attempts", [])]
    complete = [attempt["usage"] for attempt in attempts
                if all(isinstance((attempt.get("usage") or {}).get(key), int)
                       for key in ("input_tokens", "output_tokens"))]
    return {"logical_calls": len(state.get("calls", [])), "provider_attempts": len(attempts),
            "attempts_missing_usage": len(attempts) - len(complete),
            "observed_input_tokens": sum(item["input_tokens"] for item in complete),
            "observed_output_tokens": sum(item["output_tokens"] for item in complete),
            "observed_total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in complete)}


def paired_summary(by_task: dict[str, list[dict]]) -> dict:
    tasks = sorted(by_task)
    if not tasks:
        return {"paired_tasks": 0, "model_mae": None, "history_median_mae": None,
                "model_minus_median": None, "difference_ci95": None, "ratio_ci95": None}
    ours = np.array([np.mean([row["model_absolute_error"] for row in by_task[task]])
                     for task in tasks])
    baseline = np.array([np.mean([row["history_median_absolute_error"] for row in by_task[task]])
                         for task in tasks])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    diffs = ours[indices].mean(axis=1) - baseline[indices].mean(axis=1)
    ratios = ours[indices].mean(axis=1) / np.maximum(baseline[indices].mean(axis=1), 1e-9)
    return {"paired_tasks": len(tasks), "paired_task_ids": tasks,
            "model_mae": float(ours.mean()), "history_median_mae": float(baseline.mean()),
            "model_minus_median": float(ours.mean() - baseline.mean()),
            "difference_ci95": np.quantile(diffs, [.025, .975]).tolist(),
            "model_over_median": float(ours.mean() / max(baseline.mean(), 1e-9)),
            "ratio_ci95": np.quantile(ratios, [.025, .975]).tolist(),
            "per_task": {task: {"model_mae": float(ours[i]),
                                "history_median_mae": float(baseline[i]),
                                "points": len(by_task[task])}
                         for i, task in enumerate(tasks)}}


def score(runs_root: Path) -> None:
    frozen = check_freeze()
    receipt = read_json(OUTPUT / "prediction_receipt.json")
    prediction_path = OUTPUT / "predictions.jsonl"
    if (receipt["predeclared_sha256"] != sha256(OUTPUT / "predeclared.json")
            or receipt["predictions_sha256"] != sha256(prediction_path)
            or receipt["predictor_script_sha256"] != sha256(Path(__file__))
            or receipt["runs_root"] != str(runs_root.resolve())):
        raise ValueError("predeclaration, predictor or label-free prediction export changed")
    found = run_inventory(runs_root, frozen)
    if set(found) != {task for task, row in receipt["roster_status"].items()
                      if row["status"] != "not_started"}:
        raise ValueError("run roster changed after prediction export")
    predictions = [json.loads(line) for line in prediction_path.open(encoding="utf-8")]
    if len(predictions) != receipt["rows"] or len({row["point_id"] for row in predictions}) != len(predictions):
        raise ValueError("prediction export rows changed")
    by_task_prediction = defaultdict(dict)
    for row in predictions:
        by_task_prediction[row["task_id"]][row["point_id"]] = row
    keys = key_map()
    task_reports = {}
    scored_rows = []
    for task in EXPECTED_IDS:
        run_dir = found.get(task)
        if run_dir is None:
            task_reports[task] = {"status": "not_started", "correctness": {"correct": None}}
            continue
        state = read_json(run_dir / "state.json")
        if source_hashes(run_dir, state) != receipt["run_source_hashes"][task]:
            raise ValueError(f"capture or actor answer changed after prediction export: {task}")
        complete = receipt["roster_status"][task]["status"] == "predicted"
        task_reports[task] = {
            "status": "eligible_completed" if complete else "censored_or_ineligible",
            "terminal_status": state.get("terminal_status"),
            "supervision_eligible": state.get("supervision_eligible"),
            "capture_status": state.get("capture_status"),
            **usage_summary(state),
            "correctness": correctness(run_dir, state, keys[task]),
        }
        if not complete:
            if by_task_prediction[task]:
                raise ValueError(f"censored task has prediction rows: {task}")
            continue
        batch = load_run(run_dir, request_surface_features=True,
                         history_cost_features=True, checkpoint_policy="first_per_call")
        indices = observed_cutoffs(batch)
        expected_points = {str(batch["point_id"][i]) for i in indices}
        if expected_points != set(by_task_prediction[task]):
            raise ValueError(f"prediction point roster differs from capture: {task}")
        for i in indices:
            row = by_task_prediction[task][str(batch["point_id"][i])]
            if row["position"] != str(batch["position"][i]) or row["cutoff_event_seq"] != int(batch["cutoff_event_seq"][i]):
                raise ValueError(f"point metadata differs: {task}")
            if int(batch["n"][i]) <= 0:
                raise ValueError(f"terminal zero row survived causal filter: {task}")
            actual = float(batch["label_call_cost" if row["position"] in ("call_start", "in_call_update")
                                 else "label_cost"][i])
            if actual < 0:
                raise ValueError(f"negative target: {task}")
            scored_rows.append({**row, "actual": actual,
                                "model_absolute_error": abs(row["prediction"] - actual)
                                if row["prediction_available"] else None,
                                "history_median_absolute_error": abs(row["history_median"] - actual)})
    positions = {}
    for position in POSITIONS:
        rows = [row for row in scored_rows if row["position"] == position]
        paired = [row for row in rows if row["prediction_available"]]
        grouped = defaultdict(list)
        for row in paired:
            grouped[row["task_id"]].append(row)
        positions[position] = {"observed_points": len(rows), "available_points": len(paired),
                               "unavailable_points": len(rows) - len(paired),
                               "observed_tasks": len({row["task_id"] for row in rows}),
                               **paired_summary(grouped)}
    completed = [row for row in task_reports.values() if row.get("terminal_status") == "completed"]
    strict = [row for row in completed if row["correctness"]["status"] == "strict_scorable"]
    correct = sum(row["correctness"]["correct"] is True for row in task_reports.values())
    report = {
        "status": "scored_frozen_eight_task_descriptive_pilot",
        "predeclared_sha256": receipt["predeclared_sha256"],
        "predictions_sha256": receipt["predictions_sha256"],
        "planned_tasks": list(EXPECTED_IDS), "planned_task_count": 8,
        "eligible_completed_task_count": sum(row["status"] == "eligible_completed"
                                            for row in task_reports.values()),
        "task_reports": task_reports, "positions": positions,
        "correctness": {"correct": correct, "completed_denominator": len(completed),
                        "strict_scorable_denominator": len(strict), "planned_denominator": 8,
                        "invalid_completed_answers": len(completed) - len(strict)},
        "interpretation": "same-task cross-model development pilot; source overlap and budget censoring limit generalization; no Table 7/F.1 cell",
    }
    with (OUTPUT / "scored_rows.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUTPUT / "report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = ["# Frozen eight-task GPT-5.4-low MMLU-Pro transfer pilot", "",
             f"Planned 8; eligible completed {report['eligible_completed_task_count']}; "
             f"strict correct {correct}/{len(completed)} completed, {correct}/8 planned.", "",
             "| Position | Observed tasks | Available points | Model MAE | History median MAE | Model−median [task bootstrap 95% CI] |",
             "|---|---:|---:|---:|---:|---:|"]
    for point, row in positions.items():
        diff = (f"{row['model_minus_median']:+,.0f} [{row['difference_ci95'][0]:+,.0f}, "
                f"{row['difference_ci95'][1]:+,.0f}]") if row["paired_tasks"] else "NA"
        ours = f"{row['model_mae']:,.0f}" if row["paired_tasks"] else "NA"
        median = f"{row['history_median_mae']:,.0f}" if row["paired_tasks"] else "NA"
        lines.append(f"| {point} | {row['observed_tasks']} | {row['available_points']} | {ours} | {median} | {diff} |")
    lines += ["", "All eight task statuses and raw rows are retained in report.json and scored_rows.jsonl.",
              "The historical predictor contains four selected task families in training and three in validation.",
              "This small same-task cross-model pilot does not fill the manuscript's Table 7 or F.1 Table 8."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT / "report.json"),
                      "eligible_completed_tasks": report["eligible_completed_task_count"],
                      "correct": correct}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "score"))
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.phase == "predict" else score)(args.runs_root.resolve())


if __name__ == "__main__":
    main()
