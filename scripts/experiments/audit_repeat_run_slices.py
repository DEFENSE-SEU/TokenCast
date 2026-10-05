from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _stats(items: list[dict[str, Any]]) -> dict[str, float | int]:
    values = sorted(float(item["value"]) for item in items)
    return {
        "n": len(values),
        "tasks": len({item["task_id"] for item in items}),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)],
    }


def _call_bucket(call_count: int) -> str:
    if call_count <= 15:
        return "1-15"
    if call_count <= 20:
        return "16-20"
    if call_count <= 25:
        return "21-25"
    return "26+"


def _load_rows(run_dir: Path, selected_tasks: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    task = run["task"]
    task_id = task["task_id"]
    if task_id not in selected_tasks:
        return None
    if state.get("capture_status") not in {"complete", "validated"}:
        return None
    if state.get("terminal_status") != "completed":
        return None

    capture_root = run_dir / state["capture_path"]
    capture = json.loads((capture_root / "capture.json").read_text(encoding="utf-8"))
    ends: dict[str, int] = {}
    checkpoints: list[tuple[str, int]] = []
    with (capture_root / "observer.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("record_type") == "attempt/end":
                visible = event.get("observer_visible_output_tokens")
                if isinstance(visible, int):
                    ends[event["attempt_id"]] = visible
            elif event.get("record_type") == "attempt/checkpoint":
                committed = event.get("observer_committed_tokens")
                if isinstance(committed, int):
                    checkpoints.append((event["attempt_id"], committed))

    totals: list[float] = []
    outputs: list[float] = []
    for call in state["calls"]:
        total = 0.0
        output = 0.0
        for attempt in call["attempts"]:
            usage = attempt["usage"]
            total += float(usage["total_tokens"])
            output += float(usage["output_tokens"])
        totals.append(total)
        outputs.append(output)

    task_text = capture["task"].get("task_text", "")
    metadata = {
        "task_id": task_id,
        "run_id": state["run_id"],
        "call_count": len(state["calls"]),
        "task_length_bytes": len(task_text.encode("utf-8")),
        "difficulty": task["difficulty_stratum"],
        "actor": run["condition"]["actor_model_id"],
    }
    rows = [
        {
            **metadata,
            "position": "task_launch",
            "target": "task_total_accounted_tokens",
            "value": sum(totals),
        }
    ]
    rows.extend(
        {
            **metadata,
            "position": "call_pre",
            "target": "call_billable_output_tokens",
            "value": value,
        }
        for value in outputs
    )
    rows.extend(
        {
            **metadata,
            "position": "task_update",
            "target": "task_provider_accounted_remaining_tokens",
            "value": sum(totals[index:]),
        }
        for index in range(1, len(totals))
    )
    rows.extend(
        {
            **metadata,
            "position": "call_update",
            "target": "attempt_observer_remaining_visible_output_tokens",
            "value": float(ends[attempt_id] - committed),
        }
        for attempt_id, committed in checkpoints
        if attempt_id in ends
    )
    return metadata, rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Repeat-run slice audit",
        "",
        "This audit reads retained DSH state and observer records for completed historical captures.",
        "It reports descriptive slices and leaves interval-by-slice scoring to a future raw-forecast export.",
        "",
        f"- source run directories: {report['cohort']['source_run_directories']}",
        f"- selected Task families: {report['cohort']['selected_task_families']}",
        f"- completed captures used here: {report['cohort']['completed_captures']}",
        f"- prior stability summary labels: {report['cohort']['prior_summary_labels']}",
        f"- actor counts: `{report['cohort']['actor_counts']}`",
        f"- difficulty counts: `{report['cohort']['difficulty_counts']}`",
        "",
        "## Position and target distributions",
        "",
        "| position | target | n | tasks | mean | median | p90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in report["position_target"].items():
        position, target = key.split("|", 1)
        lines.append(
            f"| {position} | {target} | {values['n']} | {values['tasks']} | "
            f"{values['mean']:.1f} | {values['median']:.1f} | {values['p90']:.1f} |"
        )
    for dimension, groups in report["task_launch_slices"].items():
        lines.extend(
            [
                "",
                f"## Task-launch slices by {dimension}",
                "",
                "| slice | n | tasks | mean | median | p90 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in groups.items():
            lines.append(
                f"| {name} | {values['n']} | {values['tasks']} | {values['mean']:.1f} | "
                f"{values['median']:.1f} | {values['p90']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Call count separates the long-tail Task totals: the 26+ bucket has a "
            f"mean of {report['task_launch_slices']['call_bucket']['26+']['mean']:.0f} Tokens. ",
            "Task length quartiles have overlapping distributions. Difficulty labels "
            "provide three populated strata. Actor identity is constant in this cohort. ",
            "The retained `rr-v19` summary contains the full interval candidate set; raw "
            "forecast rows and paired uncertainty are required for interval scores within these slices.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    runs_root, stability_path, output_dir = map(Path, sys.argv[1:])
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    selected_tasks = set(stability["dispersion"]["per_task"])
    entries: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        loaded = _load_rows(run_dir, selected_tasks)
        if loaded is not None:
            entries.append(loaded)

    lengths = {}
    for metadata, _rows in entries:
        lengths.setdefault(metadata["task_id"], metadata["task_length_bytes"])
    ordered_lengths = sorted(lengths.values())
    for metadata, rows in entries:
        rank = sum(value <= lengths[metadata["task_id"]] for value in ordered_lengths)
        metadata["length_bucket"] = f"q{max(1, min(4, math.ceil(4 * rank / len(ordered_lengths))))}"
        metadata["call_bucket"] = _call_bucket(metadata["call_count"])
        for row in rows:
            row.update(
                length_bucket=metadata["length_bucket"],
                call_bucket=metadata["call_bucket"],
            )

    position_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    launch_slices: dict[str, dict[str, list[dict[str, Any]]]] = {
        "call_bucket": defaultdict(list),
        "length_bucket": defaultdict(list),
        "difficulty": defaultdict(list),
        "actor": defaultdict(list),
    }
    for metadata, rows in entries:
        for row in rows:
            position_target[f"{row['position']}|{row['target']}"].append(row)
            if row["position"] == "task_launch":
                for dimension in launch_slices:
                    launch_slices[dimension][row[dimension]].append(row)

    report = {
        "schema_version": 1,
        "cohort": {
            "source_run_directories": sum(1 for path in runs_root.iterdir() if path.is_dir()),
            "selected_task_families": len(selected_tasks),
            "completed_captures": len(entries),
            "prior_summary_labels": stability["dispersion"]["n_runs"],
            "actor_counts": dict(Counter(metadata["actor"] for metadata, _ in entries)),
            "difficulty_counts": dict(Counter(metadata["difficulty"] for metadata, _ in entries)),
        },
        "position_target": {key: _stats(value) for key, value in sorted(position_target.items())},
        "task_launch_slices": {
            dimension: {
                key: _stats(value) for key, value in sorted(groups.items())
            }
            for dimension, groups in launch_slices.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slice-audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report["cohort"], ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: ... <runs/runs> <stability.json> <output-dir>")
    main()
