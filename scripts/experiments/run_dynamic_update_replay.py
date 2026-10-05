from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from scripts.experiments.build_main_table import paired_bootstrap
from token_prediction.experiments.prediction_export import iter_prediction_export

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/dynamic_update_evaluation_v1.json"
ARMS = {
    "in_call_visible": ("in_call_ours", "prefix_length_bucket", "empirical"),
    "in_call_billed": ("in_call_ours_clock", "prefix_length_bucket", "empirical"),
    "task_update": ("task_update_ours", "call_index_bucket", "call_clock_compound"),
    "in_call_update": ("ratio_grid_cs", "ratio_grid_cs_fresh", "in_call_cs"),
}
FIELDS = (
    "point_id", "source_event_id", "task_id", "run_id", "trajectory_id", "attempt_id",
    "prediction_context_id", "cutoff_event_seq", "target_value", "sample_weight",
    "prediction", "lower", "upper", "progress_known_amount", "known_offset_tokens",
    "token_unit_contract_id", "fold", "target", "position", "condition_id",
)


def boundary(row: dict) -> tuple:
    return row["run_id"], row["attempt_id"], row["cutoff_event_seq"]


def load_panel(fit_dir: Path, kind: str, development_tasks: set[str]) -> dict:
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for row in iter_prediction_export(fit_dir):
        name = row["experiment_id"].split("--")[0]
        if name not in ARMS or row["candidate_id"] not in ARMS[name]:
            continue
        if row["label_status"] != "observed":
            continue
        if row["task_id"] not in development_tasks:
            raise ValueError("prediction Task is outside the frozen development authority")
        selected = {key: row[key] for key in FIELDS}
        points = cells[name][row["candidate_id"]]
        if row["point_id"] in points:
            raise ValueError("duplicate target point in panel")
        points[row["point_id"]] = selected
    expected = {"in_call_update"} if kind == "recurrent" else set(ARMS) - {"in_call_update"}
    if set(cells) != expected:
        raise ValueError(f"incomplete target panel: {fit_dir}")
    clocks = {}
    if kind == "independent":
        for row in cells["in_call_visible"][ARMS["in_call_visible"][0]].values():
            clocks[boundary(row)] = row["known_offset_tokens"]
    for name, candidates in cells.items():
        if set(candidates) != set(ARMS[name]):
            raise ValueError(f"incomplete candidate panel: {name}")
        base = candidates[ARMS[name][0]]
        for points in candidates.values():
            if points.keys() != base.keys():
                raise ValueError("paired candidates must have the same point cohort")
            for pid, row in points.items():
                for key in set(FIELDS) - {"prediction", "lower", "upper"}:
                    if row[key] != base[pid][key]:
                        raise ValueError(f"paired candidate metadata differs: {key}")
                if name == "task_update":

                    row["clock"] = int(row["prediction_context_id"].rsplit(":call:", 1)[1])
                elif name == "in_call_billed":
                    row["clock"] = clocks[boundary(row)]
                else:
                    row["clock"] = row["known_offset_tokens"]
    return dict(cells)


def replay(rows: list[dict], name: str, cadence: int | None) -> list[dict]:


    streams: dict = defaultdict(list)
    for row in rows:
        identity = (row["run_id"], row["trajectory_id"])
        if name != "task_update":
            identity += (row["attempt_id"],)
        streams[identity].append(row)
    output = []
    for sequence in streams.values():
        last = None
        previous_clock = -1
        for row in sorted(sequence, key=lambda r: r["cutoff_event_seq"]):
            if row["clock"] < previous_clock:
                raise ValueError("observation clock decreased within a stream")
            previous_clock = row["clock"]
            # Refresh decisions use committed progress at the current checkpoint.
            refresh = last is None or (
                cadence is not None and row["clock"] - last["clock"] >= cadence
            )
            if refresh:
                last = row
            deduction = 0
            if name in {"in_call_visible", "in_call_update"}:
                if row["token_unit_contract_id"] != "observer:utf8-bytes-v1":
                    raise ValueError("byte deduction requires the byte target contract")
                deduction = row["known_offset_tokens"] - last["known_offset_tokens"]
            elif name == "task_update":
                deduction = row["progress_known_amount"] - last["progress_known_amount"]
            if deduction < 0:
                raise ValueError("accounted progress decreased within a stream")
            output.append({
                **row,
                **{key: max(0.0, last[key] - deduction) for key in ("prediction", "lower", "upper")},
                "refreshed": int(refresh),
            })
    return output


def bin_label(clock: int, edges: list[int]) -> str:
    lower = max(edge for edge in edges if edge <= clock)
    index = edges.index(lower)
    return f"{lower}-{edges[index + 1] - 1}" if index + 1 < len(edges) else f"{lower}+"


def aggregate(rows: list[dict], edges: list[int], alpha: float) -> dict:
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in rows:
        y, p, lo, hi = (row[key] for key in ("target_value", "prediction", "lower", "upper"))
        w = row["sample_weight"]
        values = {
            "mae": abs(y - p), "coverage": float(lo <= y <= hi),
            "normalized_width": (hi - lo) / max(y, 1),
            "interval_score": hi - lo + 2 / alpha * (max(lo - y, 0) + max(y - hi, 0)),
            "target_mean": y, "refresh_fraction": row.get("refreshed", 1),
        }
        for label in ("all", bin_label(row["clock"], edges)):
            cell = acc[label][row["task_id"]]
            cell["weight"] += w
            cell["n_points"] += 1
            cell["n_refreshes"] += row.get("refreshed", 1)
            for metric, value in values.items():
                cell[metric] += w * value
    return {label: {task: {
        key: value if key in {"weight", "n_points", "n_refreshes"} else value / cell["weight"]
        for key, value in cell.items()
    } for task, cell in tasks.items()} for label, tasks in acc.items()}


def evaluate_seed(fit_dir: Path, kind: str, protocol: dict, development_tasks: set[str]) -> dict:
    output = {}
    for name, arms in load_panel(fit_dir, kind, development_tasks).items():
        methods = {candidate: list(points.values()) for candidate, points in arms.items()}
        if kind == "independent":
            ours = methods[ARMS[name][0]]
            methods["first_checkpoint_hold"] = replay(ours, name, None)
            cadences = protocol["call_cadences" if name == "task_update" else "byte_cadences"]
            unit = "calls" if name == "task_update" else "bytes"
            for cadence in cadences:
                methods[f"update_{cadence}_{unit}"] = replay(ours, name, cadence)
        edges = protocol["call_bins" if name == "task_update" else "byte_bins"]
        output[name] = {method: aggregate(rows, edges, protocol["alpha"]) for method, rows in methods.items()}
    return output


def combine(seeds: list[dict], protocol: dict) -> dict:
    results = {}
    for name in seeds[0]:
        methods = {}
        for method in seeds[0][name]:
            bins = {}
            for label in seeds[0][name][method]:
                values = [seed[name][method][label] for seed in seeds]
                tasks = set(values[0])
                if any(set(value) != tasks for value in values):
                    raise ValueError("seed panels must have identical Task cohorts")
                averaged = {task: {key: fmean(value[task][key] for value in values)
                                   for key in values[0][task]} for task in sorted(tasks)}
                metrics = {key: fmean(value[key] for value in averaged.values())
                           for key in ("mae", "coverage", "normalized_width", "interval_score", "refresh_fraction")}
                metrics["wape"] = metrics["mae"] / fmean(value["target_mean"] for value in averaged.values())
                metrics.update(n_tasks=len(tasks), n_points=sum(v["n_points"] for v in averaged.values()),
                               n_refreshes=sum(v["n_refreshes"] for v in averaged.values()))
                metrics["raw_refresh_fraction"] = metrics["n_refreshes"] / metrics["n_points"]
                bins[label] = {"metrics": metrics, "tasks": averaged}
            methods[method] = bins
        reference = ARMS[name][0]
        for method, bins in methods.items():
            for label, cell in bins.items():
                cell["paired_vs_full_update"] = paired_bootstrap(
                    {t: v["mae"] for t, v in cell["tasks"].items()},
                    {t: v["mae"] for t, v in methods[reference][label]["tasks"].items()},
                    seed=protocol["bootstrap_seed"],
                )
        results[name] = methods
    return results


def save_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    flat = []
    for panel, targets in report["panels"].items():
        for target, methods in targets.items():
            for method, bins in methods.items():
                for label, cell in bins.items():
                    flat.append({"panel": panel, "target": target, "method": method, "progress_bin": label,
                                 **cell["metrics"], **{f"paired_{k}": v for k, v in cell["paired_vs_full_update"].items()}})
    with (output / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Dynamic update development replay", "",
             "Three retained split seeds; Task-paired 10,000-draw bootstrap. All rows remain in their original folds.",
             "MAE units: visible residual = UTF-8 bytes; billed final and Task residual = provider Tokens.",
             "Cadences trigger at the first retained checkpoint meeting the causal gap. Task curves use the causally visible Call index. Intervals reuse original calibration.",
             "Weights retain the source Task/trajectory/Attempt/progress-bin hierarchy and renormalize within each plotted Task bin.",
             "Full-update recurrence uses the recorded state transitions. Sparse policies apply to independent heads.", "",
             "Refresh percentage is the actual head-call count divided by the full-cadence count. CSV also retains Task-weighted refresh fractions.", "",
             "| panel | target | method | MAE | coverage | refresh % | delta vs full [95% CI] |", "|---|---|---|---:|---:|---:|---|"]
    for row in flat:
        if row["progress_bin"] == "all":
            lines.append(f"| {row['panel']} | {row['target']} | {row['method']} | {row['mae']:.2f} | {row['coverage']:.3f} | {100 * row['raw_refresh_fraction']:.1f} | {row['paired_delta']:.2f} [{row['paired_ci_lower']:.2f}, {row['paired_ci_upper']:.2f}] |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_report(output: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p["grid"] for p in report["protocol"]["panels"] if p["kind"] == "independent"]
    targets = ("in_call_visible", "in_call_billed", "task_update")
    titles = ("Visible residual (bytes)", "Final output (Tokens)", "Task residual (Tokens)")
    figure, axes = plt.subplots(len(panels), 3, figsize=(15, 3.5 * len(panels)), squeeze=False)
    tradeoff, trade_axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for i, panel in enumerate(panels):
        for j, target in enumerate(targets):
            methods = report["panels"][panel][target]
            reference = ARMS[target][0]
            cadence = "update_4_calls" if target == "task_update" else "update_512_bytes"
            labels = sorted((s for s in methods[reference] if s != "all"), key=lambda s: int(s.split("-")[0].rstrip("+")))
            ax = axes[i, j]
            for method, legend in ((reference, "Every checkpoint"), ("first_checkpoint_hold", "First checkpoint hold"),
                                   (cadence, "Every 4 calls" if target == "task_update" else "Every 512 bytes"),
                                   (ARMS[target][1], "Progress-only reference")):
                ax.plot(range(len(labels)), [methods[method][s]["metrics"]["mae"] for s in labels], marker="o", ms=3, label=legend)
            ax.set_xticks(range(len(labels)), [f"{s}\nn={methods[reference][s]['metrics']['n_tasks']}" for s in labels], fontsize=8)
            ax.set_title(f"{panel}: {titles[j]}")
            ax.set_ylabel("Task-weighted MAE")
            ax.set_xlabel("Call index" if target == "task_update" else "Committed UTF-8 bytes")
            ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            base = methods[reference]["all"]["metrics"]["mae"]
            policies = [m for m in methods if m.startswith("update_") or m == "first_checkpoint_hold"]
            points = sorted((100 * methods[m]["all"]["metrics"]["raw_refresh_fraction"],
                             methods[m]["all"]["metrics"]["mae"] / base) for m in policies)
            trade_axes[j].plot([p[0] for p in points], [p[1] for p in points], marker="o", label=panel)
    for ax, title in zip(trade_axes, titles):
        ax.set(title=title, xlabel="Head calls (% of full cadence)", ylabel="MAE / full-cadence MAE")
        ax.axhline(1, color="grey", linewidth=0.6)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    figure.tight_layout()
    for name, fig in (("progress_curves", figure), ("cadence_tradeoff", tradeoff)):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CONFIG)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    authority = json.loads((CONFIG.parents[2] / protocol["partition_authority"]).read_text(encoding="utf-8"))
    development_tasks = {row["task_id"] for row in authority["assignments"] if row["partition"] == "development"}
    report = {"protocol": protocol, "panels": {}, "source_integrity": []}
    for panel in protocol["panels"]:
        seeds = []
        for seed in protocol["split_seeds"]:
            fit = args.runs_root / panel["grid"] / f"split-{seed}" / f"fit-{protocol['fit_seed']}"
            run = json.loads((fit / "run.json").read_text(encoding="utf-8"))
            if run["prediction_record_defaults"]["run"]["source_id"] != "dsh_session_observer_v2":
                raise ValueError(f"source is outside the frozen development authority: {fit}")
            print(f"Evaluating {panel['grid']} seed {seed}", flush=True)
            seeds.append(evaluate_seed(fit, panel["kind"], protocol, development_tasks))
            with (fit / "predictions.jsonl").open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            report["source_integrity"].append({"grid": panel["grid"], "seed": seed, "predictions_sha256": digest,
                                               "historical_evaluation_scope": run["evaluation_scope"],
                                               "current_partition_authority": authority["authority_id"]})
        report["panels"][panel["grid"]] = combine(seeds, protocol)
        save_report(args.output, report)
    plot_report(args.output, report)
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
