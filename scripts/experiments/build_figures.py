from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments import figure_style as fs
from scripts.experiments.figure_style import (
    COLUMN,
    GRID,
    SEQUENTIAL,
    TEXT,
    WIDE,
    bar_kwargs,
    family_of,
    grid_on,
    legend_bottom,
    line_kwargs,
    panel_title,
    save,
)

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "workspace" / "runs"
POSITIONS = ["task_initialization", "call_start", "in_call_visible", "in_call_billed", "task_update"]
TITLES = {"task_initialization": "Task launch", "call_start": "Call start",
          "in_call_visible": "In-call, visible", "in_call_billed": "In-call, billed",
          "task_update": "Task update"}
SHORT_TITLES = {"task_initialization": "Task\nlaunch", "call_start": "Call\nstart",
                "in_call_visible": "In-call\nvisible", "in_call_billed": "In-call\nbilled",
                "task_update": "Task\nupdate"}
UNITS = {"task_initialization": "run Tokens", "call_start": "output Tokens",
         "in_call_visible": "UTF-8 bytes", "in_call_billed": "output Tokens",
         "task_update": "run Tokens"}


def read_table(path: Path):

    rows: dict = {}
    ours: dict = {}
    position = None
    for line in path.read_text(encoding="utf-8").split("\n"):
        match = re.match(r"### (\w+)\s", line)
        if match:
            position = match.group(1)
            rows[position] = {}
            continue
        if not line.startswith("| ") or position is None:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[1] in ("MAE", ""):
            continue
        name = cells[0].strip("*")
        rows[position][name] = cells
        if cells[0].startswith("**"):
            ours[position] = name
    return rows, ours


def number(text: str) -> float:
    return float(text.replace(",", "").replace("\u2212", "-"))


def load_predictions(fit_dir: Path, wanted: set[str] | None = None) -> list[dict]:
    rows = []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("label_status", "observed") != "observed":
                continue
            if wanted and record["candidate_id"] not in wanted:
                continue
            rows.append(record)
    return rows


def position_of(record: dict) -> str:
    return record["experiment_id"].split("--")[0]


def f1_one_run(output: Path) -> None:

    fit = RUNS / "mt-v18" / "split-20260719" / "fit-20260719"
    launch_fit = RUNS / "ti-v18" / "split-20260719" / "fit-20260719"
    wanted = {"task_update_ours", "empirical", "in_call_ours", "call_start_ours"}
    by = defaultdict(list)
    for record in load_predictions(fit, wanted):
        by[(position_of(record), record["candidate_id"])].append(record)


    launch = {r["run_id"]: r["target_value"] for r in by[("task_initialization", "empirical")]}
    median = float(np.median(list(launch.values())))
    run = min(launch, key=lambda r: abs(launch[r] - median))

    def series(position: str, candidate: str) -> list[dict]:
        return sorted((r for r in by[(position, candidate)] if r["run_id"] == run),
                      key=lambda r: r["cutoff_event_seq"])

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.8), gridspec_kw={"wspace": 0.26})


    ax = axes[0]
    points = series("task_update", "task_update_ours")
    prior = {r["cutoff_event_seq"]: r for r in series("task_update", "empirical")}
    start = next(r for r in load_predictions(launch_fit, {"input_output_scaled_interval"})
                 if r["run_id"] == run)
    xs = list(range(len(points) + 1))
    truth = [launch[run] / 1e3] + [r["target_value"] / 1e3 for r in points]
    mid = [start["prediction"] / 1e3] + [r["prediction"] / 1e3 for r in points]
    low = [start["lower"] / 1e3] + [r["lower"] / 1e3 for r in points]
    high = [start["upper"] / 1e3] + [r["upper"] / 1e3 for r in points]
    ax.fill_between(xs, low, high, color=fs.OURS_FILL, linewidth=0.6, edgecolor="white", zorder=1,
                    label="90% interval, ours")
    px = [i + 1 for i, r in enumerate(points) if r["cutoff_event_seq"] in prior]
    kept = [prior[r["cutoff_event_seq"]] for r in points if r["cutoff_event_seq"] in prior]
    ax.plot(px, [r["upper"] / 1e3 for r in kept], color=fs.REFERENCE.dark, linewidth=1.0,
            linestyle=fs.DASH, zorder=3, label="90% interval, empirical prior")
    ax.plot(px, [r["lower"] / 1e3 for r in kept], color=fs.REFERENCE.dark, linewidth=1.0,
            linestyle=fs.DASH, zorder=3)
    ax.step(xs, truth, where="post", color=TEXT, linewidth=1.3, zorder=4, label="realized value")
    ax.plot(xs, mid, zorder=5, label="point forecast, ours", **line_kwargs("ours"))
    ax.text(0.2, high[0], "Task launch", fontsize=6.5, color=TEXT, alpha=0.8, va="bottom", ha="left")
    grid_on(ax, "y")
    ax.set_xlim(-0.5, len(points) + 0.5)
    ax.set_ylim(bottom=min(0.0, min(low)), top=max(high) * 1.16)
    ax.set_xlabel("call index $k$ within the run")
    ax.set_ylabel("Tokens the run still needs (thousands)")
    panel_title(ax, "a", f"Task update, {len(points)} points")
    handles, labels = ax.get_legend_handles_labels()


    ax = axes[1]
    inside = series("in_call_visible", "in_call_ours")
    flat = {r["cutoff_event_seq"]: r for r in series("in_call_visible", "empirical")}
    xs = list(range(len(inside)))
    ax.fill_between(xs, [r["lower"] for r in inside], [r["upper"] for r in inside],
                    color=fs.OURS_FILL, linewidth=0.6, edgecolor="white", zorder=1)
    ceiling = [flat[r["cutoff_event_seq"]]["upper"] for r in inside if r["cutoff_event_seq"] in flat]
    if ceiling:
        ax.plot([i for i, r in enumerate(inside) if r["cutoff_event_seq"] in flat], ceiling,
                color=fs.REFERENCE.dark, linewidth=1.0, linestyle=fs.DASH, zorder=3)
    ax.plot(xs, [r["target_value"] for r in inside], color=TEXT, linewidth=1.3, zorder=4)
    ax.plot(xs, [r["prediction"] for r in inside], zorder=5, **line_kwargs("ours"))
    calls = len({r["logical_call_id"] for r in inside})
    for index in range(1, len(inside)):
        if inside[index]["logical_call_id"] != inside[index - 1]["logical_call_id"]:
            ax.axvline(index - 0.5, color=GRID, linewidth=0.5, zorder=0)
    grid_on(ax, "y")
    ax.set_xlim(-0.5, len(inside) - 0.5)
    ceiling_top = max(ceiling) if ceiling else 0
    ax.set_ylim(bottom=0, top=max(max(r["upper"] for r in inside), ceiling_top) * 1.16)
    ax.set_xlabel("in-call checkpoint, in run order")
    ax.set_ylabel("remaining visible output (UTF-8 bytes)")
    panel_title(ax, "b", f"In-call update, {len(inside)} points over {calls} Calls")

    axes[0].legend(handles, labels, loc="upper right", fontsize=5.8)
    fig.tight_layout()
    save(fig, output, "f1_one_run")
    errors = [abs(r["prediction"] - r["target_value"]) for r in inside]
    print(f"  f1 run={run.split('--')[1]} launch={launch[run]:,.0f} median={median:,.0f} "
          f"in-call MAE={np.mean(errors):.1f}", flush=True)


def best_outside_ours(rows: dict, ours_id: str):

    best = None
    for name, cells in rows.items():
        if name == ours_id or family_of(name.split(":")[-1]) == "ours":
            continue
        mae = number(cells[1])
        if best is None or mae < best[1]:
            best = (name, mae)
    return best


def f2_target_spread(output: Path) -> None:

    fit = sorted((RUNS / "mt-v18").glob("split-*/fit-*"))[0]
    rows = load_predictions(fit)
    seen: dict[str, str] = {}
    targets: dict[str, list[float]] = defaultdict(list)
    for record in rows:
        position = position_of(record)
        seen.setdefault(position, record["candidate_id"])
        if record["candidate_id"] == seen[position]:
            targets[position].append(float(record["target_value"]))

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3})
    ax = axes[0]
    order = [p for p in POSITIONS if p in targets]
    ratios = []
    for index, position in enumerate(order):
        values = np.asarray(targets[position])
        scaled = values / np.median(values)
        parts = ax.violinplot([scaled], positions=[index], widths=0.75, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(SEQUENTIAL[1])
            body.set_edgecolor("none")
            body.set_alpha(1.0)
        q1, q3 = np.percentile(scaled, [25, 75])
        ax.plot([index, index], [q1, q3], color=TEXT, linewidth=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([index], [1.0], "o", color="white", markersize=3, zorder=4)
        ratios.append(np.percentile(values, 99) / np.median(values))

    ax.set_yscale("log")
    for index, ratio in enumerate(ratios):
        ax.text(index, 0.98, f"{ratio:.1f}x", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=TEXT)
    ax.text(-0.03, 0.98, "p99/p50", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=TEXT)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT_TITLES[p] for p in order], fontsize=6.5)
    ax.set_ylabel("target / its median")
    ax.axhline(1.0, color=GRID, linewidth=0.8, zorder=0)
    panel_title(ax, "a", "Every position spans an order of magnitude")

    ax = axes[1]
    report = json.loads((RUNS / "mt-v19" / "repeat" / "repeat_run_coverage.json").read_text(encoding="utf-8"))
    dispersion = report["dispersion"]
    labels = ["marginal\nmedian", "sibling-median\noracle"]
    values = [dispersion["marginal_median"]["wape"], dispersion["oracle_loo_sibling_median"]["wape"]]
    ax.bar([0], [values[0]], width=0.55, **bar_kwargs("reference"))
    ax.bar([1], [values[1]], width=0.55, **bar_kwargs("ours"))
    for index, value in enumerate(values):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("WAPE at Task launch")
    ax.set_ylim(0, max(values) * 1.25)
    ax.text(0.5, max(values) * 1.12,
            f"knowing the Task exactly removes {100 * (1 - values[1] / values[0]):.0f}% of the error",
            ha="center", fontsize=6.5, color=TEXT)
    panel_title(ax, "b", "A Task-only forecast cannot pass this floor")
    fig.tight_layout()
    save(fig, output, "f2_target_spread")


def f3_method_by_position(output: Path) -> None:

    rows, ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    fig, axes = plt.subplots(1, 5, figsize=(WIDE, 2.9), sharey=False)
    families = ["published", "generic", "llm", "reference"]
    for ax, position in zip(axes, POSITIONS):
        table = rows.get(position, {})
        if position not in ours:
            ax.axis("off")
            continue
        base = number(table[ours[position]][1])
        for slot, family in enumerate(families):
            values = [100 * (number(c[1]) - base) / base for n, c in table.items()
                      if family_of(n) == family and n != ours[position]
                      and not n.startswith("gru-v18:")]
            values = [v for v in values if v < 400]
            if not values:
                continue
            spec = fs.FAMILY[family]
            ax.scatter([slot] * len(values), values, marker=spec.marker, s=16,
                       facecolors="white" if spec.hollow else spec.base,
                       edgecolors=spec.dark, linewidths=0.8, zorder=3)
            ax.plot([slot - 0.28, slot + 0.28], [min(values)] * 2, color=spec.dark, linewidth=1.6, zorder=4)
        ax.axhline(0, color=fs.OURS.base, linewidth=1.6, zorder=2)
        ax.set_xlim(-0.6, len(families) - 0.4)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(["pub", "gen", "llm", "ref"], fontsize=6.5)
        ax.set_title(TITLES[position], fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
    axes[0].set_ylabel("MAE above the recipe (%)")
    handles = [plt.Line2D([], [], color=fs.OURS.base, linewidth=1.6, label="the reported recipe (0%)")]
    for family, label in (("published", "published method"), ("generic", "generic predictor"),
                          ("llm", "LLM-prompted protocol"), ("reference", "non-learned reference")):
        spec = fs.FAMILY[family]
        handles.append(plt.Line2D([], [], color=spec.dark, marker=spec.marker, linestyle="",
                                  markerfacecolor="white" if spec.hollow else spec.base,
                                  markersize=spec.marker_size, label=label))
    fig.tight_layout()
    legend_bottom(fig, handles, [h.get_label() for h in handles], ncol=5, y=-0.02)
    save(fig, output, "f3_method_by_position")


def f4_in_call_source(output: Path) -> None:

    curve = json.loads((RUNS / "mt-v18" / "prefix-curve" / "prefix_progress.json").read_text(encoding="utf-8"))
    shares = [b["produced_share_median"] for b in curve["bands"]]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.1), gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.08})

    ax = axes[0]
    drawn = [("in_call_ours", "ours", "base", "the recipe", False),
             ("in_call_prefix", "ours", "light", "state + prefix embedding", False),
             ("trail_last", "published", "base", "TRAIL, last token", False),
             ("s3_encoder", "published", "dark", "S3 bin classifier", False),
             ("lightgbm_generic", "generic", "base", "generic LightGBM", False),
             ("empirical", "reference", "base", "empirical prior", True)]
    for candidate, family, step, label, reference in drawn:
        values = curve["mae"].get(candidate)
        if not values:
            continue
        ys = [values[str(b["band"])] for b in curve["bands"]]
        ax.plot(shares, ys, label=label, **line_kwargs(family, step=step, reference=reference))
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([40, 100, 400]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xticks(shares)
    ax.set_xticklabels([f"{s:.0%}" for s in shares], fontsize=6.5)
    ax.set_xlabel("share of the Call already produced")
    ax.set_ylabel("MAE (UTF-8 bytes)")
    panel_title(ax, "a", "Error as the Call proceeds")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    report = json.loads((RUNS / "ia-v18" / "ablation" / "ablation_summary.json").read_text(encoding="utf-8"))
    wanted = [("in_call_visible", "ic_without_generation_prefix", "generation prefix"),
              ("in_call_visible", "ic_without_closure", "closure state"),
              ("in_call_visible", "ic_without_session_recall", "session recall"),
              ("in_call_billed", "ic_billed_without_clock", "stream clock"),
              ("in_call_billed", "ic_billed_without_generation_prefix", "generation prefix"),
              ("call_start", "cs_without_token_history", "Token history"),
              ("call_start", "cs_without_current_progress", "current progress"),
              ("task_update", "tu_without_token_history", "Token history"),
              ("task_update", "tu_with_task_info", "Task text, added")]

    row_labels, centres, lows, highs, colours = [], [], [], [], []
    for position, arm, label in wanted:
        block = report.get(position, {}).get(arm)
        if not block:
            continue
        reference = block["reference_mae"]
        relative = 100 * block["delta"] / reference
        seeds = [100 * s["delta"] / reference for s in block["seeds"]]
        row_labels.append(f"{label}  ({TITLES[position]})")
        centres.append(relative)
        lows.append(relative - min(seeds))
        highs.append(max(seeds) - relative)
        colours.append(fs.OURS.base if relative > 0 else fs.PUBLISHED.base)
    y = list(range(len(row_labels)))
    ax.errorbar(centres, y, xerr=[lows, highs], fmt="none", ecolor=TEXT, elinewidth=0.8, capsize=2, zorder=2)
    ax.scatter(centres, y, marker="o", s=26, color=colours, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=6.2)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_xlabel("MAE change when the source is removed (%)")
    panel_title(ax, "b", "What each source carries")
    fig.tight_layout()
    legend_bottom(fig, handles, labels, ncol=3, y=-0.02)
    save(fig, output, "f4_in_call_source")


def f5_gain_under_cuts(output: Path) -> None:

    ours_at = {"call_start": "call_start_ours", "in_call_visible": "in_call_ours",
               "in_call_billed": "in_call_ours_clock", "task_update": "task_update_ours"}

    def from_table(path: Path) -> dict:
        rows, ours = read_table(path)
        out = {}
        for position in POSITIONS:
            if position not in rows or position not in ours:
                continue
            rival = best_outside_ours(rows[position], ours[position])
            if rival:
                out[position] = 100 * (number(rows[position][ours[position]][1]) - rival[1]) / rival[1]
        return out

    def from_cells(cells: dict, block: str) -> dict:
        out = {}
        for cell_id, cell in cells.items():
            position = cell_id.split("--")[0]
            mine = cell.get(ours_at.get(position, ""), {}).get(block, {}).get("mae")
            if mine is None:
                continue
            rival = min((v[block]["mae"] for k, v in cell.items()
                         if family_of(k) != "ours" and block in v), default=None)
            if rival:
                out[position] = 100 * (mine - rival) / rival
        return out

    v18_rows, v18_ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    v18 = from_table(RUNS / "mt-v18" / "table" / "summary.md")
    transfer = json.loads((RUNS / "ct-v18-to-v21" / "condition_transfer.json").read_text(encoding="utf-8"))
    length = json.loads((RUNS / "le-v18" / "length_extrapolation.json").read_text(encoding="utf-8"))
    sub_rows, _ = read_table(RUNS / "sub-learner_substitution" / "table" / "summary.md")


    learner = {}
    for position in v18:
        gru = next((c for c in sub_rows.get(position, {}) if c.endswith("__learner_gru")), None)
        rival = best_outside_ours(v18_rows[position], v18_ours[position])
        if gru and rival:
            learner[position] = 100 * (number(sub_rows[position][gru][1]) - rival[1]) / rival[1]

    cuts = [("v18 held-out Tasks", v18, False),
            ("v21 reasoning off", from_table(RUNS / "mt-v21" / "table" / "summary.md"), False),
            ("v19 79 Tasks x 4 runs", from_table(RUNS / "mt-v19" / "table" / "summary.md"), False),
            ("zero-shot v18 to v21", from_cells(transfer["cells"], "zero_shot"), False),
            ("longest length band held out", from_cells(length["cells"], "length"), False),
            ("same features, sequence GRU", learner, True)]

    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(WIDE, 2.35), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, position in zip(axes, POSITIONS):
        ax.axvline(0, color=fs.REFERENCE.dark, linewidth=0.9, linestyle=fs.DOT, zorder=1)
        for index, (_, values, swapped) in enumerate(cuts):
            value = values.get(position)
            if value is None:
                ax.text(0, index, "n/a", fontsize=6.2, color=fs.REFERENCE.dark, ha="center",
                        va="center", zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
                continue
            marker = fs.GENERIC.marker if swapped else fs.OURS.marker
            ahead = value < 0
            ax.plot([value], [index], marker=marker, markersize=5.5 if ahead else 5.0,
                    color=fs.OURS.base if ahead else fs.OURS.light,
                    markerfacecolor=fs.OURS.base if ahead else "white",
                    markeredgecolor=fs.OURS.base, markeredgewidth=1.1, zorder=3, linestyle="none")
            ax.annotate(f"{value:+.0f}", (value, index), textcoords="offset points",
                        xytext=(0, 5.5), fontsize=5.8, color=TEXT, ha="center", zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        grid_on(ax, "x")
        ax.set_ylim(len(cuts) - 0.4, -1.1)
        ax.set_xlim(-44, 28)
        ax.set_xticks([-40, -20, 0, 20])
        ax.tick_params(axis="x", labelsize=6.5, labelbottom=ax is axes[0])
        ax.grid(axis="y", visible=False)
        panel_title(ax, "abcde"[POSITIONS.index(position)], SHORT_TITLES[position].replace("\n", " "))
    axes[0].set_yticks(range(len(cuts)))
    axes[0].set_yticklabels([c[0] for c in cuts], fontsize=6.5)
    axes[0].set_xticklabels(["-40%", "-20", "0", "+20"], fontsize=6.5)
    fig.tight_layout()
    fig.supxlabel("MAE against the strongest candidate outside our family, % of that candidate"
                  "  (left of the dotted rule: ahead)", fontsize=8, y=-0.03)
    save(fig, output, "f5_gain_under_cuts")
    for label, values, _ in cuts:
        print(f"  {label:<30} " + "  ".join(
            f"{p[:6]}={values[p]:+.1f}" if p in values else f"{p[:6]}=n/a" for p in POSITIONS), flush=True)


def f6_interval_range(output: Path) -> None:

    from matplotlib.ticker import FuncFormatter

    def compact(value, _):
        if abs(value) >= 1e6:
            return f"{value / 1e6:g}M"
        if abs(value) >= 1e3:
            return f"{value / 1e3:g}k"
        return f"{value:g}"

    grids = [("v18 Call start", RUNS / "mt-v18" / "frontier-cs-ic", "call_start", "output Tokens"),
             ("v18 In-call, billed", RUNS / "mt-v18" / "frontier-cs-ic", "in_call_billed", "output Tokens"),
             ("v19 Task update", RUNS / "mt-v19" / "frontier", "task_update", "run Tokens")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 3.7))
    for column, (title, path, position, unit) in enumerate(grids):
        report = json.loads((path / "interval_frontier.json").read_text(encoding="utf-8"))
        block = next((v for k, v in report["experiments"].items() if k.startswith(position)), {})
        for candidate, cells in block.items():
            family = family_of(candidate)
            if family not in ("ours", "generic", "reference"):
                continue
            curve = cells["per_alpha"]
            alphas = sorted(curve, key=float)
            style = line_kwargs(family, reference=family == "reference")
            axes[0][column].plot([curve[a]["coverage"] for a in alphas],
                                 [curve[a]["normalized_width"] for a in alphas], **style)
            axes[1][column].plot([curve[a]["exceedance"] for a in alphas],
                                 [curve[a]["extra_reserved"] for a in alphas], **style)
        axes[0][column].axvline(0.90, color=TEXT, linewidth=0.8, linestyle=fs.DOT)
        axes[1][column].axvline(0.10, color=TEXT, linewidth=0.8, linestyle=fs.DOT)
        panel_title(axes[0][column], "abc"[column], title)
        panel_title(axes[1][column], "def"[column], title)
        axes[0][column].set_xlabel("realized coverage")
        axes[1][column].set_xlabel("exceedance rate")
        axes[1][column].yaxis.set_major_formatter(FuncFormatter(compact))
        axes[1][column].set_ylabel(f"extra reserved ({unit})" if column in (0, 2) else "")
        for row in (0, 1):
            grid_on(axes[row][column], "y")
    axes[0][0].set_ylabel("width / median target")
    axes[0][0].text(0.90, 0.02, "nominal", fontsize=6.0, color=TEXT, rotation=90,
                    transform=axes[0][0].get_xaxis_transform(), ha="right", va="bottom")
    axes[1][0].text(0.10, 0.98, "target risk", fontsize=6.0, color=TEXT, rotation=90,
                    transform=axes[1][0].get_xaxis_transform(), ha="right", va="top")
    handles = [plt.Line2D([], [], **{k: v for k, v in line_kwargs("ours").items() if k != "markeredgewidth"}, label="the recipe"),
               plt.Line2D([], [], **{k: v for k, v in line_kwargs("generic").items() if k != "markeredgewidth"}, label="generic predictor"),
               plt.Line2D([], [], **{k: v for k, v in line_kwargs("reference", reference=True).items() if k != "markeredgewidth"}, label="non-learned reference")]
    axes[0][2].legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=5.8)
    fig.tight_layout()
    save(fig, output, "f6_interval_range")


def f7_budget_control(output: Path) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.9), gridspec_kw={"wspace": 0.42})
    report = json.loads((RUNS / "mt-v18" / "budget" / "budget_control.json").read_text(encoding="utf-8"))

    ax = axes[0]
    series = [(("fixed_cap", ""), "reference", "fixed cap", True),
              (("abort", "empirical"), "reference", "abort on the prior", True),
              (("abort", "lightgbm_generic"), "generic", "abort on the generic learner", False),
              (("abort", "task_update_ours"), "ours", "abort on the recipe", False)]
    for (policy, candidate), family, label, reference in series:
        points = sorted((row["spent_tokens"] / 1e6, row["completed_rate"]) for row in report["mean_rows"]
                        if row["policy"] == policy and row.get("candidate", "") == candidate)
        frontier, best = [], -1.0
        for spend, completed in points:
            if completed > best:
                frontier.append((spend, completed))
                best = completed
        if not frontier:
            continue
        style = line_kwargs(family, step="dark" if candidate == "empirical" else "base", reference=reference)
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], label=label, **style)
    ax.set_xlabel("Tokens spent (millions)")
    ax.set_ylabel("completion rate")
    panel_title(ax, "a", "Cost against completion")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    for cohort, step in (("v18", "base"), ("v19", "light")):
        block = json.loads((RUNS / f"mt-{cohort}" / "budget" / "budget_control.json").read_text(encoding="utf-8"))
        points = []
        for family_key in {row["family"] for row in block["matched_completion"] if row["family"].startswith("abort:")}:
            rows = [r for r in block["matched_completion"] if r["family"] == family_key]
            mae = rows[0].get("forecast_mae")
            if mae is None:
                continue
            savings = sorted(100 * r["saving"] for r in rows)
            points.append((mae / 1000.0, savings[len(savings) // 2], family_key.split(":", 1)[1]))
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], color=GRID, linewidth=1.0, zorder=1)
        for mae, saving, name in points:
            family = "ours" if name.endswith("_ours") else family_of(name)
            spec = fs.FAMILY[family]
            ax.scatter([mae], [saving], marker=spec.marker, s=26 if family == "ours" else 18,
                       facecolors="white" if spec.hollow else getattr(spec, step),
                       edgecolors=spec.dark, linewidths=0.8, zorder=3)
        if cohort == "v18":
            for mae, saving, name in points:
                if name in ("task_update_ours", "lightgbm_generic", "empirical"):
                    ax.annotate(name.replace("_", " "), (mae, saving), textcoords="offset points",
                                xytext=(6, 2), fontsize=6, color=TEXT)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    ax.set_xlabel("Task-update MAE (thousand)")
    ax.set_ylabel("median saving (%)")
    panel_title(ax, "b", "Error against saving")

    ax = axes[2]
    budgets = sorted({row["budget"] for row in report["mean_rows"]})[:-1]
    for family_name, candidate, step in (("ours", "task_update_ours", "base"), ("generic", "lightgbm_generic", "base")):
        xs, ys = [], []
        for budget in budgets:
            rows = [r for r in report["mean_rows"]
                    if r["policy"] == "abort" and r.get("candidate") == candidate and r["budget"] == budget]
            if not rows:
                continue
            best = min(rows, key=lambda r: r["false_stop_rate"])
            xs.append(budget / 1000.0)
            ys.append(100 * best["false_stop_rate"])
        ax.plot(xs, ys, **line_kwargs(family_name, step=step))
    ax.set_xlabel("budget (thousand)")
    ax.set_ylabel("false-stop rate (%)")
    panel_title(ax, "c", "False stops")
    axes[0].legend(handles, labels, loc="lower right", fontsize=5.8)
    fig.tight_layout()
    save(fig, output, "f7_budget_control")


def f8_cost_vs_error(output: Path) -> None:

    overhead = json.loads((RUNS / "overhead-idle" / "online_overhead.json").read_text(encoding="utf-8"))
    build_ms = overhead["build"]["ms_per_row_median"]
    passes = overhead["representation_pass_ms"]
    rows, ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    position = "in_call_visible"
    table = rows[position]
    pass_ms = passes[position]["proxy_ms_median"] + passes[position]["encoder_ms_median"]

    fig, ax = plt.subplots(figsize=(COLUMN, 2.5))
    groups = [("ours", [ours[position]], build_ms), ("generic", ["lightgbm_generic", "sequence_gru"], build_ms),
              ("published", ["trail_last", "egtp_pool", "s3_encoder", "tie_encoder", "ltr_c", "magnus",
                             "ssjf_reg", "ssjf_mc", "estp", "trail_mean", "tpv"], pass_ms),
              ("llm", ["response_length_prompt_calibrated"], 7500.0)]
    for family, names, cost in groups:
        spec = fs.FAMILY[family]
        values = [number(table[n][1]) for n in names if n in table]
        if not values:
            continue
        ax.scatter([cost] * len(values), values, marker=spec.marker,
                   s=48 if family == "ours" else 20,
                   facecolors="white" if spec.hollow else spec.base,
                   edgecolors=spec.dark, linewidths=0.9, zorder=3,
                   label={"ours": "the recipe", "generic": "generic predictor",
                          "published": "published head on proxy states",
                          "llm": "the model's own estimate"}[family])
    ax.set_xscale("log")
    ax.set_xlabel("online path per point (ms, log)")
    ax.set_ylabel("MAE (UTF-8 bytes)")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.set_title("What one forecast costs", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, output, "f8_cost_vs_error")


def _slices() -> dict:
    path = RUNS / "mt-v18" / "slices" / "prediction_slices.json"
    return json.loads(path.read_text(encoding="utf-8"))["positions"]


def a1_conditional_coverage(output: Path) -> None:

    slices = _slices()
    drawn = [("in_call_ours", "task_update_ours", "ours", "base", "the recipe", False),
             ("lightgbm_generic", "lightgbm_generic", "generic", "base", "generic LightGBM", False),
             ("trail_last", "trail_last", "published", "base", "TRAIL, last token", False),
             ("empirical", "empirical", "reference", "base", "empirical prior", True)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})

    ax = axes[0]
    for visible, _, family, step, label, reference in drawn:
        bands = slices["in_call_visible"].get(visible, {}).get("coverage_by_share")
        if not bands:
            continue
        ax.plot([b["share_median"] for b in bands], [b["coverage"] for b in bands],
                label=label, **line_kwargs(family, step=step, reference=reference))
    ax.set_xlabel("share of the Call already produced (band median)")
    ax.set_ylabel("realized coverage of the 90% interval")
    panel_title(ax, "a", "In-call visible, by produced share")
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[1]
    for _, update, family, step, label, reference in drawn:
        quintiles = slices["task_update"].get(update, {}).get("coverage_by_quintile")
        if not quintiles:
            continue
        ax.plot([q["quintile"] + 1 for q in quintiles], [q["coverage"] for q in quintiles],
                **line_kwargs(family, step=step, reference=reference))
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("call-index quintile within the run")
    panel_title(ax, "b", "Task update, by position in the run")

    for ax in axes:
        grid_on(ax, "y")
        ax.axhline(0.90, color=fs.REFERENCE.dark, linewidth=0.9, linestyle=fs.DOT, zorder=1)
        ax.set_ylim(0.66, 1.015)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    axes[0].text(0.03, 0.905, "nominal 0.90", fontsize=6.5, color=TEXT, alpha=0.8,
                 transform=axes[0].get_yaxis_transform(), va="bottom")
    fig.tight_layout()
    legend_bottom(fig, handles, labels, ncol=4, y=-0.02)
    save(fig, output, "a1_conditional_coverage")


def a2_error_ecdf(output: Path) -> None:

    slices = _slices()
    rows, ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    cells = ["call_start", "in_call_visible", "in_call_billed", "task_update"]
    probabilities = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(1, len(cells), figsize=(WIDE, 2.5), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    from matplotlib.ticker import LogLocator, NullFormatter

    handles: list = []
    labels: list[str] = []
    for ax, position in zip(axes, cells):
        available = slices.get(position, {})
        mine = ours.get(position)
        picks = [(mine, "ours", "the recipe", False)]
        for family, label in [("generic", "best generic"), ("published", "best published head"),
                              ("reference", "empirical prior")]:
            pool = [c for c in available if family_of(c) == family and c in rows.get(position, {})]
            if family == "reference":
                pool = [c for c in pool if c == "empirical"] or pool
            if pool:
                picks.append((min(pool, key=lambda c: available[c]["mae"]), family, label,
                              family == "reference"))
        for candidate, family, label, reference in picks:
            block = available.get(candidate)
            if not block:
                continue
            values = np.asarray(block["error_quantiles"], dtype=float)
            keep = values > 0
            style = line_kwargs(family, reference=reference)
            style["marker"] = "none"
            line, = ax.plot(values[keep], probabilities[keep], **style)
            for probability in (0.5, 0.9):
                ax.plot([np.interp(probability, probabilities, values)], [probability],
                        marker=fs.FAMILY[family].marker, markersize=fs.FAMILY[family].marker_size,
                        color=style["color"], markerfacecolor="white" if fs.FAMILY[family].hollow else style["color"],
                        markeredgecolor=style["color"], markeredgewidth=1.0, linestyle="none", zorder=4)
            if position == "call_start":
                handles.append(line)
                labels.append(label)
            if family == "published":

                ax.text(0.97, 0.06, candidate, transform=ax.transAxes, fontsize=6.0,
                        color=fs.PUBLISHED.dark, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(0, 1.0)


        drawn_values = [np.asarray(available[c]["error_quantiles"], dtype=float)
                        for c, *_ in picks if c in available]
        if drawn_values:
            ax.set_xlim(min(np.interp(0.03, probabilities, v) for v in drawn_values) * 0.7,
                        max(np.interp(0.995, probabilities, v) for v in drawn_values) * 1.4)
        ax.set_xlabel(UNITS[position])
        panel_title(ax, "abcd"[cells.index(position)], TITLES[position])
    axes[0].set_ylabel("share of scored points at or below")
    for ax in axes:
        ax.axhline(0.9, color=fs.REFERENCE.dark, linewidth=0.7, linestyle=fs.DOT, zorder=1)
    axes[0].text(0.02, 0.905, "P90", fontsize=6.2, color=TEXT, alpha=0.8, transform=axes[0].transAxes)
    axes[0].legend(handles, labels, loc="upper left", fontsize=5.8)
    fig.tight_layout()
    save(fig, output, "a2_error_ecdf")


def a3_error_vs_ordering(output: Path) -> None:

    table = json.loads((RUNS / "mt-v18" / "table" / "main_table.json").read_text(encoding="utf-8"))
    panel = table["panels"]["mt-v18"]["experiments"]
    wanted = {"call_start": "Call start", "task_update": "Task update"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.6), gridspec_kw={"wspace": 0.24})
    from matplotlib.ticker import FixedLocator, NullFormatter

    for ax, (position, title) in zip(axes, wanted.items()):
        cell = next((v for k, v in panel.items() if k.split("--")[0] == position), None)
        if cell is None:
            continue
        arms = {name: (block["metrics"]["mae"], block["metrics"].get("spearman", 0.0))
                for name, block in cell["arms"].items()}
        floor = min(mae for mae, _ in arms.values())


        ratios = sorted(mae / floor for mae, _ in arms.values())
        edge = min(max(ratios[int(0.85 * (len(ratios) - 1))] * 1.06, 1.35), max(ratios))
        beyond = 0
        for name, (mae, rho) in arms.items():
            family = "ours" if name == cell["ours"] else family_of(name)
            spec = fs.FAMILY[family]
            ratio = mae / floor
            clipped = ratio > edge
            beyond += int(clipped)
            ax.plot([min(ratio, edge)], [rho], marker=spec.marker, markersize=spec.marker_size,
                    color=spec.base, markerfacecolor="white" if spec.hollow or clipped else spec.base,
                    markeredgecolor=spec.base, markeredgewidth=1.0, linestyle="none",
                    alpha=0.55 if clipped else 1.0, zorder=4 if family == "ours" else 3)
        mine = arms[cell["ours"]][0]
        near = sorted((n for n in arms if n != cell["ours"]),
                      key=lambda n: abs(arms[n][0] - mine))[:3]
        for offset, name in enumerate([cell["ours"], *near]):
            mae, rho = arms[name]
            ax.annotate(name, (min(mae / floor, edge), rho), textcoords="offset points",
                        xytext=(7, 7 - 9 * offset), fontsize=5.8, color=TEXT,
                        arrowprops={"arrowstyle": "-", "color": fs.REFERENCE.dark,
                                    "linewidth": 0.6, "shrinkA": 0, "shrinkB": 2})
        if beyond:
            ax.text(0.985, 0.5, f"{beyond} beyond, to {max(ratios):.1f}x", transform=ax.transAxes,
                    fontsize=6.0, color=fs.REFERENCE.dark, rotation=90, ha="right", va="center")
        ax.axhline(0, color=fs.REFERENCE.dark, linewidth=0.9, linestyle=fs.DOT, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.985, edge * 1.03)
        ax.xaxis.set_major_locator(FixedLocator([1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("MAE, multiple of the best in the panel")
        panel_title(ax, "ab"[list(wanted).index(position)], title)
    axes[0].set_ylabel("Spearman rho with the realized value")
    fig.tight_layout()
    save(fig, output, "a3_error_vs_ordering")


def a4_transfer_regimes(output: Path) -> None:

    transfer = json.loads((RUNS / "ct-v18-to-v21" / "condition_transfer.json").read_text(encoding="utf-8"))
    regimes = [("zero_shot", "zero-shot"), ("few_shot_5", "5"), ("few_shot_10", "10"),
               ("few_shot_20", "20"), ("pooled", "pooled"), ("in_domain", "in-domain")]
    ours_at = {"call_start": "call_start_ours", "in_call_visible": "in_call_ours",
               "in_call_billed": "in_call_ours_clock", "task_update": "task_update_ours"}
    drawn = [("ours", "base", "the recipe", False), ("generic", "base", "generic LightGBM", False),
             ("reference", "base", "empirical prior", True)]
    cells = [p for p in POSITIONS if p in ours_at and p in transfer["cells"]]
    fig, axes = plt.subplots(1, len(cells), figsize=(WIDE, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    handles, labels = [], []
    for ax, position in zip(axes, cells):
        cell = transfer["cells"][position]
        arms = {"ours": ours_at[position], "generic": "lightgbm_generic", "reference": "empirical"}
        for family, step, label, reference in drawn:
            block = cell.get(arms[family])
            if not block or "in_domain" not in block:
                continue
            base = block["in_domain"]["mae"]
            xs, ys = [], []
            for index, (key, _) in enumerate(regimes):
                if key in block:
                    xs.append(index)
                    ys.append(100 * (block[key]["mae"] - base) / base)
            line, = ax.plot(xs, ys, **line_kwargs(family, step=step, reference=reference))
            if position == cells[0]:
                handles.append(line)
                labels.append(label)
        grid_on(ax, "y")
        ax.axhline(0, color=TEXT, linewidth=0.9, zorder=1)
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels([r[1] for r in regimes], fontsize=6.0, rotation=38, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(-13, 66)
        panel_title(ax, "abcd"[cells.index(position)], TITLES[position])
    axes[0].set_ylabel("MAE above the in-domain fit (%)")
    fig.tight_layout()
    fig.supxlabel("target-cohort Tasks the fit was allowed to see", fontsize=8, y=-0.09)
    legend_bottom(fig, handles, labels, ncol=3, y=-0.125)
    save(fig, output, "a4_transfer_regimes")


def a5_llm_protocols(output: Path) -> None:

    prompted = json.loads((RUNS / "rlp-v18" / "summary.json").read_text(encoding="utf-8"))
    pooled: dict = defaultdict(lambda: defaultdict(list))
    for row in prompted["mae_by_position"]:
        kind = "calibrated" if row["candidate"].endswith("_calibrated") else "raw"
        pooled[row["position"]][kind].append(row["mae"])
    for line in (RUNS / "se-mini-v18" / "summary.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].startswith("split-"):
            continue
        kind = "calibrated" if cells[2].endswith("_calibrated") else "raw"
        pooled[cells[1]][kind].append(float(cells[3]))

    order = [("call_start", "asked mid-Call, Call start", "output Tokens"),
             ("in_call_billed", "asked mid-Call, In-call billed", "output Tokens"),
             ("in_call_visible", "asked mid-Call, In-call visible", "UTF-8 bytes"),
             ("task_initialization", "asked before the run, Task launch", "run Tokens"),
             ("task_update", "asked before the run, Task update", "run Tokens")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.7),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.75})

    ax = axes[0]
    rows = [(label, float(np.mean(pooled[key]["raw"])), float(np.mean(pooled[key]["calibrated"])), unit)
            for key, label, unit in order if pooled[key]["raw"] and pooled[key]["calibrated"]]
    for index, (_, raw, calibrated, _) in enumerate(rows):
        ax.annotate("", xy=(calibrated, index), xytext=(raw, index),
                    arrowprops={"arrowstyle": "-|>", "color": fs.LLM.dark, "linewidth": 1.5,
                                "shrinkA": 3, "shrinkB": 3})
        ax.plot([raw], [index], marker=fs.LLM.marker, markersize=fs.LLM.marker_size, linestyle="none",
                markerfacecolor="white", markeredgecolor=fs.LLM.dark, markeredgewidth=1.2, zorder=3)
        ax.plot([calibrated], [index], marker=fs.LLM.marker, markersize=fs.LLM.marker_size,
                linestyle="none", color=fs.LLM.base, markeredgecolor=fs.LLM.dark,
                markeredgewidth=1.0, zorder=3)
        ax.annotate(f"{calibrated:,.0f}", (calibrated, index), textcoords="offset points",
                    xytext=(-6, 0), fontsize=6.0, color=TEXT, ha="right", va="center")
        ax.annotate(f"{raw:,.0f}", (raw, index), textcoords="offset points",
                    xytext=(6, 0), fontsize=6.0, color=TEXT, ha="left", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{label}\n({unit})" for label, _, _, unit in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(20, 3e6)
    ax.set_ylim(len(rows) - 0.4, -0.9)
    ax.set_xlabel("MAE in the position's own unit")
    ax.grid(axis="y", visible=False)
    ask = plt.Line2D([], [], marker=fs.LLM.marker, markersize=fs.LLM.marker_size, linestyle="none",
                     markerfacecolor="white", markeredgecolor=fs.LLM.dark, markeredgewidth=1.2,
                     label="as asked")
    fixed = plt.Line2D([], [], marker=fs.LLM.marker, markersize=fs.LLM.marker_size, linestyle="none",
                       color=fs.LLM.base, markeredgecolor=fs.LLM.dark, markeredgewidth=1.0,
                       label="after calibration")
    ax.legend([ask, fixed], ["as asked", "after calibration"], loc="lower right", fontsize=5.8)
    panel_title(ax, "a", "What the ask costs in error")

    ax = axes[1]
    replay = []
    for line in (RUNS / "pe-v18-mini" / "comparison.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10 or cells[0] not in TITLES:
            continue
        replay.append((TITLES[cells[0]], int(cells[1]), number(cells[7]), number(cells[8])))
    ticks, position = [], 0
    for title, points, mine, theirs in replay:
        for offset, (value, family, who) in enumerate([(mine, "ours", "the recipe"),
                                                       (theirs, "llm", "prefix replay")]):
            ax.barh(position + offset, value, height=0.66, **bar_kwargs(family))
            if value <= 0:

                ax.plot([0], [position + offset], marker="|", markersize=7,
                        color=fs.FAMILY[family].dark, markeredgewidth=1.6, zorder=4)
            inside = value > 0.25
            ax.annotate(f"{value:.3f}", (value, position + offset), textcoords="offset points",
                        xytext=(-5 if inside else 4, 0), fontsize=6.2, zorder=5,
                        color="white" if inside else TEXT,
                        ha="right" if inside else "left", va="center")
            ticks.append((position + offset, f"{who}, {title}"))
        position += 2.5
    ax.axvline(0.90, color=fs.REFERENCE.dark, linewidth=0.9, linestyle=fs.DOT, zorder=1)
    ax.text(0.885, -0.85, "nominal 0.90", fontsize=6.2, color=TEXT, alpha=0.8, ha="right")
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=6.2)
    ax.set_ylim(max(t[0] for t in ticks) + 0.5, -1.2)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("realized coverage on the shared points")
    ax.grid(axis="y", visible=False)
    counts = " and ".join(f"{r[1]:,}" for r in replay)
    panel_title(ax, "b", f"Interval coverage ({counts} shared points)")
    fig.tight_layout()
    save(fig, output, "a5_llm_protocols")


FIGURES = {
    "f1_one_run": f1_one_run,
    "f2_target_spread": f2_target_spread,
    "f3_method_by_position": f3_method_by_position,
    "f4_in_call_source": f4_in_call_source,
    "f5_gain_under_cuts": f5_gain_under_cuts,
    "f6_interval_range": f6_interval_range,
    "f7_budget_control": f7_budget_control,
    "f8_cost_vs_error": f8_cost_vs_error,
    "a1_conditional_coverage": a1_conditional_coverage,
    "a2_error_ecdf": a2_error_ecdf,
    "a3_error_vs_ordering": a3_error_vs_ordering,
    "a4_transfer_regimes": a4_transfer_regimes,
    "a5_llm_protocols": a5_llm_protocols,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "docs" / "figures")
    parser.add_argument("--figure", action="append", default=[], choices=sorted(FIGURES))
    args = parser.parse_args()
    fs.apply()
    for name in args.figure or sorted(FIGURES):
        try:
            FIGURES[name](args.output)
        except (FileNotFoundError, KeyError, StopIteration) as error:
            print(f"skipped {name}: {type(error).__name__} {error}", flush=True)


if __name__ == "__main__":
    main()
