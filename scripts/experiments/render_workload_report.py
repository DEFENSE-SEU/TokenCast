from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"direct": "Direct", "direct_n": "Direct + N", "direct_m": "Direct + M",
         "direct_nm": "Direct + N, M", "direct_horizons": "Direct + horizons",
         "lightgbm": "LightGBM", "history_median": "History median"}


def method_figure(path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    ax.axhspan(0, 1.35, color="#f2f4f7", zorder=0)
    ax.text(0.05, 3.8, "PREDICTION: OBSERVED PREFIX ONLY", fontsize=8, weight="bold")
    ax.text(0.05, 0.13, "TRAINING ONLY: FUTURE LABELS FROM COMPLETED TRAJECTORIES", fontsize=7.5)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="#455468", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                    color="#455468", linewidth=1,
                                    linestyle="--" if dashed else "-"))

    box(0.15, 2.0, 2.3, 1.0, "Causal execution features\nusage, feedback, context", "#e8eff8")
    box(3.15, 2.0, 2.1, 1.0, "Shared MLP\n128 → 64 · SiLU", "#e8eff8")
    box(6.0, 2.0, 1.5, 1.0, "Direct head\nprovider tokens", "#dbeee7")
    ax.text(8.75, 2.5, "$\\widehat R_t$\nremaining total", ha="center", va="center")
    arrow((2.55, 2.5), (3.05, 2.5))
    arrow((5.35, 2.5), (5.9, 2.5))
    arrow((7.6, 2.5), (8.15, 2.5))
    box(3.15, 0.55, 2.1, 0.58, "Auxiliary heads: N, M", "#ece4f2")
    box(6.0, 0.55, 3.4, 0.58, "Future attempts · client-byte reuse", "#ffffff")
    arrow((4.2, 1.9), (4.2, 1.22), True)
    arrow((5.9, 0.83), (5.35, 0.83), True)
    ax.text(5.6, 1.36, "masked auxiliary loss", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.015, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def results_figure(result, path):
    comparisons = [(name, value) for name, value in result["comparisons"].items()
                   if name.endswith("_vs_direct") and not name.startswith("history_median")]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axvline(0, color="#939ba6", linewidth=0.8, linestyle="--")
    for i, (name, value) in enumerate(comparisons):
        center = value["delta_mae"] / 1000
        low, high = [v / 1000 for v in value["delta_mae_ci95"]]
        color = "#7d8997" if name.startswith("lightgbm") else "#416a96"
        ax.errorbar(center, i, xerr=[[center - low], [high - center]],
                    fmt="o", color=color, capsize=3, markersize=5, linewidth=1.5)
        ax.annotate(f"{center:+.2f} [{low:+.2f}, {high:+.2f}]",
                    (high, i), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.set_yticks(range(len(comparisons)),
                  [NAMES[name.removesuffix("_vs_direct")] for name, _ in comparisons])
    ax.invert_yaxis()
    ax.set_ylim(len(comparisons) - 0.5, -0.5)
    ax.set_xlabel("Paired MAE difference from Direct (k provider tokens; lower is better)", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + (right - left) * 0.7)
    ax.set_title(f"Development diagnostic · {result['n_tasks']} tasks · {result['n_points']} prefixes", fontsize=10)
    fig.text(0.5, 0.025, "Pointwise 95% paired task-bootstrap intervals; seeds averaged within each task.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_results(result, path, run_name, dataset):
    def escaped(value):
        return value.replace("_", r"\_")

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{New workload-supervision development diagnostic on "
             + str(result["n_tasks"]) + " tasks and " + str(result["n_points"])
             + r" Task Update points. All fitted models predict provider-accounted remaining total.}",
             r"\label{tab:workload-prototype}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Model & MAE (tokens) & WAPE (\%) \\", r"\midrule"]
    for arm, metric in result["metrics"].items():
        lines.append(f"{NAMES[arm]} & {metric['mae']:,.0f} & {metric['wape'] * 100:.1f}" + r" \\")
    for arm, entry in result["unevaluated"].items():
        missing = ", ".join("M" if name == "reuse" else name for name in entry["missing"])
        lines.append(NAMES[arm] + r" & \multicolumn{2}{c}{Not evaluated: missing " + escaped(missing) + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"These are within-cohort development results; independent generalization remains untested. "
              + (r"The reuse-dependent arms remain unevaluated. " if result["unevaluated"] else "")
              + f"The run uses {len(result['split_seeds'])} task-isolated split seeds and "
              + str(len(next(iter(dataset["partitions"].values())))) + " outer folds per seed. "
              r"Within each fold, every learned arm receives nine validation-selected trials. "
              r"Causal feature transforms and target transforms are fitted on training tasks. "
              r"A separate calibration partition remains unused in this point-prediction diagnostic.",
              r"The reproducibility package contains the exported cohort, split assignments, "
              r"fit reports, out-of-fold predictions, and selected model weights. "
              r"Its local run path is \texttt{workspace/runs/" + escaped(run_name) + r"}; "
              r"the cohort configuration is \texttt{" + escaped(Path(dataset["config_path"]).name) + r"}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper/tokencast-research")
    parser.add_argument("--run", type=Path, default=ROOT / "workspace/runs/workload-context-v1")
    args = parser.parse_args()
    (args.paper / "figures").mkdir(parents=True, exist_ok=True)
    (args.paper / "results").mkdir(parents=True, exist_ok=True)
    method_figure(args.paper / "figures/method-overview.pdf")
    if (args.run / "results.json").exists():
        result = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
        dataset = json.loads((args.run / "dataset.json").read_text(encoding="utf-8"))
        results_figure(result, args.paper / "figures/workload-diagnostic.pdf")
        latex_results(result, args.paper / "results/prototype-summary.tex", args.run.name, dataset)


if __name__ == "__main__":
    main()
