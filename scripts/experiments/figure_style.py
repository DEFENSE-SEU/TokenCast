from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Family:


    light: str
    base: str
    dark: str
    marker: str
    marker_size: float
    hollow: bool = False


OURS = Family("#9095CE", "#5C61A8", "#3B3F73", "o", 4.0)
GENERIC = Family("#93D2CF", "#4FB4B0", "#2E7C79", "s", 3.4)
PUBLISHED = Family("#E3A099", "#CE6A61", "#8E4038", "^", 3.8)
LLM = Family("#CBE3B2", "#A5CC80", "#6E9350", "D", 3.4)
REFERENCE = Family("#8C8480", "#5C534E", "#3B3532", "P", 3.8, hollow=True)

FAMILY = {"ours": OURS, "generic": GENERIC, "published": PUBLISHED, "llm": LLM, "reference": REFERENCE}

OURS_FILL = "#DCDDF2"
GRID = "#E9E9EE"
TEXT = "#26262E"
FRAME = "#3A3A44"
SEQUENTIAL = ["#EEEDF8", "#C7C7E3", "#98A1CB", "#667FB0", "#3D5B82"]

SOLID = (0, ())
DASH = (0, (4, 2))
DOT = (0, (1, 1.5))

WIDE = 6.75
COLUMN = 3.25
OURS_WIDTH = 1.6
OTHER_WIDTH = 1.1


_THIN_SAFE = {GENERIC.base: GENERIC.dark, LLM.base: LLM.dark}


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 3.5,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#C4C4CE",
        "legend.facecolor": "white",
        "legend.fancybox": False,
        "legend.borderpad": 0.32,
        "legend.labelspacing": 0.28,
        "legend.borderaxespad": 0.35,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0,
        "lines.linewidth": OTHER_WIDTH,
        "lines.markersize": 3.2,
        "axes.linewidth": 0.7,
        "axes.edgecolor": FRAME,
        "axes.labelcolor": TEXT,
        "xtick.color": FRAME,
        "ytick.color": FRAME,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "text.color": TEXT,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
    })


def grid_on(ax, axis: str = "y") -> None:

    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def line_kwargs(family: str, *, step: str = "base", reference: bool = False) -> dict:

    spec = FAMILY[family]
    colour = {"light": spec.light, "base": spec.base, "dark": spec.dark}[step]
    width = OURS_WIDTH if family == "ours" else OTHER_WIDTH
    if width < 1.5:
        colour = _THIN_SAFE.get(colour, colour)
    style = DASH if reference else SOLID
    kwargs = {
        "color": colour,
        "linewidth": width,
        "linestyle": style,
        "marker": spec.marker,
        "markersize": spec.marker_size,
        "markeredgewidth": 0.9 if spec.hollow else 0.0,
    }
    if spec.hollow:
        kwargs["markerfacecolor"] = "white"
        kwargs["markeredgecolor"] = colour
    return kwargs


def bar_kwargs(family: str, *, step: str = "base") -> dict:

    spec = FAMILY[family]
    colour = {"light": spec.light, "base": spec.base, "dark": spec.dark}[step]
    return {"color": colour, "edgecolor": spec.dark, "linewidth": 0.4}


def point_kwargs(family: str, *, hollow: bool = False, step: str = "base") -> dict:

    spec = FAMILY[family]
    colour = {"light": spec.light, "base": spec.base, "dark": spec.dark}[step]
    return {
        "marker": spec.marker,
        "markersize": spec.marker_size,
        "linestyle": "none",
        "color": colour,
        "markerfacecolor": "white" if hollow or spec.hollow else colour,
        "markeredgecolor": colour,
        "markeredgewidth": 0.9,
    }


OUR_INITIALIZER_POOL = {
    "input_output_scaled_interval", "input_output_workload_fusion", "difficulty_profile_residual_mapping",
    "difficulty_profile_scaled_interval", "input_workload_mapping", "difficulty_text_workload_mapping",
    "difficulty_workload_scaled_interval", "output_workload_mapping", "task_structure_workload_mapping",
    "workload_mapping", "problem_text_workload_mapping", "repo_text_workload_mapping",
    "workload_self_prediction_fusion", "cross_model_workload_mapping",
}


OUR_INITIALIZER_SUFFIX = "_self_prediction_mapping"


def family_of(candidate: str) -> str:

    if candidate in OUR_INITIALIZER_POOL or candidate.endswith(OUR_INITIALIZER_SUFFIX):
        return "ours"
    if candidate.endswith(("_ours", "_ours_clock", "_clock")) or candidate.startswith("in_call_ours") or candidate in {
        "input_output_scaled_interval", "in_call_prefix",
    } or "__learner_lightgbm" in candidate:
        return "ours"
    if candidate in {"trail_last", "trail_mean", "egtp_pool", "estp", "s3_encoder", "ssjf_reg", "ssjf_mc",
                     "magnus", "ltr_c", "tie_encoder", "tpv"} or candidate.startswith("trail_last_layer"):
        return "published"
    if candidate.startswith(("response_length_prompt", "progressive_interval", "self_estimate", "published_gpt52")):
        return "llm"
    if candidate in {"lightgbm_generic", "sequence_gru"} or "__learner_" in candidate:
        return "generic"
    return "reference"


def panel_title(ax, letter: str, text: str) -> None:

    ax.set_title(f"({letter}) {text}", fontsize=7.5, fontweight="bold", pad=3.5)


def legend_inside(ax, handles=None, labels=None, *, loc: str = "best", ncol: int = 1, **kwargs):

    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    return ax.legend(handles, labels, loc=loc, ncol=ncol, **kwargs)


def legend_bottom(fig, handles, labels, *, ncol: int, y: float = -0.02) -> None:

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=6.5, handletextpad=0.45, columnspacing=1.2)


def save(fig, output, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}")
    plt.close(fig)
    print(f"wrote {name}.pdf and {name}.png", flush=True)
