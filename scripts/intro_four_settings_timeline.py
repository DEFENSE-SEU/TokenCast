from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "intro_four_settings_timeline"

INK = "#20313A"
MUTED = "#59707A"
OUTLINE = "#294B59"
TASK_BLUE = "#B7DEEE"
GEN_BLUE = "#D1EAF3"
PREDICT = "#DDF0D8"
TARGET = "#64AF76"
TARGET_DARK = "#2F7946"
GUIDE = "#CCD9DD"


def txt(ax, x, y, value, size=18, color=INK, weight="normal", ha="left"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", fontfamily="DejaVu Sans")


def rect(ax, x, y, w, h, fill, line=OUTLINE, lw=1.45):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=line, linewidth=lw))


def prediction_row(ax, n, y, title, timing, marker_x, target_x, end_x):
    ax.add_patch(Circle((48, y + 18), 17, facecolor="white",
                        edgecolor=TARGET_DARK, linewidth=1.8))
    txt(ax, 48, y + 18, str(n), 18, TARGET_DARK, "bold", "center")
    txt(ax, 79, y + 28, title, 18, INK, "bold")
    txt(ax, 79, y + 4, timing, 16, MUTED)

    h = 36
    rect(ax, marker_x, y, target_x - marker_x, h, PREDICT,
         line=TARGET_DARK, lw=1.4)
    rect(ax, target_x, y, end_x - target_x, h, TARGET,
         line=TARGET_DARK, lw=1.4)


def main():
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14.5, 7.2), dpi=180, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1450)
    ax.set_ylim(0, 720)
    ax.axis("off")


    rect(ax, 365, 623, 1035, 47, TASK_BLUE, lw=1.7)
    txt(ax, 882.5, 646.5, "Agent task", 23, INK, "bold", "center")
    for x, w, label in [
        (365, 260, "Generation 1"),
        (690, 270, "Generation 2"),
        (1140, 260, "Generation K"),
    ]:
        rect(ax, x, 539, w, 45, GEN_BLUE)
        txt(ax, x + w / 2, 561.5, label, 19, INK, ha="center")
    ax.add_patch(FancyArrowPatch((630, 561), (682, 561), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.5, color=MUTED))
    txt(ax, 656, 601, "tool feedback", 13, MUTED, ha="center")
    txt(ax, 1050, 561.5, "···", 31, MUTED, ha="center")

    txt(ax, 36, 505, "ONE GENERATION", 15, MUTED, "bold")
    prediction_row(ax, 1, 437, "Response length", "before generation",
                   660, 690, 960)
    prediction_row(ax, 2, 365, "Remaining response length", "during generation",
                   815, 845, 960)

    ax.plot([35, 1400], [330, 330], color=GUIDE, linewidth=1.1)
    txt(ax, 36, 302, "ENTIRE AGENT TASK", 15, MUTED, "bold")
    prediction_row(ax, 3, 237, "Total task tokens", "before the task starts",
                   335, 365, 1400)
    prediction_row(ax, 4, 165, "Remaining task tokens", "while the agent runs",
                   515, 545, 1400)

    rect(ax, 601, 84, 27, 21, TARGET, line=TARGET_DARK, lw=1.2)
    txt(ax, 640, 94.5, "tokens being forecast", 16, MUTED)
    ax.add_patch(FancyArrowPatch((365, 45), (1400, 45), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.25, color=MUTED))
    txt(ax, 882.5, 22, "execution time", 15, MUTED, ha="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), facecolor="white", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
