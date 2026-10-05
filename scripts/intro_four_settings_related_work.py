from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "intro_four_settings_related_work"
REGULAR = FontProperties(fname=r"C:\Windows\Fonts\comic.ttf")
BOLD = FontProperties(fname=r"C:\Windows\Fonts\comicbd.ttf")

INK = "#25363B"
MUTED = "#5B7077"
BLUE_NAME = "#245FC3"
GREEN = "#67B478"
GREEN_DARK = "#327944"
PALE_GREEN = "#DDEEDD"
LEAD = "#DCEFD9"
TASK_BLUE = "#C3E3F0"
GEN_BLUE = "#D8EDF5"
ORANGE = "#E5AB43"
FRAME = "#BCD8C0"


def text(ax, x, y, value, size=19, color=INK, bold=False, ha="left"):
    ax.text(x, y, value, fontsize=size, color=color, ha=ha, va="center",
            fontproperties=BOLD if bold else REGULAR)


def box(ax, x, y, w, h, fill, edge=INK, lw=1.6, radius=0):
    if radius:
        shape = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw)
    else:
        shape = Rectangle((x, y), w, h, facecolor=fill,
                          edgecolor=edge, linewidth=lw)
    ax.add_patch(shape)


def forecast_row(ax, n, y, title, timing, methods, lead_x, green_x, end_x,
                 focus=False, adjacent=None, title_size=23):
    if focus:
        box(ax, 49, y - 39, 1491, 93, "#F3FAF2", edge="#A7D1AE", lw=1.4,
            radius=10)
    ax.add_patch(Circle((77, y + 18), 18, facecolor="white",
                        edgecolor=GREEN_DARK, linewidth=1.8))
    text(ax, 77, y + 18, str(n), 18, GREEN_DARK, True, "center")
    text(ax, 110, y + 30, title, title_size, INK, True)
    text(ax, 110, y + 2, timing, 18, MUTED)
    text(ax, 110, y - 28, methods, 18, BLUE_NAME, True)
    if adjacent:
        text(ax, 480, y - 28, adjacent, 16, MUTED)

    h = 37
    box(ax, lead_x, y, green_x - lead_x, h, LEAD, edge=GREEN_DARK, lw=1.5)
    box(ax, green_x, y, end_x - green_x, h, GREEN, edge=GREEN_DARK, lw=1.5)


def warning(ax, x, y):
    ax.add_patch(Polygon([[x, y + 19], [x - 16, y - 10], [x + 16, y - 10]],
                         closed=True, facecolor="#F8D475",
                         edgecolor="#695226", linewidth=1.5))
    text(ax, x, y - 2, "!", 20, "#695226", True, "center")


def main():
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 10), dpi=160, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.axis("off")

    box(ax, 30, 35, 1540, 930, "#FFFFFF", edge=FRAME, lw=2.2, radius=17)
    box(ax, 30, 915, 1540, 50, PALE_GREEN, edge="none", lw=0, radius=15)
    text(ax, 800, 941, "Token consumption forecasts along one agent task", 27,
         INK, True, "center")

    box(ax, 400, 815, 1120, 50, TASK_BLUE, edge=INK, lw=1.8)
    text(ax, 960, 840, "Agent task", 24, INK, True, "center")
    for x, w, label in [
        (400, 250, "Generation 1"),
        (720, 280, "Generation 2"),
        (1230, 290, "Generation K"),
    ]:
        box(ax, x, 743, w, 46, GEN_BLUE, edge=INK, lw=1.5)
        text(ax, x + w / 2, 766, label, 20, INK, ha="center")
    ax.add_patch(FancyArrowPatch((655, 766), (713, 766), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.5, color=MUTED))
    text(ax, 684, 800, "tool feedback", 14, MUTED, ha="center")
    text(ax, 1114, 766, "···", 28, MUTED, ha="center")

    text(ax, 59, 703, "REQUEST LEVEL PRIOR WORK", 17, GREEN_DARK, True)
    forecast_row(ax, 1, 632, "Response length", "before generation",
                 "S³, SSJF, TIE", 690, 720, 1000)
    forecast_row(ax, 2, 527, "Remaining response length", "during generation",
                 "TRAIL, PLP", 850, 880, 1000)

    ax.plot([58, 1538], [465, 465], color="#C9DCCE", linewidth=1.2)
    text(ax, 59, 439, "AGENT TASK FORECASTS", 17, GREEN_DARK, True)
    forecast_row(ax, 3, 365, "Total task tokens", "before task execution",
                 "Self-Prediction (input + output)", 370, 400, 1520,
                 title_size=21)
    forecast_row(ax, 4, 255, "Remaining task tokens", "after tool feedback",
                 "TokenCast (input + output)", 750, 780, 1520, focus=True,
                 adjacent="Related: Chimera, Heddle")

    box(ax, 60, 72, 710, 105, "#FFF9EA", edge="#DDC784", lw=1.3, radius=11)
    warning(ax, 95, 128)
    text(ax, 125, 147, "Declared workflow schedulers", 18, INK, True)
    text(ax, 125, 120, "DSPy, Parrot, SGLang", 18, BLUE_NAME, True)
    text(ax, 125, 93, "Dependency relations are supplied before runtime", 16, MUTED)

    box(ax, 790, 72, 750, 105, "#F2F8FB", edge="#BED4DD", lw=1.3, radius=11)
    warning(ax, 825, 128)
    text(ax, 855, 147, "Related online forecasts", 18, INK, True)
    text(ax, 855, 120, "Chimera, Heddle", 18, BLUE_NAME, True)
    text(ax, 855, 93, "Workflow output or RL trajectory length", 16, MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), facecolor="white", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
