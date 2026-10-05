from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "intro_four_settings_argument"
REGULAR = FontProperties(fname=r"C:\Windows\Fonts\comic.ttf")
BOLD = FontProperties(fname=r"C:\Windows\Fonts\comicbd.ttf")

INK = "#263238"
GRAY = "#52646B"
BLUE = "#275FC3"
GREEN = "#36844E"
TARGET = "#A9D7B2"
PALE_GREEN = "#E1EFE1"
FRAME = "#B7D5BB"
BLUE_BAR = "#CCE6F2"
YELLOW = "#FFF5DE"
YELLOW_EDGE = "#D8B45E"


def t(ax, x, y, s, size=24, color=INK, bold=False, ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            fontproperties=BOLD if bold else REGULAR)


def box(ax, x, y, w, h, fill="white", edge=INK, lw=1.6, radius=0):
    if radius:
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw)
    else:
        patch = Rectangle((x, y), w, h, facecolor=fill,
                          edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)


def arrow(ax, start, end, color=INK, lw=2, dashed=False, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
        mutation_scale=19, linewidth=lw, color=color,
        linestyle=(0, (5, 4)) if dashed else "solid"))


def span(ax, start_x, end_x, y, h=30):
    box(ax, start_x, y, end_x - start_x, h, TARGET, GREEN, 1.5)
    ax.plot([start_x, start_x], [y - 5, y + h + 5], color=GREEN,
            linewidth=2.7, solid_capstyle="round")


def badge(ax, x, y, number):
    ax.add_patch(Circle((x, y), 19, facecolor="white",
                        edgecolor=GREEN, linewidth=2))
    t(ax, x, y, str(number), 23, GREEN, True, "center")


def warning(ax, x, y):
    ax.add_patch(Polygon([[x, y + 20], [x - 18, y - 12], [x + 18, y - 12]],
                         closed=True, facecolor="#F8D779",
                         edgecolor="#765B2B", linewidth=1.7))
    t(ax, x, y - 2, "!", 19, "#765B2B", True, "center")


def row(ax, number, title_y, title, methods, scope, start_x, end_x):
    badge(ax, 544, title_y - 16, number)
    t(ax, 575, title_y, title, 23, INK, True)
    t(ax, 575, title_y - 29, methods, 19, BLUE, True)
    warning(ax, 1430, title_y)
    t(ax, 1460, title_y, scope, 17, "#A65D37")
    span(ax, start_x, end_x, title_y - 66)


def main():
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18, 10.2), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1800)
    ax.set_ylim(0, 1020)
    ax.axis("off")


    box(ax, 70, 846, 300, 105, "#FFF9ED", "#DFC98F", 1.7, 14)
    t(ax, 220, 915, "User task", 30, INK, True, "center")
    t(ax, 220, 871, "task description", 22, GRAY, ha="center")
    arrow(ax, (220, 845), (220, 773))
    box(ax, 90, 655, 260, 118, "white", INK, 2, 12)
    t(ax, 220, 719, "LLM agent", 31, INK, True, "center")
    t(ax, 220, 675, "calls + tools", 22, GRAY, ha="center")
    arrow(ax, (350, 714), (460, 714))
    box(ax, 70, 476, 300, 91, "#EAF4F9", "#B6CFDB", 1.6, 12)
    t(ax, 220, 525, "Tool feedback", 27, INK, True, "center")
    t(ax, 220, 491, "changes the next step", 19, GRAY, ha="center")
    arrow(ax, (220, 567), (220, 648), color=GRAY, lw=1.8)


    box(ax, 470, 75, 1275, 890, "white", FRAME, 2.4, 18)
    box(ax, 470, 910, 1275, 55, PALE_GREEN, "none", 0, 16)
    t(ax, 1107, 938, "Four forecasting settings across an agent run", 29,
      INK, True, "center")
    t(ax, 1640, 888, "illustrative run", 18, GRAY, ha="right")
    box(ax, 780, 825, 900, 43, BLUE_BAR, INK, 1.7)
    t(ax, 1230, 846.5, "Agent task", 24, INK, True, "center")
    for x, w, label in [
        (780, 200, "Generation 1"),
        (1090, 250, "Generation 2"),
        (1510, 170, "Generation K"),
    ]:
        box(ax, x, 766, w, 38, BLUE_BAR, INK, 1.4)
        t(ax, x + w / 2, 785, label, 19, INK, ha="center")
    arrow(ax, (988, 785), (1078, 785), GRAY, 1.5)
    t(ax, 1035, 811, "tool result", 16, GRAY, ha="center")
    t(ax, 1432, 785, "···", 28, GRAY, ha="center")


    box(ax, 505, 405, 1205, 340, "#FFFFFF", "#C7D7CC", 1.5, 13)
    t(ax, 530, 718, "Prior forecasting settings", 24, GREEN, True)
    box(ax, 1460, 714, 30, 13, TARGET, GREEN, 1.2)
    t(ax, 1500, 720, "forecasted span", 16, GRAY)
    row(ax, 1, 684, "Response length before generation",
        "S³ · SSJF · TIE", "single response", 1090, 1340)
    row(ax, 2, 586, "Remaining response during generation",
        "TRAIL · PLP", "single response", 1220, 1340)
    row(ax, 3, 488, "Total task tokens before execution",
        "Self-Prediction", "one-time estimate", 780, 1680)


    box(ax, 505, 323, 1205, 71, YELLOW, YELLOW_EDGE, 1.4, 10)
    warning(ax, 540, 358)
    t(ax, 577, 369, "Future calls and repeated input depend on tool feedback",
      22, INK, True)
    t(ax, 577, 340, "Declared workflows (DSPy · Parrot · SGLang) specify dependencies in advance",
      18, GRAY)
    arrow(ax, (1630, 405), (1630, 394), GREEN, 1.7)
    arrow(ax, (1630, 323), (1630, 310), GREEN, 1.7)
    arrow(ax, (370, 522), (505, 358), GRAY, 1.7, dashed=True)

    box(ax, 505, 155, 1205, 155, "#F2FAF2", "#85C091", 2, 12)
    badge(ax, 544, 261, 4)
    t(ax, 577, 277, "Remaining task tokens during execution", 25,
      INK, True)
    t(ax, 577, 246, "TokenCast", 23, BLUE, True)
    t(ax, 577, 179, "Observed calls and feedback inform future input + output", 19,
      GREEN)
    span(ax, 990, 1680, 210, 32)
    t(ax, 550, 119,
      "Forecast the remaining input and output tokens as the agent task unfolds.",
      22, GREEN, True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), facecolor="white", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
