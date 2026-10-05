from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "intro_four_settings_ppt_inspired"
HAND = FontProperties(fname=r"C:\Windows\Fonts\comic.ttf")
HAND_BOLD = FontProperties(fname=r"C:\Windows\Fonts\comicbd.ttf")
SANS = FontProperties(fname=r"C:\Windows\Fonts\arial.ttf")
SANS_BOLD = FontProperties(fname=r"C:\Windows\Fonts\arialbd.ttf")

INK = "#243039"
SECONDARY = "#51646B"
BLUE = "#2564B6"
BLUE_FILL = "#D8EDF7"
GREEN = "#33824C"
GREEN_FILL = "#A9DAB7"
GREEN_FRAME = "#B7D9BA"
GREEN_PALE = "#EFF8EF"
YELLOW_FILL = "#FFF5DD"
YELLOW_EDGE = "#D6AB52"
ORANGE = "#A9672C"
GRAY_FILL = "#E5E9E9"
GUIDE = "#B9CCBE"


def label(ax, x, y, value, size=25, color=INK, family="sans", bold=False,
          ha="left", va="center"):
    prop = ({("sans", False): SANS, ("sans", True): SANS_BOLD,
             ("hand", False): HAND, ("hand", True): HAND_BOLD}
            [(family, bold)])
    ax.text(x, y, value, fontsize=size, color=color, fontproperties=prop,
            ha=ha, va=va)


def box(ax, x, y, w, h, fill="white", edge=INK, lw=1.5, radius=0):
    if radius:
        shape = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw)
    else:
        shape = Rectangle((x, y), w, h, facecolor=fill,
                          edgecolor=edge, linewidth=lw)
    ax.add_patch(shape)


def arrow(ax, a, b, color=INK, lw=2.0, dashed=False, scale=21):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
        mutation_scale=scale, linewidth=lw, color=color,
        linestyle=(0, (6, 5)) if dashed else "solid"))


def badge(ax, x, y, value, fill="white", edge=GREEN):
    ax.add_patch(Circle((x, y), 24, facecolor=fill,
                        edgecolor=edge, linewidth=2.4))
    label(ax, x, y + 1, str(value), 27, edge, "hand", True, "center")


def warning(ax, x, y, scale=1):
    pts = [[x, y + 18*scale], [x - 17*scale, y - 13*scale],
           [x + 17*scale, y - 13*scale]]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#F8D775",
                         edgecolor="#765A27", linewidth=1.5))
    label(ax, x, y - 2*scale, "!", 19*scale, "#765A27",
          "sans", True, "center")


def forecast(ax, start, end, y, h=34):
    box(ax, start, y, end-start, h, GREEN_FILL, GREEN, 1.8)
    ax.plot([start, start], [y-8, y+h+8], color=GREEN,
            lw=3.0, solid_capstyle="round")


def prior_row(ax, n, y, title, timing, methods, scope, start, end):
    box(ax, 500, y, 1780, 107, "white", "#C8DACA", 1.5, 12)
    for x in [1050, 1305, 1390, 1535, 1690, 2230]:
        ax.plot([x, x], [y+8, y+99], color="#E1EAE2",
                linewidth=1, linestyle=(0, (4, 5)))
    badge(ax, 545, y+54, n)
    label(ax, 590, y+76, title, 30, INK, "hand", True)
    label(ax, 590, y+35, timing, 23, SECONDARY)
    label(ax, 1190, y+78, methods, 23, BLUE, "hand", True)
    warning(ax, 2030, y+78, 0.85)
    label(ax, 2060, y+78, scope, 21, ORANGE)
    forecast(ax, start, end, y+20, 27)


def main():
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(24, 13), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2400)
    ax.set_ylim(0, 1300)
    ax.axis("off")


    box(ax, 75, 1010, 320, 120, "#FFFAEE", "#DCC891", 1.8, 14)
    label(ax, 235, 1080, "User task", 37, INK, "hand", True, "center")
    label(ax, 235, 1038, "task description", 25, SECONDARY,
          "sans", False, "center")
    arrow(ax, (235, 1008), (235, 875))
    box(ax, 75, 725, 320, 150, "white", INK, 2.1, 13)
    label(ax, 235, 815, "LLM agent", 39, INK, "hand", True, "center")
    label(ax, 235, 758, "model calls + tools", 24, SECONDARY,
          "sans", False, "center")
    arrow(ax, (235, 722), (235, 625), SECONDARY)
    box(ax, 75, 475, 320, 150, "#EDF6FA", "#AFCEDC", 1.8, 13)
    label(ax, 235, 570, "Tool feedback", 29, INK,
          "hand", True, "center")
    label(ax, 235, 515, "changes the next call", 22, SECONDARY,
          "sans", False, "center")
    ax.plot([75, 38, 38, 72], [552, 552, 800, 800],
            color=SECONDARY, linewidth=1.8)
    arrow(ax, (40, 800), (73, 800), SECONDARY, 1.8, False, 17)
    arrow(ax, (397, 551), (480, 419), SECONDARY, 1.8, True, 18)


    box(ax, 450, 80, 1880, 1160, "white", GREEN_FRAME, 2.7, 20)
    box(ax, 450, 1158, 1880, 82, "#DCEEDD", "none", 0, 18)
    label(ax, 1390, 1199, "Four forecasting settings across an agent run",
          42, INK, "hand", True, "center")
    box(ax, 500, 900, 1780, 235, "white", "#CAD9CE", 1.5, 14)
    label(ax, 545, 1094, "One task", 30, INK, "hand", True)
    label(ax, 545, 977, "Model calls", 29, INK, "hand", True)
    box(ax, 1050, 1045, 1180, 47, BLUE_FILL, INK, 1.7)
    label(ax, 1640, 1069, "Agent task", 28, INK, "hand", True, "center")
    for x, width, title in [(1050, 240, "Call 1"),
                            (1390, 300, "Call 2"),
                            (2035, 195, "Call K")]:
        box(ax, x, 947, width, 58, BLUE_FILL, INK, 1.6)
        label(ax, x+width/2, 976, title, 27, INK, "hand", False, "center")
    arrow(ax, (1300, 977), (1374, 977), SECONDARY, 1.5, False, 17)
    label(ax, 1338, 1020, "tool result", 18, SECONDARY,
          "sans", False, "center")
    label(ax, 1863, 974, "···", 35, SECONDARY,
          "hand", False, "center")

    label(ax, 520, 871, "Prior forecasting settings", 30,
          GREEN, "hand", True)
    box(ax, 1960, 864, 36, 19, GREEN_FILL, GREEN, 1.2)
    label(ax, 2008, 874, "forecasted span", 20, SECONDARY)

    prior_row(ax, 1, 745, "Response length", "before generation",
              "S³ · SSJF · TIE", "single response", 1390, 1690)
    prior_row(ax, 2, 625, "Remaining response", "during generation",
              "TRAIL · PLP", "single response", 1535, 1690)
    prior_row(ax, 3, 505, "Total task tokens", "before execution",
              "Self-Prediction", "task-start only", 1050, 2230)


    box(ax, 500, 380, 1780, 105, YELLOW_FILL, YELLOW_EDGE, 1.8, 12)
    warning(ax, 548, 433, 1.15)
    label(ax, 600, 449,
          "Future calls and their input context depend on tool feedback",
          30, INK, "hand", True)
    label(ax, 600, 407,
          "Declared workflows (DSPy · Parrot · SGLang) specify dependencies in advance",
          23, SECONDARY)
    arrow(ax, (2190, 502), (2190, 487), GREEN, 2.0, False, 19)
    arrow(ax, (2190, 379), (2190, 360), GREEN, 2.0, False, 19)


    box(ax, 500, 155, 1780, 205, GREEN_PALE, "#83BC8C", 2.2, 14)
    badge(ax, 545, 309, 4, GREEN_PALE)
    label(ax, 590, 315, "Remaining task tokens", 33, INK,
          "hand", True)
    label(ax, 590, 273, "during agent execution", 26, SECONDARY)
    label(ax, 590, 209, "TokenCast", 34, BLUE, "hand", True)
    box(ax, 1050, 230, 255, 43, GRAY_FILL, "#809094", 1.6)
    forecast(ax, 1305, 2230, 230, 43)
    label(ax, 1178, 198, "observed", 20, SECONDARY,
          "sans", False, "center")
    label(ax, 1768, 198, "future input + output", 21, GREEN,
          "sans", True, "center")
    label(ax, 525, 111,
          "Observed execution continually revises the remaining-task forecast",
          32, GREEN, "hand", True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), facecolor="white", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
