"""Widescreen banner for the README hero image / GitHub social preview.

A vertical roadmap of the docs book's 9 chapters (docs/_quarto.yml's own
chapter order and titles) beside a Rastrigin landscape computed and drawn
here (the same function as docs/images/rastrigin-landscape-hero.png), in
optimlab.viz.theme's colors. Chapters share one accent color: they are a
sequence, not categories, and nine items would cycle an 8-color palette.

CHAPTERS is hand-copied from docs/_quarto.yml / README.md's chapter list
and CATEGORICAL_LIGHT / CHART_CHROME from src/optimlab/viz/theme.py, since
importing optimlab here would pull in its full dependency stack (jax,
plotly, ...) for what is just a static label list. Keep in sync by hand.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# From src/optimlab/viz/theme.py — CHART_CHROME["light"] / CATEGORICAL_LIGHT.
PAGE = "#f9f9f7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRIDLINE = "#e1e0d9"
ACCENT = "#2a78d6"  # CATEGORICAL_LIGHT[0]
CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# (chapter number, short label) — from docs/_quarto.yml's chapter order.
CHAPTERS = [
    (1, "Foundations: Convexity & Gradients"),
    (2, "Linear Programming"),
    (3, "Least Squares"),
    (4, "Nonsmooth & Global Optimization"),
    (5, "Constraints & Duality"),
    (6, "Bayesian Modeling & Estimation"),
    (7, "High-Dimensional Non-Convexity"),
    (8, "Domain Applications"),
    (9, "Cross-Domain & the Solver Arena"),
]

FONT = "Helvetica Neue, Arial, sans-serif"


def rounded_rect(ax, x, y, w, h, color, *, rounding=0.08, alpha=1.0, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=0, facecolor=color, alpha=alpha, zorder=zorder,
        mutation_aspect=1,
    ))


def main() -> None:
    plt.rcParams["font.family"] = FONT

    # 12.8 x 6.4in @ 200dpi = 2560x1280px, 2x GitHub's recommended 1280x640
    # social preview size (retina-sharp, GitHub downsamples).
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Rastrigin landscape, computed here and drawn as a 3D surface on the right,
    # fully inside the canvas on a transparent background.
    ax3 = fig.add_axes([0.46, -0.08, 0.56, 0.95], projection="3d")
    ax3.set_facecolor((0, 0, 0, 0))
    g = np.linspace(-5.12, 5.12, 260)
    X, Y = np.meshgrid(g, g)
    Z = 20 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y)
    cmap = LinearSegmentedColormap.from_list("blues", ["#dbe8f8", ACCENT, "#14457f"])
    ax3.plot_surface(X, Y, Z, cmap=cmap, rstride=2, cstride=2, linewidth=0,
                     antialiased=True, alpha=0.95)
    ax3.view_init(elev=28, azim=-58)
    ax3.set_box_aspect((1, 1, 0.5), zoom=1.02)
    ax3.set_axis_off()

    # Header.
    ax.text(0.45, 0.62, "optimization-lab", fontsize=25, fontweight="bold",
             color=INK, ha="left", va="top", zorder=3)
    ax.text(0.45, 1.28, "A living lab for applied mathematical optimization — "
                        "from-scratch solvers, real comparisons",
            fontsize=12.5, color=MUTED, ha="left", va="top", zorder=3)

    # Vertical chapter roadmap, left column.
    top, bottom, gap = 1.85, 6.05, 0.12
    n = len(CHAPTERS)
    chip_h = (bottom - top - (n - 1) * gap) / n
    chip_x, chip_w = 0.45, 5.9
    y = top
    for num, label in CHAPTERS:
        rounded_rect(ax, chip_x, y, chip_w, chip_h, "#ffffff", rounding=chip_h / 2.2, zorder=3)
        ax.add_patch(FancyBboxPatch((chip_x, y), chip_w, chip_h,
                                    boxstyle=f"round,pad=0,rounding_size={chip_h / 2.2}",
                                    linewidth=1.0, edgecolor=GRIDLINE, facecolor="none",
                                    zorder=3, mutation_aspect=1))
        ax.add_patch(plt.Circle((chip_x + chip_h / 2 + 0.04, y + chip_h / 2), chip_h * 0.36,
                                color=ACCENT, zorder=4))
        ax.text(chip_x + chip_h / 2 + 0.04, y + chip_h / 2, f"{num}", fontsize=9.5,
                 fontweight="bold", color="white", ha="center", va="center", zorder=5)
        ax.text(chip_x + chip_h + 0.12, y + chip_h / 2, label, fontsize=10,
                 color=INK, ha="left", va="center", zorder=5)
        y += chip_h + gap

    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "social-preview.png"
    fig.savefig(str(out), dpi=200, facecolor=PAGE, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
