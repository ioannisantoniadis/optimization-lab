"""Widescreen banner for the README hero image / GitHub social preview.

A vertical roadmap of the docs book's 9 chapters (docs/_quarto.yml's own
chapter order and titles), colored with optimlab.viz.theme's validated
categorical palette, with the existing Rastrigin-landscape hero render
(docs/images/rastrigin-landscape-hero.png) faded in on the right — reusing
this repo's own art and color system rather than inventing a new one.

CHAPTERS is hand-copied from docs/_quarto.yml / README.md's chapter list
and CATEGORICAL_LIGHT / CHART_CHROME from src/optimlab/viz/theme.py, since
importing optimlab here would pull in its full dependency stack (jax,
plotly, ...) for what is just a static label list. Keep in sync by hand.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# From src/optimlab/viz/theme.py — CHART_CHROME["light"] / CATEGORICAL_LIGHT.
PAGE = "#f9f9f7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRIDLINE = "#e1e0d9"
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

    # Faded Rastrigin-landscape hero render, bottom-right, behind everything.
    hero_path = Path(__file__).resolve().parents[1] / "docs" / "images" / "rastrigin-landscape-hero.png"
    if hero_path.exists():
        img = plt.imread(str(hero_path))
        src_w, src_h = 6.6, 6.6 * img.shape[0] / img.shape[1]
        x0 = 12.8 - src_w + 0.5
        y0 = 6.4 - src_h + 0.35
        ax.imshow(img, extent=(x0, x0 + src_w, y0 + src_h, y0), alpha=0.8, zorder=1)

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
    chip_x, chip_w = 0.45, 6.35
    y = top
    for num, label in CHAPTERS:
        color = CATEGORICAL[(num - 1) % len(CATEGORICAL)]
        rounded_rect(ax, chip_x, y, chip_w, chip_h, color,
                     rounding=chip_h / 2.2, alpha=0.94, zorder=3)
        ax.text(chip_x + 0.22, y + chip_h / 2, f"{num}", fontsize=10.5,
                 fontweight="800", color="white", ha="left", va="center", zorder=4)
        ax.text(chip_x + 0.62, y + chip_h / 2, label, fontsize=9.6,
                 fontweight="600", color="white", ha="left", va="center", zorder=4)
        y += chip_h + gap

    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "social-preview.png"
    fig.savefig(str(out), dpi=200, facecolor=PAGE, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
