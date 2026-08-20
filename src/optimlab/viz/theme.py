"""Shared Plotly styling: a validated colorblind-safe categorical palette for telling
solvers apart, a single-hue sequential ramp for loss/objective magnitude, and a light/dark
layout template. Every figure-building function in `optimlab.viz` goes through this
module rather than picking colors ad hoc, so a "solver race" plot and a convergence plot
always mean the same thing by the same color.

Palette values and the light/dark chart-chrome tokens come from Anthropic's `dataviz`
skill reference palette (colorblind-validated categorical order; run
`node scripts/validate_palette.js` in that skill to re-check if you change any hex).
"""

from __future__ import annotations

# Fixed categorical order — never re-cycle or reassign per plot. Validated (light mode):
# worst adjacent CVD deltaE 9.1, worst adjacent normal-vision deltaE 19.6 (target >=8 / >=15).
# Slot 1 (blue) doubles as the sequential ramp's hue, so trajectory/line plots that also
# carry a blue sequential background (e.g. a contour plot) should start assigning solver
# colors from slot 2 (orange) instead — see `contrasting_categorical()` below.
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue   (reserve when a blue sequential surface is in the same figure)
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]

# Beyond ~4 series, per-pair colorblind separation can no longer be guaranteed by hue
# alone (see palette.md) — every figure that can show many solvers at once (convergence
# comparisons, solver races) must keep the legend visible and rely on Plotly's hover
# tooltip to carry the solver name, so identity is never color-alone even past that point.
MAX_RELIABLE_CATEGORICAL = 4

# Sequential (blue, light -> dark), used for loss/objective-value surfaces and contours.
SEQUENTIAL_BLUE_LIGHT = [
    (0.00, "#cde2fb"), (0.14, "#b7d3f6"), (0.28, "#9ec5f4"), (0.42, "#86b6ef"),
    (0.57, "#6da7ec"), (0.71, "#5598e7"), (0.85, "#3987e5"), (1.00, "#0d366b"),
]

CHART_CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "gridline": "#2c2c2a",
        "axis": "#383835",
    },
}


def categorical_colors(dark: bool = False) -> list[str]:
    """The 8-slot categorical order. Index into this directly so a given solver name
    always maps to the same slot across figures, rather than getting whatever's left
    after other series were assigned (see `optimlab.viz.compare.solver_color_map`).
    """
    return CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT


def contrasting_categorical(dark: bool = False) -> list[str]:
    """Categorical order starting at slot 2 (orange), for lines drawn over a blue
    sequential surface (e.g. trajectories over a loss contour) so the first series
    doesn't camouflage into the background.
    """
    colors = categorical_colors(dark)
    return colors[1:] + colors[:1]


def sequential_blue() -> list[tuple[float, str]]:
    """A Plotly-ready colorscale (list of (position, hex) stops) for loss/objective
    magnitude — pass directly as `colorscale=` to `go.Contour`/`go.Surface`.
    """
    return SEQUENTIAL_BLUE_LIGHT


def layout_template(dark: bool = False, **overrides) -> dict:
    """Base `go.Figure(layout=...)` kwargs shared by every chart in this module.

    Dict-valued overrides (`legend=dict(...)`, `margin=dict(...)`, ...) are shallow-merged
    into the matching default rather than replacing it outright, so a caller can move e.g.
    just the legend's `x`/`y` without losing its default `bgcolor`/`font`.
    """
    chrome = CHART_CHROME["dark" if dark else "light"]
    layout = {
        "paper_bgcolor": chrome["page"],
        "plot_bgcolor": chrome["surface"],
        "font": {
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": chrome["text_primary"],
            "size": 13,
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": chrome["text_secondary"]},
        },
        "xaxis": {
            "gridcolor": chrome["gridline"],
            "linecolor": chrome["axis"],
            "zerolinecolor": chrome["gridline"],
            "title_font": {"color": chrome["text_secondary"]},
            "tickfont": {"color": chrome["muted"]},
        },
        "yaxis": {
            "gridcolor": chrome["gridline"],
            "linecolor": chrome["axis"],
            "zerolinecolor": chrome["gridline"],
            "title_font": {"color": chrome["text_secondary"]},
            "tickfont": {"color": chrome["muted"]},
        },
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout
