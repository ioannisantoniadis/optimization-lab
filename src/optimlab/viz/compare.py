"""Multi-solver comparison figures: trajectories overlaid on a landscape ("solver race")
and convergence curves. This is the visual heart of the "why does method X beat method Y
here" story — see `notebooks/marimo/gradient_descent_explorer.py` for the interactive
version.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import OptimizeResult, Problem
from optimlab.viz.landscape import contour_figure
from optimlab.viz.theme import MAX_RELIABLE_CATEGORICAL, contrasting_categorical, layout_template

_OTHER_COLOR = "#898781"  # muted gray — folded-in color past MAX_RELIABLE_CATEGORICAL


def solver_color_map(names: list[str], dark: bool = False) -> dict[str, str]:
    """Assign each solver name a fixed color from the categorical order, in the order
    given. Past `MAX_RELIABLE_CATEGORICAL` (4), later names fold to a shared muted gray —
    beyond that point per-pair colorblind separation isn't guaranteed by hue alone (see
    `theme.py`), so the legend + hover tooltip, not color, are what carry identity.
    """
    palette = contrasting_categorical(dark=dark)
    colors: dict[str, str] = {}
    for i, name in enumerate(names):
        colors[name] = palette[i % len(palette)] if i < MAX_RELIABLE_CATEGORICAL else _OTHER_COLOR
    return colors


def add_trajectory(
    fig: go.Figure,
    result: OptimizeResult,
    *,
    name: str | None = None,
    color: str = "#eb6834",
) -> go.Figure:
    """Overlay one solver run's path onto an existing 2D figure (typically a
    `contour_figure`). A white-outlined marker line keeps the path legible regardless of
    what's underneath it on the sequential blue background.
    """
    traj = np.asarray(result.trajectory)
    if traj.shape[1] != 2:
        raise ValueError(f"add_trajectory needs a 2D trajectory, got shape {traj.shape}")

    label = name or result.solver_name
    fig.add_trace(
        go.Scatter(
            x=traj[:, 0], y=traj[:, 1],
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 2},
            marker={"size": 5, "color": color, "line": {"color": "#ffffff", "width": 0.5}},
            hovertemplate=f"{label}<br>x=%{{x:.4f}}, y=%{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[traj[-1, 0]], y=[traj[-1, 1]],
            mode="markers",
            marker={"symbol": "circle", "size": 11, "color": color, "line": {"color": "#ffffff", "width": 1.5}},
            showlegend=False,
            hovertemplate=f"{label} final<br>x=%{{x:.4f}}, y=%{{y:.4f}}<extra></extra>",
        )
    )
    return fig


def race_figure(
    problem: Problem,
    results: dict[str, OptimizeResult],
    *,
    resolution: int = 150,
    domain: tuple[float, float] | None = None,
    log_z: bool = True,
    dark: bool = False,
) -> go.Figure:
    """The headline comparison plot: every solver's path, overlaid on the same contour.
    Same starting point in, wildly different paths out — this is usually the single
    figure that makes "gradient descent zig-zags, Newton doesn't" land intuitively.
    """
    fig = contour_figure(problem, resolution=resolution, domain=domain, log_z=log_z, dark=dark)
    colors = solver_color_map(list(results), dark=dark)
    for name, result in results.items():
        add_trajectory(fig, result, name=name, color=colors[name])
    fig.update_layout(title=f"{problem.name} — solver race")
    return fig


def convergence_figure(
    results: dict[str, OptimizeResult],
    *,
    metric: str = "f",
    log_y: bool = True,
    dark: bool = False,
) -> go.Figure:
    """Objective value (or gradient norm) vs. iteration, one line per solver, single
    (log-scaled by default) y-axis — never a second axis: with solvers converging at
    very different rates, a shared log scale is what makes the comparison honest.
    """
    if metric not in {"f", "grad_norm"}:
        raise ValueError("metric must be 'f' or 'grad_norm'")
    attr = "f_trajectory" if metric == "f" else "grad_norm_trajectory"
    y_title = "objective value f(x)" if metric == "f" else "||grad f(x)||"

    colors = solver_color_map(list(results), dark=dark)
    fig = go.Figure()
    for name, result in results.items():
        series = np.asarray(getattr(result, attr))
        if log_y:
            series = np.maximum(series, 1e-16)  # keep log-scale well-defined near zero
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(series)), y=series,
                mode="lines", name=name,
                line={"color": colors[name], "width": 2},
                hovertemplate=f"{name}<br>iter=%{{x}}<br>{y_title}=%{{y:.3g}}<extra></extra>",
            )
        )

    fig.update_layout(
        **layout_template(
            dark=dark,
            title="Convergence comparison",
            xaxis_title="iteration",
            yaxis_title=y_title,
            yaxis_type="log" if log_y else "linear",
        ),
    )
    return fig
