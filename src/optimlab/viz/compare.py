"""Multi-solver comparison figures: trajectories overlaid on a landscape ("solver race")
and convergence curves. This is the visual heart of the "why does method X beat method Y
here" story — see `notebooks/marimo/gradient_descent_explorer.py` for the interactive
version.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import OptimizeResult, Problem
from optimlab.viz.landscape import (
    _evaluate_grid,
    _resolve_domain,
    contour_figure,
    surface_figure,
    transform_values,
)
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


def add_trajectory_3d(
    fig: go.Figure,
    result: OptimizeResult,
    *,
    log_z: bool,
    z_min: float,
    name: str | None = None,
    color: str = "#eb6834",
) -> go.Figure:
    """Overlay one solver run's path onto an existing `surface_figure`, lifted onto the
    surface itself (`z = f(x)` at each visited point, run through the same display
    transform as the surface) rather than flattened onto the base plane — this is what
    actually shows a solver *descending*, which the 2D `add_trajectory` version can't.
    """
    traj = np.asarray(result.trajectory)
    if traj.shape[1] != 2:
        raise ValueError(f"add_trajectory_3d needs a 2D trajectory, got shape {traj.shape}")
    z = transform_values(result.f_trajectory, log_z, z_min)

    label = name or result.solver_name
    # A small vertical offset keeps the path from being visually swallowed by the
    # surface mesh it's sitting on (z-fighting), without perceptibly changing height.
    z_lift = z + 0.01 * (np.ptp(z) or 1.0)
    fig.add_trace(
        go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=z_lift,
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 5},
            marker={"size": 3, "color": color},
            hovertemplate=f"{label}<br>x=%{{x:.4f}}, y=%{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[z_lift[-1]],
            mode="markers",
            marker={"symbol": "circle", "size": 6, "color": color, "line": {"color": "#ffffff", "width": 1}},
            showlegend=False,
            hovertemplate=f"{label} final<br>x=%{{x:.4f}}, y=%{{y:.4f}}<extra></extra>",
        )
    )
    return fig


def surface_race_figure(
    problem: Problem,
    results: dict[str, OptimizeResult],
    *,
    resolution: int = 100,
    domain: tuple[float, float] | None = None,
    log_z: bool = True,
    dark: bool = False,
) -> go.Figure:
    """`race_figure`'s 3D counterpart: every solver's path, descending the actual
    surface rather than viewed from directly above. Slower to explore interactively
    (rotating a 3D scene is more friction than panning a 2D contour) but far more
    immediately legible for "how does this solver actually reach the minimum," which is
    exactly what a top-down-only view struggles to convey.
    """
    if problem.n_dim != 2:
        raise ValueError(f"surface_race_figure needs a 2D problem, got n_dim={problem.n_dim}")

    fig = surface_figure(problem, resolution=resolution, domain=domain, log_z=log_z, dark=dark)
    fig.data[0].update(opacity=0.92, showlegend=False)

    low, high = _resolve_domain(problem, domain)
    _, _, Z = _evaluate_grid(problem.f, (low, high), (low, high), resolution)
    z_min = float(Z.min()) if log_z else 0.0

    colors = solver_color_map(list(results), dark=dark)
    for name, result in results.items():
        add_trajectory_3d(fig, result, log_z=log_z, z_min=z_min, name=name, color=colors[name])

    fig.update_layout(title=f"{problem.name} — solver race (3D)")
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
    pad_converged: bool = True,
) -> go.Figure:
    """Objective value (or gradient norm) vs. iteration, one line per solver, single
    (log-scaled by default) y-axis — never a second axis: with solvers converging at
    very different rates, a shared log scale is what makes the comparison honest.

    `pad_converged=True` (default) extends any solver that converged before the others
    with a dotted line holding its final value out to the longest run in the group.
    Without this, a solver that converges in, say, 5 iterations next to one that takes
    60 draws as a stub that looks cut off rather than "done" — the dotted extension is
    what actually shows it reached and *stayed at* the minimum. A solver that hit
    `max_iter` without converging is never padded: its line honestly stops wherever it
    stopped, still mid-descent, which is real information worth keeping visible (see
    `tests/test_viz.py` for the case this guards against).
    """
    if metric not in {"f", "grad_norm"}:
        raise ValueError("metric must be 'f' or 'grad_norm'")
    attr = "f_trajectory" if metric == "f" else "grad_norm_trajectory"
    y_title = "objective value f(x)" if metric == "f" else "||grad f(x)||"

    colors = solver_color_map(list(results), dark=dark)
    max_len = max((len(getattr(r, attr)) for r in results.values()), default=0)

    fig = go.Figure()
    for name, result in results.items():
        series = np.asarray(getattr(result, attr), dtype=float)
        if log_y:
            series = np.maximum(series, 1e-16)  # keep log-scale well-defined near zero
        color = colors[name]

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(series)), y=series,
                mode="lines", name=name,
                line={"color": color, "width": 2},
                hovertemplate=f"{name}<br>iter=%{{x}}<br>{y_title}=%{{y:.3g}}<extra></extra>",
            )
        )

        if pad_converged and result.converged and len(series) < max_len:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(series) - 1, max_len), y=np.full(max_len - len(series) + 1, series[-1]),
                    mode="lines", name=name, showlegend=False,
                    line={"color": color, "width": 2, "dash": "dot"},
                    hovertemplate=f"{name} (converged, holding)<br>iter=%{{x}}<br>{y_title}=%{{y:.3g}}<extra></extra>",
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
