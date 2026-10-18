"""Figures that show a derivative-free search's *internal state*, not just its best
point: a population method's whole swarm or generation at a few chosen iterations, and
Nelder-Mead's simplex as it reflects, expands, contracts and shrinks. Both need the
per-iteration history a solver records only when asked (`record_history=True`, stored
in `OptimizeResult.history`).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimlab.core import ArrayLike, OptimizeResult, Problem
from optimlab.viz.landscape import _display_z, _evaluate_grid, _resolve_domain
from optimlab.viz.theme import contrasting_categorical, layout_template, sequential_blue


def _contour_trace(
    problem: Problem, x_range: tuple[float, float], y_range: tuple[float, float], resolution: int
) -> go.Contour:
    X, Y, Z = _evaluate_grid(problem.f, x_range, y_range, resolution)
    Z_display, _title, _z_min = _display_z(Z, True)
    return go.Contour(
        x=X[0], y=Y[:, 0], z=Z_display, colorscale=sequential_blue(), showscale=False,
        ncontours=18, contours={"showlines": True, "coloring": "fill"},
        line={"width": 0.5, "color": "rgba(11,11,11,0.3)"}, hoverinfo="skip",
    )


def population_snapshots_figure(
    problem: Problem,
    result: OptimizeResult,
    iterations: list[int],
    *,
    title: str = "",
    domain: tuple[float, float] | None = None,
    resolution: int = 90,
    dark: bool = False,
) -> go.Figure:
    """The whole population at a few chosen iterations, one contour panel each — the
    picture that shows *how* a swarm or a genetic algorithm searches (spread out over
    the region first, then gathering into one basin), which a best-point-only
    trajectory can't. Needs `result.history["population"]` (pass `record_history=True`
    to `particle_swarm` / `genetic_algorithm`).
    """
    if "population" not in result.history:
        raise ValueError("result has no population history; rerun with record_history=True")
    populations = result.history["population"]
    iterations = [min(i, len(populations) - 1) for i in iterations]
    color = contrasting_categorical(dark=dark)[0]
    low, high = _resolve_domain(problem, domain)

    fig = make_subplots(rows=1, cols=len(iterations), horizontal_spacing=0.03,
                        subplot_titles=[f"iteration {i}" for i in iterations])
    contour = _contour_trace(problem, (low, high), (low, high), resolution)
    for col, it in enumerate(iterations, start=1):
        fig.add_trace(contour, row=1, col=col)
        pop = np.asarray(populations[it])
        fig.add_trace(
            go.Scatter(
                x=pop[:, 0], y=pop[:, 1], mode="markers", name="population",
                showlegend=col == 1,
                marker={"size": 6, "color": color, "line": {"color": "#ffffff", "width": 0.6}},
                hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
            ),
            row=1, col=col,
        )
        if problem.minimum is not None:
            minima = np.atleast_2d(problem.minimum)
            fig.add_trace(
                go.Scatter(
                    x=minima[:, 0], y=minima[:, 1], mode="markers", name="global minimum",
                    showlegend=col == 1,
                    marker={"symbol": "star", "size": 12, "color": "#ffffff",
                            "line": {"color": "#0b0b0b", "width": 1.2}},
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
        fig.update_xaxes(range=[low, high], showticklabels=False, row=1, col=col)
        fig.update_yaxes(range=[low, high], showticklabels=False, row=1, col=col)

    fig.update_layout(
        **layout_template(
            dark=dark, title=title, height=300,
            legend={"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"},
            margin={"l": 20, "r": 20, "t": 70 if title else 40, "b": 30},
        )
    )
    return fig


_MOVE_COLOR_SLOT = {"reflect": 0, "expand": 1, "contract": 2, "shrink": 3}


def simplex_steps_figure(
    problem: Problem,
    result: OptimizeResult,
    *,
    n_steps: int = 15,
    domain: tuple[float, float] | None = None,
    resolution: int = 120,
    dark: bool = False,
) -> go.Figure:
    """Nelder-Mead's first `n_steps` simplices drawn over the contour, each triangle
    colored by the move that produced it (reflect / expand / contract / shrink), so
    the four rules can be seen at work: long expansions down a slope, contractions as
    the simplex nears the valley floor. Needs `result.history["simplex"]` (pass
    `record_history=True` to `nelder_mead`). 2D problems only.
    """
    if "simplex" not in result.history:
        raise ValueError("result has no simplex history; rerun with record_history=True")
    if problem.n_dim != 2:
        raise ValueError(f"simplex_steps_figure needs a 2D problem, got n_dim={problem.n_dim}")
    simplices = result.history["simplex"][: n_steps + 1]
    moves = ["start"] + list(result.history["move"][:n_steps])
    palette = contrasting_categorical(dark=dark)

    if domain is None:  # a square window around the drawn simplices, not the whole domain
        points = np.vstack(simplices)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        half = 0.6 * float(np.ptp(points, axis=0).max()) or 1.0
        x_range = (center[0] - half, center[0] + half)
        y_range = (center[1] - half, center[1] + half)
    else:
        x_range = y_range = domain

    fig = go.Figure(_contour_trace(problem, x_range, y_range, resolution))
    shown: set[str] = set()
    for k, (simplex, move) in enumerate(zip(simplices, moves)):
        tri: ArrayLike = np.vstack([simplex, simplex[:1]])
        color = "#0b0b0b" if move == "start" else palette[_MOVE_COLOR_SLOT[move]]
        fig.add_trace(
            go.Scatter(
                x=tri[:, 0], y=tri[:, 1], mode="lines", name=move, legendgroup=move,
                showlegend=move not in shown,
                line={"color": color, "width": 2 if move == "start" else 1.6},
                hovertemplate=f"step {k}: {move}<extra></extra>",
            )
        )
        shown.add(move)

    fig.update_layout(
        **layout_template(
            dark=dark, title=f"Nelder–Mead: the first {len(simplices) - 1} moves",
            xaxis_title="x₁", yaxis_title="x₂",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 120, "t": 50, "b": 50},
        )
    )
    fig.update_xaxes(range=list(x_range), constrain="domain")
    fig.update_yaxes(range=list(y_range), scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def bayes_opt_step_figure(
    f,
    X: ArrayLike,
    bounds: tuple[float, float],
    *,
    length_scale: float = 1.0,
    variance: float = 1.0,
    resolution: int = 300,
    dark: bool = False,
) -> go.Figure:
    """One step of Bayesian optimization on a 1D function, made visible: the evaluated
    points, the Gaussian-process surrogate's mean and ±2 std band (top), and the
    Expected Improvement acquisition it implies (bottom), whose maximizer is the next
    point to evaluate. `f` is drawn as a dotted line for reference only — the surrogate
    never sees it except at `X`.
    """
    from optimlab.optimizers.bayesian_optimization import expected_improvement, gp_posterior

    X = np.asarray(X, dtype=float).ravel()
    y = np.array([float(f(x)) for x in X])
    xs = np.linspace(bounds[0], bounds[1], resolution)
    post = gp_posterior(X[:, None], y, xs[:, None], length_scale=length_scale, variance=variance)
    ei = expected_improvement(post.mean, post.std, float(y.min()))
    next_x = float(xs[np.argmax(ei)])
    blue = "#3987e5" if dark else "#2a78d6"
    ink = "#ffffff" if dark else "#0b0b0b"
    accent = contrasting_categorical(dark=dark)[0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    band = np.r_[post.mean + 2 * post.std, (post.mean - 2 * post.std)[::-1]]
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=band, fill="toself", mode="lines",
                             fillcolor="rgba(42,120,214,0.15)", line={"width": 0},
                             name="surrogate mean ± 2 std", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=[float(f(x)) for x in xs], mode="lines",
                             name="true f (unknown to the method)",
                             line={"color": "#898781", "dash": "dot"}), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=post.mean, mode="lines", name="surrogate mean",
                             line={"color": blue, "width": 2}), 1, 1)
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="evaluated points",
                             marker={"size": 9, "color": ink}), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=ei, mode="lines", name="expected improvement",
                             line={"color": accent, "width": 2}, fill="tozeroy"), 2, 1)
    fig.add_vline(x=next_x, line={"color": accent, "dash": "dash", "width": 1.5})
    fig.update_layout(
        **layout_template(dark=dark, height=480,
                          legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
                          margin={"l": 60, "r": 230, "t": 30, "b": 50})
    )
    fig.update_yaxes(title_text="f(x)", row=1, col=1)
    fig.update_yaxes(title_text="EI(x)", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    return fig
