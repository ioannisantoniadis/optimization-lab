"""Figures for `optimlab.optimizers.barrier_method`: the central path an interior point
solver actually walks, and the KKT stationarity condition it converges to, made
geometric rather than left as algebra. `duality_gap_figure` covers the third explicit
Phase 4 visual goal (a duality gap, over iterations) for both the barrier method's
`n_constraints / t` estimate and LP strong duality.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import OptimizeResult
from optimlab.optimizers.barrier_method import ConstrainedProblem
from optimlab.viz.landscape import _evaluate_grid
from optimlab.viz.theme import contrasting_categorical, layout_template, sequential_blue


def _resolve_domain(trajectory: np.ndarray, domain: tuple[float, float] | None) -> tuple[float, float]:
    if domain is not None:
        return domain
    low, high = float(trajectory.min()), float(trajectory.max())
    pad = max(high - low, 1.0) * 0.5
    return (low - pad, high + pad)


def central_path_figure(
    problem: ConstrainedProblem,
    result: OptimizeResult,
    *,
    resolution: int = 150,
    domain: tuple[float, float] | None = None,
    dark: bool = False,
) -> go.Figure:
    """`f`'s contour, the infeasible region shaded out, each constraint's boundary
    `g_i(x) = 0` drawn as a curve, and the barrier method's central path (one point per
    outer iteration — see `barrier_method`'s docstring for why not per Newton step)
    walking from `problem.x0` to the constrained optimum, always staying inside the
    feasible region as it goes.
    """
    if problem.n_dim != 2:
        raise ValueError(f"central_path_figure needs a 2D problem, got n_dim={problem.n_dim}")

    trajectory = np.asarray(result.trajectory)
    low, high = _resolve_domain(trajectory, domain)
    X, Y, Z = _evaluate_grid(problem.f, (low, high), (low, high), resolution)

    fig = go.Figure(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z,
            colorscale=sequential_blue(),
            ncontours=20,
            contours={"showlines": True, "coloring": "fill"},
            line={"width": 0.6, "color": "rgba(11,11,11,0.35)"},
            colorbar={"title": "f(x)", "len": 0.55, "y": 0.26, "x": 1.02},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{z:.3f}<extra></extra>",
        )
    )

    infeasible = np.zeros_like(Z, dtype=bool)
    for g in problem.inequality_constraints:
        _, _, Zg = _evaluate_grid(g, (low, high), (low, high), resolution)
        infeasible |= Zg > 0
        fig.add_trace(
            go.Contour(
                x=X[0], y=Y[:, 0], z=Zg,
                showscale=False, showlegend=False, hoverinfo="skip",
                contours={"start": 0, "end": 0, "size": 1, "coloring": "lines"},
                line={"width": 2, "color": "rgba(11,11,11,0.6)"},
            )
        )

    fig.add_trace(
        go.Contour(
            x=X[0], y=Y[:, 0], z=infeasible.astype(float),
            showscale=False, hoverinfo="skip",
            contours={"start": 0, "end": 1, "size": 0.5, "coloring": "fill"},
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(137,135,129,0.65)"]],
            line={"width": 0},
        )
    )

    color = contrasting_categorical(dark=dark)[0]
    fig.add_trace(
        go.Scatter(
            x=trajectory[:, 0], y=trajectory[:, 1],
            mode="lines+markers", name="central path",
            line={"color": color, "width": 2.5},
            marker={"size": 7, "color": color, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="outer iter %{pointNumber}<br>x=%{x:.4f}, y=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trajectory[-1, 0]], y=[trajectory[-1, 1]], mode="markers", name="optimum",
            marker={"symbol": "star", "size": 16, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
            hovertemplate=f"optimum<br>x=%{{x:.4f}}, y=%{{y:.4f}}<br>f={result.f:.4g}<extra></extra>",
        )
    )

    fig.update_layout(
        **layout_template(
            dark=dark, title=f"{problem.name} — central path", xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 140, "t": 50, "b": 50},
        ),
    )
    fig.update_xaxes(range=[low, high])
    fig.update_yaxes(range=[low, high], scaleanchor="x", scaleratio=1)
    return fig


def _add_arrow(
    fig: go.Figure, tail: np.ndarray, vector: np.ndarray, *, color: str, offset: np.ndarray | None = None
) -> None:
    """Draws the arrow from `tail + offset` to `tail + offset + vector` — `offset` (a
    small perpendicular nudge, unrelated to `vector`'s own direction/length) exists only
    so that multiple KKT vectors which are *exactly* parallel (the whole geometric point
    of the stationarity condition) don't render as a single arrow hiding the rest.
    """
    root = tail if offset is None else tail + offset
    head = root + vector
    fig.add_annotation(
        x=float(head[0]), y=float(head[1]), ax=float(root[0]), ay=float(root[1]),
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor=color,
    )


def kkt_geometry_figure(
    problem: ConstrainedProblem,
    result: OptimizeResult,
    *,
    resolution: int = 150,
    domain: tuple[float, float] | None = None,
    active_ratio: float = 5.0,
    dark: bool = False,
) -> go.Figure:
    """The KKT stationarity condition `grad f(x*) + sum_i lambda_i grad g_i(x*) = 0`,
    drawn as vectors rooted at the optimum: `-grad f(x*)` (red) sits exactly opposite the
    *sum* of the active constraints' outward normals, each scaled by its estimated
    multiplier `lambda_i` (blue, dashed gray for the sum) -- the geometric content behind
    "the objective can't decrease without leaving the feasible region."

    Multipliers come from the converged barrier problem itself: at the barrier method's
    last iterate, `t * grad f(x*) + sum_i grad g_i(x*) / s_i = 0` (`s_i = -g_i(x*)`), so
    `lambda_i = 1 / (t * s_i)` are already the exact KKT multipliers in the limit
    `t -> infinity` this run approximated. `t` itself isn't in `OptimizeResult`, so it's
    recovered from the final duality gap (`gap = n_constraints / t`, stored as this
    solver's `grad_norm_trajectory`, matching `barrier_method`'s own convention).
    `active_ratio` keeps only constraints whose slack `s_i` is within that factor of the
    smallest slack -- the ones actually pinning the optimum in place, not the many that
    happen to also be satisfied but nowhere near tight.
    """
    if problem.n_dim != 2:
        raise ValueError(f"kkt_geometry_figure needs a 2D problem, got n_dim={problem.n_dim}")

    x_star = np.asarray(result.x)
    final_gap = float(np.asarray(result.grad_norm_trajectory)[-1])
    t_final = problem.n_constraints / final_gap

    slacks = np.array([-float(g(x_star)) for g in problem.inequality_constraints])
    active = slacks < active_ratio * slacks.min()

    trajectory = np.asarray(result.trajectory)
    low, high = _resolve_domain(trajectory, domain)
    X, Y, Z = _evaluate_grid(problem.f, (low, high), (low, high), resolution)

    fig = go.Figure(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z, colorscale=sequential_blue(), opacity=0.85,
            ncontours=20, contours={"showlines": True, "coloring": "fill"},
            line={"width": 0.6, "color": "rgba(11,11,11,0.25)"}, showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_star[0]], y=[x_star[1]], mode="markers", name="optimum",
            marker={"symbol": "star", "size": 16, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
            hovertemplate="optimum<br>x=%{x:.4f}, y=%{y:.4f}<extra></extra>",
        )
    )

    palette = contrasting_categorical(dark=dark)
    grad_f = problem.grad(x_star)
    scale = 0.35 * (high - low) / max(float(np.linalg.norm(grad_f)), 1e-12)

    # All of these vectors are, by KKT stationarity, exactly parallel (that's the point
    # being illustrated) -- drawn from the identical root x_star they'd stack into one
    # indistinguishable arrow, so each gets a small perpendicular nudge purely for visual
    # separation. The nudge direction is perpendicular to -grad_f, not derived from any
    # of the vectors' own directions, so it never changes what's actually being shown.
    perp = np.array([-grad_f[1], grad_f[0]])
    perp_norm = np.linalg.norm(perp)
    perp = perp / perp_norm if perp_norm > 1e-12 else np.array([0.0, 1.0])
    step_unit = 0.05 * (high - low) * perp

    active_idx = [i for i, is_active in enumerate(active) if is_active]
    # Lay every series out along the perpendicular in a fixed order (-grad f, then one
    # slot per active constraint, then the sum last) so slot k is simply k * step_unit --
    # purely a rendering order, unrelated to each vector's actual magnitude or direction.
    series_slots = {"grad_f": 0, **{i: 1 + k for k, i in enumerate(active_idx)}, "sum": 1 + len(active_idx)}
    n_slots = len(series_slots)

    def _offset_for(slot: int) -> np.ndarray:
        return (slot - (n_slots - 1) / 2) * step_unit

    _add_arrow(fig, x_star, -scale * grad_f, color="#e34948", offset=_offset_for(series_slots["grad_f"]))
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", name="-∇f(x*)", line={"color": "#e34948", "width": 2.5})
    )

    lambda_sum = np.zeros(2)
    for i, (g_grad, s, is_active) in enumerate(zip(problem._g_grad, slacks, active, strict=True)):
        if not is_active:
            continue
        lam = 1.0 / (t_final * s)
        vec = lam * g_grad(x_star)
        lambda_sum += vec
        color = palette[(i + 1) % len(palette)]
        _add_arrow(fig, x_star, scale * vec, color=color, offset=_offset_for(series_slots[i]))
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                name=f"λ_{i}·∇g_{i}(x*) [λ_{i}={lam:.3g}]", line={"color": color, "width": 2.5},
            )
        )

    _add_arrow(fig, x_star, scale * lambda_sum, color="#52514e", offset=_offset_for(series_slots["sum"]))
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", name="Σ λ_i·∇g_i(x*)",
            line={"color": "#52514e", "width": 2.5, "dash": "dot"},
        )
    )

    fig.update_layout(
        **layout_template(
            dark=dark, title=f"{problem.name} — KKT stationarity at the optimum",
            xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 220, "t": 50, "b": 50},
        ),
    )
    vector_extent = scale * max(float(np.linalg.norm(grad_f)), float(np.linalg.norm(lambda_sum)), 1e-6)
    slot_extent = np.linalg.norm(step_unit) * n_slots
    pad = (vector_extent + slot_extent) * 1.3
    fig.update_xaxes(range=[x_star[0] - pad, x_star[0] + pad])
    fig.update_yaxes(range=[x_star[1] - pad, x_star[1] + pad], scaleanchor="x", scaleratio=1)
    return fig


def duality_gap_figure(
    gaps: list[float] | np.ndarray,
    *,
    tol: float | None = None,
    dark: bool = False,
) -> go.Figure:
    """Duality gap (or any other upper bound on distance-to-optimal-objective, e.g. LP
    primal/dual objective difference across a sequence of instances) vs. iteration, on a
    log axis -- the barrier method's `n_constraints / t` shrinking geometrically each
    outer step is what actually makes "strong duality" a *rate*, not just an endpoint
    fact.
    """
    gaps = np.asarray(gaps, dtype=float)
    color = contrasting_categorical(dark=dark)[0]

    fig = go.Figure(
        go.Scatter(
            x=np.arange(len(gaps)), y=np.maximum(gaps, 1e-16),
            mode="lines+markers", name="duality gap",
            line={"color": color, "width": 2.5}, marker={"size": 6, "color": color},
            hovertemplate="outer iter %{x}<br>gap=%{y:.4g}<extra></extra>",
        )
    )
    if tol is not None:
        fig.add_hline(y=tol, line={"color": "#898781", "width": 1.5, "dash": "dash"}, annotation_text="tol")

    fig.update_layout(
        **layout_template(
            dark=dark, title="Duality gap vs. iteration",
            xaxis_title="outer iteration", yaxis_title="duality gap", yaxis_type="log",
        ),
    )
    return fig
