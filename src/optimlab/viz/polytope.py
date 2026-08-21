"""2D feasible-region visualization for `optimlab.optimizers.linear_programming` — the
picture behind the fact simplex exploits directly: a linear objective over a convex
polytope always has an optimum at a *vertex*, so a solver never needs to look anywhere
in the polytope's interior.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.optimizers.linear_programming import LinearProgram, LPResult
from optimlab.viz.theme import contrasting_categorical, layout_template

_TOL = 1e-7


def _half_planes(lp: LinearProgram, bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Every constraint as `a . x <= b`, including the implicit `x >= 0` and a bounding
    box (from `bounds`) so an unbounded feasible region still closes into a polygon —
    the box is a viewport clip, not part of the LP itself.
    """
    rows_a, rows_b = [], []
    if lp.A_ub is not None:
        rows_a.append(lp.A_ub)
        rows_b.append(lp.b_ub)
    if lp.A_eq is not None:
        # a.x = b becomes two inequalities: a.x <= b and -a.x <= -b.
        rows_a.append(lp.A_eq)
        rows_b.append(lp.b_eq)
        rows_a.append(-lp.A_eq)
        rows_b.append(-lp.b_eq)
    rows_a.append(-np.eye(2))
    rows_b.append(np.zeros(2))  # x >= 0, y >= 0

    low, high = bounds
    rows_a.append(np.eye(2))
    rows_b.append(np.array([high, high]))
    rows_a.append(-np.eye(2))
    rows_b.append(np.array([-low, -low]))

    return np.vstack(rows_a), np.concatenate(rows_b)


def _feasible_polygon_vertices(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Every pairwise intersection of the half-plane boundary lines that satisfies every
    other constraint is a vertex of the (convex) feasible region — standard approach for
    plotting a 2D LP's feasible polygon. Points are returned sorted by angle around their
    centroid, i.e. in polygon-boundary order, ready to hand straight to `go.Scatter`
    with `fill="toself"`.
    """
    candidates = []
    m = A.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            M = np.array([A[i], A[j]])
            if abs(np.linalg.det(M)) < _TOL:
                continue
            point = np.linalg.solve(M, [b[i], b[j]])
            if np.all(A @ point <= b + 1e-6):
                candidates.append(point)

    if len(candidates) < 3:
        return np.zeros((0, 2))

    points = np.array(candidates)
    # De-duplicate near-identical vertices (multiple constraint pairs often meet at the
    # same point, e.g. any two of x>=0/y>=0/a box edge at a shared corner).
    unique = []
    for p in points:
        if not any(np.allclose(p, u, atol=1e-6) for u in unique):
            unique.append(p)
    points = np.array(unique)

    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def polytope_figure(
    lp: LinearProgram,
    result: LPResult | None = None,
    *,
    bounds: tuple[float, float] | None = None,
    dark: bool = False,
) -> go.Figure:
    """The feasible region of a 2D `LinearProgram`, shaded, with `result.vertices` (if
    given) overlaid as the path `simplex` actually walked across it — from-vertex to
    adjacent-vertex, never through the interior.
    """
    if lp.n_vars != 2:
        raise ValueError(f"polytope_figure needs a 2-variable LinearProgram, got n_vars={lp.n_vars}")

    if bounds is None:
        # Two passes: a generous box first, just to close off an otherwise-unbounded
        # region so it has *some* finite polygon at all; then re-tighten the box to
        # that polygon's own extent, so the visible plot isn't dominated by empty space
        # out toward an arbitrary large cutoff (b_ub coefficients don't reliably predict
        # the x/y scale of the feasible region, so guessing bounds from them directly
        # tends to zoom out far more than the region actually needs).
        rough_A, rough_b = _half_planes(lp, (0.0, 1000.0))
        rough_polygon = _feasible_polygon_vertices(rough_A, rough_b)
        extent = [10.0]
        if rough_polygon.shape[0] >= 3:
            extent += list(rough_polygon.max(axis=0))
        if result is not None and result.vertices:
            extent += list(np.abs(np.array(result.vertices)).max(axis=0))
        bounds = (0.0, max(extent) * 1.25)

    A, b = _half_planes(lp, bounds)
    polygon = _feasible_polygon_vertices(A, b)

    fig = go.Figure()
    if polygon.shape[0] >= 3:
        fig.add_trace(
            go.Scatter(
                x=np.append(polygon[:, 0], polygon[0, 0]),
                y=np.append(polygon[:, 1], polygon[0, 1]),
                mode="lines", fill="toself", name="feasible region",
                line={"color": "#2a78d6", "width": 1.5},
                fillcolor="rgba(42,120,214,0.18)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=polygon[:, 0], y=polygon[:, 1], mode="markers", name="vertices",
                marker={"size": 6, "color": "#2a78d6"},
                hovertemplate="vertex<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            )
        )

    if result is not None and result.vertices:
        path = np.array(result.vertices)
        color = contrasting_categorical(dark=dark)[0]
        fig.add_trace(
            go.Scatter(
                x=path[:, 0], y=path[:, 1], mode="lines+markers", name="simplex path",
                line={"color": color, "width": 2.5, "dash": "dot"},
                marker={"size": 9, "color": color, "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="pivot %{pointNumber}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[path[-1, 0]], y=[path[-1, 1]], mode="markers", name="optimum",
                marker={"symbol": "star", "size": 16, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
                hovertemplate=f"optimum<br>x=%{{x:.3f}}, y=%{{y:.3f}}<br>objective={result.objective:.4g}<extra></extra>",
            )
        )

    fig.update_layout(
        **layout_template(
            dark=dark, title=lp.name, xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 140, "t": 50, "b": 50},
        ),
    )
    fig.update_xaxes(range=[bounds[0] - 0.5, bounds[1] + 0.5])
    fig.update_yaxes(range=[bounds[0] - 0.5, bounds[1] + 0.5], scaleanchor="x", scaleratio=1)
    return fig
