"""Plotly figures for a single objective's landscape: filled contour and 3D surface, both
over the sequential blue ramp (`optimlab.viz.theme`), 2D problems only (that's the whole
point of these — they're the "look at the landscape with your own eyes" companion to the
high-dimensional tools in `optimlab.landscapes`, which exist precisely because you *can't*
do this once n_dim > ~3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from optimlab.core import Objective, Problem
from optimlab.viz.theme import layout_template, sequential_blue


def _evaluate_grid(
    f: Objective, x_range: tuple[float, float], y_range: tuple[float, float], resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate `f` over a `resolution x resolution` grid. Uses `jax.vmap` so both
    n-dimensional-style objectives (`jnp.sum(x**2)`, assuming a 1D vector) and
    2D-indexing-style objectives (`x[0], x[1]`) vectorize correctly — vmap maps `f` over
    a batch dimension while keeping each call's "1D vector" semantics intact, which a
    naive `f(X, Y)` broadcast would get wrong for the n-dimensional-style functions.
    """
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    try:
        batched_f = jax.vmap(lambda p: f(p))
        Z = np.asarray(batched_f(jnp.asarray(points, dtype=jnp.float64))).reshape(X.shape)
    except Exception:  # noqa: BLE001 - vmap/tracing can fail in many JAX-internal ways;
        # any of them should fall back to the always-correct pointwise loop below.
        Z = np.array([float(f(p)) for p in points]).reshape(X.shape)
    return X, Y, Z


def _resolve_domain(problem: Problem, domain: tuple[float, float] | None) -> tuple[float, float]:
    if domain is not None:
        return domain
    if problem.domain is not None:
        return problem.domain
    return (-5.0, 5.0)


def _display_z(Z: np.ndarray, log_z: bool) -> tuple[np.ndarray, str]:
    # Short colorbar titles on purpose: the colorbar shares the right margin with the
    # legend in race_figure, and a long title collides with legend entries there.
    if not log_z:
        return Z, "f(x)"
    return np.log10(Z - Z.min() + 1e-6), "log10 f(x)"


def contour_figure(
    problem: Problem,
    *,
    resolution: int = 150,
    domain: tuple[float, float] | None = None,
    log_z: bool = True,
    dark: bool = False,
) -> go.Figure:
    """A filled 2D contour of `problem.f`, with any known minima marked. `log_z=True`
    (the default) compresses the huge dynamic range typical of these benchmark functions
    — e.g. Rosenbrock's valley floor vs. its corners span several orders of magnitude —
    so the shape near the minimum doesn't get washed out to a single flat color.
    """
    if problem.n_dim != 2:
        raise ValueError(f"contour_figure needs a 2D problem, got n_dim={problem.n_dim}")

    low, high = _resolve_domain(problem, domain)
    X, Y, Z = _evaluate_grid(problem.f, (low, high), (low, high), resolution)
    Z_display, colorbar_title = _display_z(Z, log_z)

    fig = go.Figure(
        go.Contour(
            x=X[0], y=Y[:, 0], z=Z_display,
            colorscale=sequential_blue(),
            contours={"showlines": False},
            # Pinned low/short so it never collides with the legend, which sits
            # top-right in the same right-hand margin (see race_figure).
            colorbar={"title": colorbar_title, "len": 0.55, "y": 0.26, "x": 1.02},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{z:.3f}<extra></extra>",
        )
    )

    if problem.minimum is not None:
        minima = [problem.minimum] if problem.minimum.ndim == 1 else list(problem.minimum)
        fig.add_trace(
            go.Scatter(
                x=[m[0] for m in minima], y=[m[1] for m in minima],
                mode="markers", name="known minimum",
                marker={"symbol": "star", "size": 14, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
                hovertemplate="known minimum<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        **layout_template(
            dark=dark, title=problem.name, xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 140, "t": 50, "b": 50},
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def surface_figure(
    problem: Problem,
    *,
    resolution: int = 100,
    domain: tuple[float, float] | None = None,
    log_z: bool = False,
    dark: bool = False,
) -> go.Figure:
    """A 3D surface of `problem.f` — the "textbook picture" of a landscape. Genuinely
    only tells you something for 2 parameters; see `optimlab.landscapes` (ROADMAP Phase 6)
    for how we look at a million-parameter version of this same question.
    """
    if problem.n_dim != 2:
        raise ValueError(f"surface_figure needs a 2D problem, got n_dim={problem.n_dim}")

    low, high = _resolve_domain(problem, domain)
    X, Y, Z = _evaluate_grid(problem.f, (low, high), (low, high), resolution)
    Z_display, colorbar_title = _display_z(Z, log_z)

    fig = go.Figure(
        go.Surface(
            x=X, y=Y, z=Z_display,
            colorscale=sequential_blue(),
            colorbar={"title": colorbar_title},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{z:.3f}<extra></extra>",
        )
    )
    chrome_bg = "#1a1a19" if dark else "#fcfcfb"
    fig.update_layout(
        **layout_template(
            dark=dark,
            title=problem.name,
            scene={
                "xaxis_title": "x0", "yaxis_title": "x1", "zaxis_title": colorbar_title,
                "bgcolor": chrome_bg,
            },
        ),
    )
    return fig
