"""Plotly figures for `optimlab.linalg`: a fitted line against its data, a residual
diagnostic that works regardless of how many features `A` has, the geometric meaning of
condition number made literal (a circle becomes an ellipse), and how ridge regression's
shrinkage grows with its regularization strength.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import ArrayLike
from optimlab.linalg.svd import svd
from optimlab.viz.theme import contrasting_categorical, layout_template


def regression_fit_figure(
    A: ArrayLike, b: ArrayLike, x: ArrayLike, *, name: str = "least squares fit", dark: bool = False
) -> go.Figure:
    """Data + fitted line for **single-feature** regression (`A` a column vector) — the
    classic textbook picture. For multi-feature fits, see `residuals_figure` instead:
    there's no single line to draw once there's more than one feature, but "predicted vs.
    residual" is exactly as informative regardless of feature count.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] != 1:
        raise ValueError(f"regression_fit_figure needs a single-feature A (shape (m, 1)), got {A.shape}")
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    t = np.array([A[:, 0].min(), A[:, 0].max()])
    fitted = t * x[0] + (x[1] if x.size > 1 else 0.0)
    color = contrasting_categorical(dark=dark)[0]

    fig = go.Figure(
        [
            go.Scatter(
                x=A[:, 0], y=b, mode="markers", name="data",
                marker={"size": 7, "color": "#898781", "line": {"color": "#ffffff", "width": 0.5}},
                hovertemplate="x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=t, y=fitted, mode="lines", name=name,
                line={"color": color, "width": 2.5},
                hovertemplate="fit<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(**layout_template(dark=dark, xaxis_title="x", yaxis_title="y"))
    return fig


def residuals_figure(A: ArrayLike, b: ArrayLike, x: ArrayLike, *, dark: bool = False) -> go.Figure:
    """Predicted value vs. residual, the standard regression diagnostic that works no
    matter how many features `A` has: a well-fit model's residuals scatter randomly
    around zero with no visible pattern, while a trend, curve, or funnel shape signals
    the *linear* model is missing something (nonlinearity, heteroscedasticity, ...) that
    a bigger residual norm alone wouldn't tell you.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    predicted = A @ np.asarray(x, dtype=float)
    residuals = predicted - b
    color = contrasting_categorical(dark=dark)[0]

    fig = go.Figure(
        go.Scatter(
            x=predicted, y=residuals, mode="markers", name="residuals",
            marker={"size": 7, "color": color, "line": {"color": "#ffffff", "width": 0.5}},
            hovertemplate="predicted=%{x:.3f}<br>residual=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#898781", "width": 1, "dash": "dash"})
    fig.update_layout(**layout_template(dark=dark, xaxis_title="predicted value", yaxis_title="residual"))
    return fig


def svd_conditioning_figure(A: ArrayLike, *, dark: bool = False) -> go.Figure:
    """The geometric meaning of condition number, made literal: the unit circle (every
    direction, weighted equally) mapped through a 2x2 `A` becomes an ellipse whose
    semi-axes *are* `A`'s singular values — see `condition_number()`'s docstring for the
    full argument this figure is illustrating. A near-circular ellipse means `A` is
    well-conditioned; a needle-thin one means it's not.
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError(f"svd_conditioning_figure needs a 2x2 matrix, got shape {A.shape}")

    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    result = svd(A)
    colors = contrasting_categorical(dark=dark)

    fig = go.Figure(
        [
            go.Scatter(
                x=circle[0], y=circle[1], mode="lines", name="unit circle",
                line={"color": "#898781", "width": 1.5, "dash": "dash"}, hoverinfo="skip",
            ),
            go.Scatter(
                x=ellipse[0], y=ellipse[1], mode="lines", name="A · (unit circle)",
                line={"color": colors[0], "width": 2.5}, fill="toself",
                fillcolor="rgba(235,104,52,0.12)", hoverinfo="skip",
            ),
        ]
    )
    for i, s in enumerate(result.s):
        axis = result.U[:, i] * s
        fig.add_trace(
            go.Scatter(
                x=[0, axis[0]], y=[0, axis[1]], mode="lines+markers",
                name=f"singular value {i} = {s:.3g}",
                line={"color": colors[i + 1], "width": 3},
                marker={"size": [0, 8], "color": colors[i + 1]},
                hovertemplate=f"σ{i}=%{{customdata:.4g}}<extra></extra>", customdata=[s, s],
            )
        )

    span = max(1.2, float(np.abs(ellipse).max()) * 1.2)
    fig.update_layout(
        **layout_template(
            dark=dark, title=f"condition number = {result.condition_number:.3g}",
            xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 200, "t": 50, "b": 50},
        ),
    )
    fig.update_xaxes(range=[-span, span])
    fig.update_yaxes(range=[-span, span], scaleanchor="x", scaleratio=1)
    return fig


def ridge_path_figure(
    A: ArrayLike, b: ArrayLike, alphas: ArrayLike, *, feature_names: list[str] | None = None, dark: bool = False
) -> go.Figure:
    """Each coefficient's value as `alpha` grows — the "regularization path." Every
    coefficient shrinks toward zero as `alpha` increases (visible directly from ridge's
    `s / (s^2 + alpha)` shrinkage factor: as `alpha -> infinity`, every term -> 0), but
    coefficients riding on small singular values (the directions `condition_number`
    flags as noise-prone) shrink fastest, since `alpha` dominates a small `s^2` sooner.
    """
    from optimlab.linalg.regression import ridge_regression

    A = np.asarray(A, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    n_features = A.shape[1]
    names = feature_names or [f"x{i}" for i in range(n_features)]
    colors = contrasting_categorical(dark=dark)

    coefficients = np.array([ridge_regression(A, b, alpha).x for alpha in alphas])

    fig = go.Figure()
    for j in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=alphas, y=coefficients[:, j], mode="lines", name=names[j],
                line={"color": colors[j % len(colors)], "width": 2},
                hovertemplate=f"{names[j]}<br>alpha=%{{x:.3g}}<br>coef=%{{y:.4g}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#898781", "width": 1})
    fig.update_layout(
        **layout_template(
            dark=dark, title="Ridge regularization path",
            xaxis_title="alpha", yaxis_title="coefficient value", xaxis_type="log",
        ),
    )
    fig.update_xaxes(dtick=1)  # decade ticks only (10^0, 10^1, ...) -- default log-axis
    # minor ticks (..., 2, 5, 10, 20, 50, ...) get crowded at this figure's width.
    return fig
