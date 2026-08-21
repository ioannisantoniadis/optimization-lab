"""Figures for `optimlab.inference`: comparing a closed-form posterior against the
Laplace (Gaussian) approximation and MCMC samples on the same axes, an MCMC chain's own
trace/histogram diagnostics, and a fitted Gaussian mixture drawn as data points plus
each component's covariance ellipse.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimlab.core import ArrayLike
from optimlab.inference.em import GMMResult
from optimlab.inference.laplace import GaussianApprox
from optimlab.viz.theme import contrasting_categorical, layout_template


def posterior_figure(
    *,
    x_range: tuple[float, float],
    true_pdf: Callable[[np.ndarray], np.ndarray] | None = None,
    laplace: GaussianApprox | None = None,
    mcmc_samples: ArrayLike | None = None,
    resolution: int = 400,
    dark: bool = False,
) -> go.Figure:
    """Overlay up to three views of a single scalar parameter's posterior: the true
    density (a line, when a closed form exists), the Laplace approximation's Gaussian
    (a dashed line, sharing the true density's exact peak location only when the true
    posterior is itself symmetric), and MCMC samples (a normalized histogram). Any
    subset may be omitted — pass only what you have.
    """
    colors = contrasting_categorical(dark=dark)
    xs = np.linspace(*x_range, resolution)
    fig = go.Figure()

    if mcmc_samples is not None:
        samples = np.asarray(mcmc_samples).ravel()
        fig.add_trace(
            go.Histogram(
                x=samples, histnorm="probability density", name="MCMC samples",
                marker={"color": "#898781", "opacity": 0.55},
                xbins={"start": x_range[0], "end": x_range[1], "size": (x_range[1] - x_range[0]) / 60},
            )
        )

    if true_pdf is not None:
        fig.add_trace(
            go.Scatter(
                x=xs, y=true_pdf(xs), mode="lines", name="true posterior",
                line={"color": colors[0], "width": 3},
            )
        )

    if laplace is not None:
        mean, std = float(laplace.mean[0]), float(np.sqrt(laplace.cov[0, 0]))
        density = np.exp(-0.5 * ((xs - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        fig.add_trace(
            go.Scatter(
                x=xs, y=density, mode="lines", name="Laplace approximation",
                line={"color": colors[1], "width": 3, "dash": "dash"},
            )
        )

    fig.update_layout(
        **layout_template(dark=dark, title="Posterior comparison", xaxis_title="parameter", yaxis_title="density"),
    )
    return fig


def mcmc_trace_figure(samples: ArrayLike, *, dark: bool = False) -> go.Figure:
    """A chain's raw trace (value at each kept step) next to its marginal histogram —
    the standard first-look MCMC diagnostic: a trace that looks like flat, structureless
    noise ("good mixing") gives some confidence the chain explored the posterior rather
    than getting stuck; a trace with visible drift or long flat stretches (poor mixing)
    is a warning the samples may not be a trustworthy stand-in for the true posterior yet.
    """
    samples = np.asarray(samples).ravel()
    color = contrasting_categorical(dark=dark)[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=("Trace", "Marginal"))
    fig.add_trace(
        go.Scatter(
            x=np.arange(samples.size), y=samples, mode="lines", name="chain",
            line={"color": color, "width": 1}, showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            y=samples, histnorm="probability density", marker={"color": color, "opacity": 0.7}, showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(**layout_template(dark=dark, title="MCMC chain diagnostics"))
    fig.update_xaxes(title_text="iteration", row=1, col=1)
    fig.update_yaxes(title_text="parameter value", row=1, col=1)
    return fig


def _ellipse_points(mean: np.ndarray, cov: np.ndarray, *, n_std: float, resolution: int = 100) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    axes = eigenvectors @ np.diag(n_std * np.sqrt(np.maximum(eigenvalues, 0.0)))
    return (axes @ circle).T + mean


def gmm_figure(X: ArrayLike, result: GMMResult, *, n_std: float = 2.0, dark: bool = False) -> go.Figure:
    """A fitted `optimlab.inference.em.GMMResult` made geometric: every data point
    colored by its most-likely component (`argmax` of its soft responsibilities — a hard
    assignment for plotting only, the fit itself never hardens it), each component's mean
    marked, and an `n_std`-standard-deviation covariance ellipse per component (the
    Gaussian analogue of a confidence region) drawn directly from that component's
    covariance matrix's eigendecomposition.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError(f"gmm_figure needs 2D data, got shape {X.shape}")

    colors = contrasting_categorical(dark=dark)
    assignments = np.argmax(result.responsibilities, axis=1)
    n_components = result.means.shape[0]

    fig = go.Figure()
    for k in range(n_components):
        color = colors[k % len(colors)]
        mask = assignments == k
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"component {k}",
                marker={"size": 6, "color": color, "opacity": 0.6},
                hovertemplate=f"component {k}<extra></extra>",
            )
        )
        ellipse = _ellipse_points(result.means[k], result.covariances[k], n_std=n_std)
        fig.add_trace(
            go.Scatter(
                x=ellipse[:, 0], y=ellipse[:, 1], mode="lines", showlegend=False,
                line={"color": color, "width": 2.5},
                hovertemplate=f"component {k}, {n_std}-sigma<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[result.means[k, 0]], y=[result.means[k, 1]], mode="markers", showlegend=False,
                marker={"symbol": "x", "size": 12, "color": "#ffffff", "line": {"color": color, "width": 2}},
                hovertemplate=f"component {k} mean<extra></extra>",
            )
        )

    fig.update_layout(
        **layout_template(
            dark=dark, title="Fitted Gaussian mixture", xaxis_title="x0", yaxis_title="x1",
            legend={"y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
            margin={"l": 60, "r": 140, "t": 50, "b": 50},
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
