"""Figures for `optimlab.highdim`: the collapse of P(local minimum) with dimension,
random-direction concentration of measure, a Hessian's eigenspectrum as a scree plot,
a filter-normalized loss-landscape slice, a generic curve-comparison plot (reused for
both the linear-interpolation and mode-connectivity diagnostics), and NTK concentration
across width.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import ArrayLike
from optimlab.highdim.loss_landscape import LossSlice2D
from optimlab.highdim.ntk import NTKConcentrationResult
from optimlab.highdim.random_landscapes import CriticalPointStats
from optimlab.viz.theme import contrasting_categorical, layout_template, sequential_blue


def saddle_point_figure(stats: CriticalPointStats, *, dark: bool = False) -> go.Figure:
    """`P(local min)` and `P(saddle)` vs. dimension, log-y — the collapse is fast enough
    that a linear axis would flatten everything past dimension 4 into an indistinguishable
    line at zero.
    """
    colors = contrasting_categorical(dark=dark)
    fig = go.Figure()
    for name, values, color in [
        ("P(local minimum)", stats.p_local_min, colors[0]),
        ("P(saddle)", stats.p_saddle, colors[1]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=stats.dims, y=np.maximum(values, 1e-6), mode="lines+markers", name=name,
                line={"color": color, "width": 2.5}, marker={"size": 7, "color": color},
                hovertemplate=f"{name}<br>dim=%{{x}}<br>p=%{{y:.4g}}<extra></extra>",
            )
        )
    fig.update_layout(
        **layout_template(
            dark=dark, title="Probability a random critical point is a local minimum",
            xaxis_title="dimension", yaxis_title="probability", yaxis_type="log",
        ),
    )
    return fig


def cosine_similarity_figure(similarities_by_dim: dict[int, ArrayLike], *, dark: bool = False) -> go.Figure:
    """Overlaid histograms of pairwise cosine similarities, one per dimension in
    `similarities_by_dim` — low dimensions spread widely toward -1/+1, high dimensions
    collapse toward a narrow spike at 0.
    """
    colors = contrasting_categorical(dark=dark)
    fig = go.Figure()
    for i, (dim, sims) in enumerate(similarities_by_dim.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Histogram(
                x=np.asarray(sims), name=f"dim={dim}", histnorm="probability density",
                marker={"color": color, "opacity": 0.6}, xbins={"start": -1.0, "end": 1.0, "size": 0.02},
            )
        )
    fig.update_layout(
        **layout_template(
            dark=dark, title="Pairwise cosine similarity of random directions",
            xaxis_title="cosine similarity", yaxis_title="density", barmode="overlay",
        ),
    )
    return fig


def hessian_spectrum_figure(eigenvalues: ArrayLike, *, dark: bool = False, title: str = "Hessian eigenspectrum") -> go.Figure:
    """A scree plot — eigenvalues sorted descending against their rank — rather than a
    histogram: a "few large outliers over a near-zero bulk" spectrum shows up as a
    sharp cliff from the first few points down to a long near-flat tail, which a
    histogram's binning tends to blur (the bulk's many near-identical small values pile
    into one bar, and the rare outliers each get their own near-invisible sliver).
    """
    eigenvalues = np.sort(np.asarray(eigenvalues))[::-1]
    color = contrasting_categorical(dark=dark)[0]
    fig = go.Figure(
        go.Scatter(
            x=np.arange(1, eigenvalues.size + 1), y=eigenvalues, mode="markers",
            marker={"size": 6, "color": color},
            hovertemplate="rank %{x}<br>eigenvalue=%{y:.4g}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#898781", "width": 1, "dash": "dot"})
    fig.update_layout(
        **layout_template(dark=dark, title=title, xaxis_title="rank (sorted descending)", yaxis_title="eigenvalue"),
    )
    return fig


def loss_landscape_figure(slice_: LossSlice2D, *, log_z: bool = True, dark: bool = False) -> go.Figure:
    """A filter-normalized 2D slice through weight space, drawn as a filled contour
    exactly like `optimlab.viz.landscape.contour_figure` — the base point (the trained
    minimum the slice is centered on) sits at the origin `(0, 0)` by construction.
    """
    Z = slice_.Z
    z_display, colorbar_title = (np.log10(Z - Z.min() + 1e-8), "log10 loss") if log_z else (Z, "loss")
    fig = go.Figure(
        go.Contour(
            x=slice_.A[0], y=slice_.B[:, 0], z=z_display, colorscale=sequential_blue(),
            ncontours=20, contours={"showlines": True, "coloring": "fill"},
            line={"width": 0.6, "color": "rgba(11,11,11,0.35)"},
            colorbar={"title": colorbar_title},
            hovertemplate="a=%{x:.3f}<br>b=%{y:.3f}<br>%{z:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0], mode="markers", name="trained minimum",
            marker={"symbol": "star", "size": 14, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
        )
    )
    fig.update_layout(
        **layout_template(
            dark=dark, title="Filter-normalized loss landscape slice",
            xaxis_title="direction 1", yaxis_title="direction 2",
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def curve_comparison_figure(
    curves: dict[str, tuple[ArrayLike, ArrayLike]],
    *,
    title: str,
    xaxis_title: str = "t",
    yaxis_title: str = "loss",
    log_y: bool = False,
    dark: bool = False,
) -> go.Figure:
    """One line per named curve `(xs, ys)` — the shared plotting code behind both the
    linear-interpolation diagnostic and the mode-connectivity comparison, which are
    both, in the end, just "loss along a 1D path" plotted against each other.
    """
    colors = contrasting_categorical(dark=dark)
    fig = go.Figure()
    for i, (name, (xs, ys)) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.asarray(xs), y=np.asarray(ys), mode="lines+markers", name=name,
                line={"color": color, "width": 2.5}, marker={"size": 5, "color": color},
            )
        )
    fig.update_layout(
        **layout_template(
            dark=dark, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
            yaxis_type="log" if log_y else "linear",
        ),
    )
    return fig


def ntk_concentration_figure(result: NTKConcentrationResult, *, dark: bool = False) -> go.Figure:
    """Mean relative NTK difference between independent random initializations vs.
    width, log-x, with a shaded +/-1 std band across the seed pairs sampled at each
    width — the concentration Jacot et al. 2018's infinite-width limit predicts, shown
    as an empirical trend rather than derived.
    """
    color = contrasting_categorical(dark=dark)[0]
    widths = result.widths
    upper = result.mean_relative_diff + result.std_relative_diff
    lower = np.maximum(result.mean_relative_diff - result.std_relative_diff, 0.0)

    fig = go.Figure(
        [
            go.Scatter(
                x=np.concatenate([widths, widths[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                mode="lines", fill="toself", fillcolor="rgba(42,120,214,0.15)", line={"width": 0},
                showlegend=False, hoverinfo="skip",
            ),
            go.Scatter(
                x=widths, y=result.mean_relative_diff, mode="lines+markers", name="mean relative NTK difference",
                line={"color": color, "width": 2.5}, marker={"size": 7, "color": color},
            ),
        ]
    )
    fig.update_layout(
        **layout_template(
            dark=dark, title="NTK concentration across width",
            xaxis_title="hidden width", yaxis_title="relative difference between independent inits",
            xaxis_type="log",
        ),
    )
    return fig
