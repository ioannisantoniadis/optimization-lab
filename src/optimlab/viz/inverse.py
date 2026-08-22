"""Figures for `optimlab.inverse`: the true/blurred/recovered image triptych a
deblurring result is judged by, and the observed-vs-fitted trajectory comparison a
system-identification result is judged by.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimlab.core import ArrayLike
from optimlab.viz.theme import contrasting_categorical, layout_template, sequential_blue


def deblurring_figure(
    true_image: ArrayLike, observed: ArrayLike, recovered: ArrayLike, *, dark: bool = False
) -> go.Figure:
    """True, blurred/noisy observed, and recovered images side by side on a shared
    color scale — sharing the scale is what makes "recovered looks closer to true than
    observed does" a fair visual claim rather than an artifact of independently
    stretched contrast per panel.
    """
    images = [np.asarray(true_image), np.asarray(observed), np.asarray(recovered)]
    zmin, zmax = min(im.min() for im in images), max(im.max() for im in images)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("true", "observed (blurred + noisy)", "recovered"))
    for col, image in enumerate(images, start=1):
        fig.add_trace(
            go.Heatmap(
                z=image, colorscale=sequential_blue(), zmin=zmin, zmax=zmax,
                showscale=(col == 3), hoverinfo="skip",
            ),
            row=1, col=col,
        )
    fig.update_layout(**layout_template(dark=dark, title="Image deblurring"))
    n_rows, n_cols = images[0].shape
    # Explicit ranges, not autorange: a scaleanchor'd 1:1 aspect constraint across
    # three side-by-side subplots fights the shared figure width and pads the y-range
    # out to several times the image's own size to compensate -- plain reversed ranges
    # keep each panel tightly cropped to its actual pixel data instead.
    for col in range(1, 4):
        fig.update_yaxes(range=[n_rows - 0.5, -0.5], row=1, col=col)
        fig.update_xaxes(range=[-0.5, n_cols - 0.5], row=1, col=col)
    return fig


def system_id_figure(
    t: ArrayLike, observed: ArrayLike, fitted: ArrayLike, *, dark: bool = False
) -> go.Figure:
    """Noisy observed positions (markers) against the fitted model's trajectory
    (line) — a good fit means the line runs through the middle of the scatter, not
    that it touches every noisy point exactly.
    """
    colors = contrasting_categorical(dark=dark)
    fig = go.Figure(
        [
            go.Scatter(
                x=np.asarray(t), y=np.asarray(observed), mode="markers", name="observed",
                marker={"size": 5, "color": "#898781", "opacity": 0.6},
            ),
            go.Scatter(
                x=np.asarray(t), y=np.asarray(fitted), mode="lines", name="fitted model",
                line={"color": colors[0], "width": 2.5},
            ),
        ]
    )
    fig.update_layout(
        **layout_template(dark=dark, title="System identification", xaxis_title="time", yaxis_title="position"),
    )
    return fig
