"""Figures for `optimlab.ml`: a physics-informed network's prediction against the true
solution of the differential equation it was trained to satisfy — the network's own
loss never referenced this true solution, so agreement here is a genuine outside check.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.core import ArrayLike
from optimlab.viz.theme import contrasting_categorical, layout_template


def pinn_solution_figure(
    xs: ArrayLike, predicted: ArrayLike, true: ArrayLike, *, dark: bool = False
) -> go.Figure:
    colors = contrasting_categorical(dark=dark)
    fig = go.Figure(
        [
            go.Scatter(
                x=np.asarray(xs), y=np.asarray(true), mode="lines", name="analytic solution",
                line={"color": colors[0], "width": 4},
            ),
            go.Scatter(
                x=np.asarray(xs), y=np.asarray(predicted), mode="markers", name="PINN prediction",
                marker={"size": 7, "color": colors[1]},
            ),
        ]
    )
    fig.update_layout(
        **layout_template(dark=dark, title="Physics-informed network vs. the analytic solution", xaxis_title="x", yaxis_title="y"),
    )
    return fig
