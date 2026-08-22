"""A figure for `optimlab.problems.sociology`: each user's fair allocation next to
each resource's usage against its capacity — the two halves of "did this allocation
actually respect the constraints, and how did it split the resource."
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimlab.core import ArrayLike
from optimlab.viz.theme import contrasting_categorical, layout_template


def fair_allocation_figure(A: ArrayLike, capacities: ArrayLike, x: ArrayLike, *, dark: bool = False) -> go.Figure:
    A = np.asarray(A, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    x = np.asarray(x, dtype=float)
    colors = contrasting_categorical(dark=dark)

    usage = A @ x
    users = [f"user {i}" for i in range(len(x))]
    resources = [f"resource {j}" for j in range(len(capacities))]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Allocation per user", "Usage vs. capacity"))
    fig.add_trace(
        go.Bar(x=users, y=x, marker={"color": colors[0]}, showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=resources, y=usage, name="usage", marker={"color": colors[1]}),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=resources, y=capacities, name="capacity", marker={"color": "#898781", "opacity": 0.4}),
        row=1, col=2,
    )
    fig.update_layout(**layout_template(dark=dark, title="Proportionally fair resource allocation"), barmode="group")
    fig.update_yaxes(title_text="allocation", row=1, col=1)
    fig.update_yaxes(title_text="amount", row=1, col=2)
    return fig
