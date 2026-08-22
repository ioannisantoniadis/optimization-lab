"""A figure for `optimlab.problems.economics`: the efficient frontier itself — risk on
the x-axis, return on the y-axis, the classic Markowitz picture.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.problems.economics import EfficientFrontier
from optimlab.viz.theme import contrasting_categorical, layout_template


def efficient_frontier_figure(frontier: EfficientFrontier, *, dark: bool = False) -> go.Figure:
    color = contrasting_categorical(dark=dark)[0]
    min_idx = int(np.argmin(frontier.risks))

    fig = go.Figure(
        [
            go.Scatter(
                x=frontier.risks, y=frontier.target_returns, mode="lines+markers", name="efficient frontier",
                line={"color": color, "width": 2.5}, marker={"size": 6, "color": color},
                hovertemplate="risk=%{x:.4f}<br>return=%{y:.4f}<extra></extra>",
            ),
            go.Scatter(
                x=[frontier.risks[min_idx]], y=[frontier.target_returns[min_idx]], mode="markers",
                name="global minimum variance",
                marker={"symbol": "star", "size": 16, "color": "#ffffff", "line": {"color": "#0b0b0b", "width": 1.5}},
            ),
        ]
    )
    fig.update_layout(
        **layout_template(
            dark=dark, title="Markowitz efficient frontier", xaxis_title="risk (portfolio std dev)",
            yaxis_title="expected return",
        ),
    )
    return fig
