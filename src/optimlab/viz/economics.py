"""A figure for `optimlab.problems.economics`: the efficient frontier itself — risk on
the x-axis, return on the y-axis, the classic Markowitz picture.

The target-return sweep traces the whole minimum-variance curve, but only its upper
branch (at or above the global-minimum-variance return) is the *efficient* frontier;
every point on the lower branch is dominated by the upper-branch portfolio of equal
risk and higher return. The two branches are drawn differently so the figure doesn't
label a dominated portfolio "efficient".
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from optimlab.problems.economics import EfficientFrontier
from optimlab.viz.theme import CHART_CHROME, contrasting_categorical, layout_template


def efficient_frontier_figure(frontier: EfficientFrontier, *, dark: bool = False) -> go.Figure:
    color = contrasting_categorical(dark=dark)[0]
    muted = CHART_CHROME["dark" if dark else "light"]["muted"]
    min_idx = int(np.argmin(frontier.risks))
    min_return = frontier.target_returns[min_idx]
    # Both branches include the vertex itself, so the drawn curve stays continuous.
    upper = frontier.target_returns >= min_return
    lower = frontier.target_returns <= min_return
    hover = "risk=%{x:.4f}<br>return=%{y:.4f}<extra></extra>"

    fig = go.Figure(
        [
            go.Scatter(
                x=frontier.risks[lower], y=frontier.target_returns[lower], mode="lines+markers",
                name="dominated (inefficient) branch",
                line={"color": muted, "width": 2, "dash": "dash"}, marker={"size": 5, "color": muted},
                hovertemplate=hover,
            ),
            go.Scatter(
                x=frontier.risks[upper], y=frontier.target_returns[upper], mode="lines+markers",
                name="efficient frontier",
                line={"color": color, "width": 2.5}, marker={"size": 6, "color": color},
                hovertemplate=hover,
            ),
            go.Scatter(
                x=[frontier.risks[min_idx]], y=[min_return], mode="markers",
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
