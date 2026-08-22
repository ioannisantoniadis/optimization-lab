"""Figures for `optimlab.control`: a state/control trajectory pair (shared by LQR and
the pendulum swing-up — both are, in the end, "state over time" plus "the control that
produced it"), and a grid-world's value function with its greedy policy drawn as
arrows on top.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimlab.control.dynamic_programming import GridWorld
from optimlab.core import ArrayLike
from optimlab.viz.theme import contrasting_categorical, layout_template, sequential_blue

_ARROW_SYMBOLS = {0: "▲", 1: "▼", 2: "◀", 3: "▶"}  # up, down, left, right — matches ACTIONS' order


def trajectory_and_control_figure(
    t: ArrayLike,
    states: ArrayLike,
    controls: ArrayLike,
    *,
    state_labels: list[str] | None = None,
    control_labels: list[str] | None = None,
    title: str = "Trajectory and control",
    dark: bool = False,
) -> go.Figure:
    """State (top) and control (bottom) sharing a time axis. `states` is
    `(len(t), n_state_dims)`; `controls` is `(len(t) - 1, n_control_dims)` — one fewer
    row than `states`, since a control is applied *between* consecutive states, not
    reshaping-detected from either array's shape. Works for LQR (a linear state
    vector) and the pendulum swing-up (angle and angular velocity) alike.
    """
    t = np.asarray(t)
    states = np.asarray(states).reshape(len(t), -1)
    controls = np.asarray(controls).reshape(len(t) - 1, -1)
    state_labels = state_labels or [f"x{i}" for i in range(states.shape[1])]
    control_labels = control_labels or [f"u{i}" for i in range(controls.shape[1])]

    colors = contrasting_categorical(dark=dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("State", "Control"), row_heights=[0.6, 0.4])
    for i, label in enumerate(state_labels):
        fig.add_trace(
            go.Scatter(x=t, y=states[:, i], mode="lines", name=label, line={"color": colors[i % len(colors)], "width": 2.5}),
            row=1, col=1,
        )
    for i, label in enumerate(control_labels):
        fig.add_trace(
            go.Scatter(
                x=t[:-1], y=controls[:, i], mode="lines", name=label,
                line={"color": colors[(states.shape[1] + i) % len(colors)], "width": 2.5, "dash": "dot"},
            ),
            row=2, col=1,
        )
    fig.update_layout(**layout_template(dark=dark, title=title))
    fig.update_xaxes(title_text="time", row=2, col=1)
    return fig


def gridworld_figure(world: GridWorld, V: ArrayLike, policy: ArrayLike, *, dark: bool = False) -> go.Figure:
    """The converged value function as a heatmap, the greedy policy's action at every
    non-obstacle, non-goal cell as an arrow glyph on top, and the goal marked
    separately — a value-iteration result made geometric rather than left as a table
    of numbers.
    """
    V = np.asarray(V).copy()
    annotations = []
    for r in range(world.n_rows):
        for c in range(world.n_cols):
            if (r, c) in world.obstacles:
                V[r, c] = np.nan
                continue
            if (r, c) == world.goal:
                annotations.append({"x": c, "y": r, "text": "★", "showarrow": False, "font": {"size": 20}})
                continue
            annotations.append(
                {"x": c, "y": r, "text": _ARROW_SYMBOLS[int(policy[r, c])], "showarrow": False, "font": {"size": 16}}
            )

    fig = go.Figure(
        go.Heatmap(z=V, colorscale=sequential_blue(), hoverongaps=False, colorbar={"title": "value"})
    )
    fig.update_layout(
        **layout_template(dark=dark, title="Grid-world value function and policy", xaxis_title="col", yaxis_title="row"),
        annotations=annotations,
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig
