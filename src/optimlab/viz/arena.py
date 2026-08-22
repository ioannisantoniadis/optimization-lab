"""A figure for `optimlab.arena`: every solver's final objective value, one bar each,
sorted best-to-worst — a solver that raised an exception (didn't apply to this
`Problem`) gets its own visibly-marked bar rather than silently vanishing from the
comparison.
"""

from __future__ import annotations

import plotly.graph_objects as go

from optimlab.arena import ArenaReport
from optimlab.viz.theme import layout_template


def arena_figure(report: ArenaReport, *, log_y: bool = True, dark: bool = False) -> go.Figure:
    rows = sorted(report.summary_rows(), key=lambda r: (r["f"] is None, r["f"]))
    names = [r["name"] for r in rows]
    converged_color, unconverged_color, failed_color = "#1baf7a", "#eda100", "#e34948"

    ys, colors, hover = [], [], []
    floor = min((r["f"] for r in rows if r["f"] not in (None, 0.0)), default=1e-16)
    for r in rows:
        if r["f"] is None:
            ys.append(floor * 0.1 if log_y else 0.0)
            colors.append(failed_color)
            hover.append(f"{r['name']}<br>failed: {r['error']}<extra></extra>")
        else:
            ys.append(max(r["f"], floor * 0.1) if log_y else r["f"])
            colors.append(converged_color if r["converged"] else unconverged_color)
            hover.append(
                f"{r['name']}<br>f={r['f']:.4g}<br>n_iter={r['n_iter']}<br>"
                f"wall_time={r['wall_time'] * 1000:.2f}ms<br>converged={r['converged']}<extra></extra>"
            )

    fig = go.Figure(
        go.Bar(x=names, y=ys, marker={"color": colors}, hovertemplate=hover, showlegend=False)
    )
    for color, label in [(converged_color, "converged"), (unconverged_color, "max_iter reached"), (failed_color, "not applicable")]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker={"color": color}, name=label, showlegend=True))

    fig.update_layout(
        **layout_template(
            dark=dark, title=f"Solver arena: {report.problem_name}",
            xaxis_title="solver", yaxis_title="final objective value", yaxis_type="log" if log_y else "linear",
        ),
        showlegend=True,
    )
    fig.update_xaxes(tickangle=-30)
    return fig
