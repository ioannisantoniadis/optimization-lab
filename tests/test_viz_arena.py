import numpy as np
import plotly.graph_objects as go

from optimlab.arena import run_arena
from optimlab.core import Problem
from optimlab.viz import arena_figure


def test_arena_figure_has_one_bar_per_solver_no_stray_legend_trace():
    problem = Problem(f=lambda x: x[0] ** 2 + x[1] ** 2, x0=np.array([3.0, 3.0]))
    report = run_arena(problem)
    fig = arena_figure(report)

    assert isinstance(fig, go.Figure)
    bar_trace = next(t for t in fig.data if t.x is not None and list(t.x) == [r["name"] for r in sorted(report.summary_rows(), key=lambda r: (r["f"] is None, r["f"]))])
    assert bar_trace.showlegend is False
    legend_names = {t.name for t in fig.data if t.showlegend is not False and t.name}
    assert legend_names == {"converged", "max_iter reached", "not applicable"}


def test_arena_figure_marks_failed_solvers_distinctly():
    problem = Problem(f=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, x0=np.array([0.0, 0.0]))
    report = run_arena(problem)  # genetic_algorithm/particle_swarm fail: no domain
    fig = arena_figure(report)

    bar_trace = next(t for t in fig.data if t.showlegend is False)
    names = list(bar_trace.x)
    colors = list(bar_trace.marker.color)
    failed_color = "#e34948"
    for name in ("genetic_algorithm", "particle_swarm"):
        assert colors[names.index(name)] == failed_color
