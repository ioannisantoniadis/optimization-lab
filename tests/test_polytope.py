import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.optimizers.linear_programming import LinearProgram, simplex
from optimlab.viz import polytope_figure


def test_polytope_figure_shades_the_feasible_region_and_shows_the_simplex_path():
    lp = LinearProgram(name="classic_lp", c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
    result = simplex(lp)
    fig = polytope_figure(lp, result)

    names = [trace.name for trace in fig.data]
    assert "feasible region" in names
    assert "simplex path" in names
    assert "optimum" in names

    path_trace = fig.data[names.index("simplex path")]
    np.testing.assert_allclose([path_trace.x[-1], path_trace.y[-1]], result.x, atol=1e-8)


def test_polytope_figure_region_vertices_are_all_feasible():
    lp = LinearProgram(name="pentagon_lp", c=[-1.0, -2.0], A_ub=[[1, 3], [1, 1], [3, 1]], b_ub=[15, 7, 15])
    fig = polytope_figure(lp)
    region = fig.data[[t.name for t in fig.data].index("feasible region")]
    pts = np.column_stack([region.x[:-1], region.y[:-1]])  # last point repeats the first (closes the polygon)
    A_ub, b_ub = np.asarray(lp.A_ub), np.asarray(lp.b_ub)
    assert np.all(A_ub @ pts.T <= b_ub[:, None] + 1e-6)
    assert np.all(pts >= -1e-8)


def test_polytope_figure_rejects_non_2d_lp():
    lp = LinearProgram(c=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="2-variable"):
        polytope_figure(lp)


def test_polytope_figure_without_a_result_still_draws_the_region():
    lp = LinearProgram(c=[-1.0, -1.0], A_ub=[[1, 1]], b_ub=[4])
    fig = polytope_figure(lp)
    assert isinstance(fig, go.Figure)
    names = [trace.name for trace in fig.data]
    assert "feasible region" in names
    assert "simplex path" not in names
