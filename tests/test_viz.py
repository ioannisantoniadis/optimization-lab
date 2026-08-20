import plotly.graph_objects as go
import pytest

from optimlab.landscapes import get
from optimlab.optimizers import bfgs, gradient_descent, newton_method
from optimlab.viz import (
    contour_figure,
    convergence_figure,
    race_figure,
    solver_color_map,
    surface_figure,
)


def _himmelblau_results():
    bf = get("himmelblau")
    return bf, {
        "gradient_descent": gradient_descent(bf.problem(x0=[-4.0, 4.0]), lr=0.01, max_iter=200),
        "newton": newton_method(bf.problem(x0=[-4.0, 4.0])),
        "bfgs": bfgs(bf.problem(x0=[-4.0, 4.0])),
    }


def test_contour_figure_builds_for_2d_problem():
    bf = get("himmelblau")
    fig = contour_figure(bf.problem(x0=[0.0, 0.0]))
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Contour)
    # known minima are marked as an extra trace
    assert any(isinstance(t, go.Scatter) for t in fig.data)


def test_contour_figure_rejects_non_2d_problem():
    bf = get("sphere")
    with pytest.raises(ValueError, match="2D"):
        contour_figure(bf.problem(n_dim=5))


def test_surface_figure_builds_for_2d_problem():
    bf = get("rosenbrock")
    fig = surface_figure(bf.problem(x0=[-1.0, 1.0]), resolution=30)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Surface)


def test_race_figure_overlays_one_trajectory_per_solver():
    bf, results = _himmelblau_results()
    fig = race_figure(bf.problem(x0=[-4.0, 4.0]), results)
    # 1 contour + 1 known-minima marker + 2 traces (line, endpoint) per solver
    assert len(fig.data) == 2 + 2 * len(results)


def test_convergence_figure_has_one_line_per_solver():
    _, results = _himmelblau_results()
    fig = convergence_figure(results)
    assert len(fig.data) == len(results)
    assert fig.layout.yaxis.type == "log"


def test_solver_color_map_folds_past_reliable_limit():
    names = [f"solver_{i}" for i in range(6)]
    colors = solver_color_map(names)
    assert len(set(colors.values())) <= 5  # 4 distinct + 1 shared "other" gray
    assert colors["solver_4"] == colors["solver_5"] == "#898781"
