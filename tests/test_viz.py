import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.core import Problem
from optimlab.landscapes import get
from optimlab.optimizers import bfgs, gradient_descent, newton_method
from optimlab.viz import (
    contour_figure,
    convergence_figure,
    race_figure,
    solver_color_map,
    surface_figure,
    surface_race_figure,
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


def test_surface_figure_defaults_to_log_z():
    """log_z=False on a steep function like Rosenbrock would collapse the minimum's
    neighborhood to a razor-thin spike — the whole point of the log transform.
    """
    bf = get("rosenbrock")
    fig = surface_figure(bf.problem(x0=[-1.0, 1.0]), resolution=20)
    assert "log10" in fig.data[0].colorbar.title.text


def test_surface_race_figure_overlays_one_3d_trajectory_per_solver():
    bf, results = _himmelblau_results()
    fig = surface_race_figure(bf.problem(x0=[-4.0, 4.0]), results, resolution=25)
    assert isinstance(fig.data[0], go.Surface)
    # 1 surface + 2 Scatter3d traces (line, endpoint) per solver
    assert len(fig.data) == 1 + 2 * len(results)
    for trace in fig.data[1:]:
        assert isinstance(trace, go.Scatter3d)


def test_surface_race_figure_rejects_non_2d_problem():
    bf = get("sphere")
    with pytest.raises(ValueError, match="2D"):
        surface_race_figure(bf.problem(n_dim=5), {})


def test_race_figure_overlays_one_trajectory_per_solver():
    bf, results = _himmelblau_results()
    fig = race_figure(bf.problem(x0=[-4.0, 4.0]), results)
    # 1 contour + 1 known-minima marker + 2 traces (line, endpoint) per solver
    assert len(fig.data) == 2 + 2 * len(results)


def test_convergence_figure_pads_converged_solvers_to_a_common_length():
    """Newton/BFGS converge in far fewer iterations than gradient_descent here, so
    without padding their lines would look cut off rather than flat-at-the-optimum.
    """
    _, results = _himmelblau_results()
    fig = convergence_figure(results)
    # each solver gets a real trace, plus a padding trace for each that converged
    # earlier than the longest-running one
    max_len = max(r.n_iter + 1 for r in results.values())
    n_padded = sum(1 for r in results.values() if r.converged and r.n_iter + 1 < max_len)
    assert len(fig.data) == len(results) + n_padded
    assert n_padded > 0  # sanity check this scenario actually exercises padding
    assert fig.layout.yaxis.type == "log"

    padding_traces = [t for t in fig.data if t.line.dash == "dot"]
    assert len(padding_traces) == n_padded
    for trace in padding_traces:
        assert trace.x[-1] == max_len - 1
        assert trace.showlegend is False


def test_convergence_figure_does_not_pad_a_solver_that_never_converged():
    """A solver that hit max_iter without converging should end exactly where it
    stopped — padding it flat would misrepresent an unfinished descent as a plateau.
    """
    def f(x):
        return 0.5 * (x[0] ** 2 + 100.0 * x[1] ** 2)

    def grad(x):
        return np.array([x[0], 100.0 * x[1]])

    unconverged = gradient_descent(
        Problem(f=f, x0=np.array([10.0, 1.0]), grad=grad), lr=0.005, max_iter=20, tol=1e-10
    )
    assert not unconverged.converged
    fig = convergence_figure({"gradient_descent": unconverged})
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == unconverged.n_iter + 1


def test_convergence_figure_pad_converged_can_be_disabled():
    _, results = _himmelblau_results()
    fig = convergence_figure(results, pad_converged=False)
    assert len(fig.data) == len(results)


def test_solver_color_map_folds_past_reliable_limit():
    names = [f"solver_{i}" for i in range(6)]
    colors = solver_color_map(names)
    assert len(set(colors.values())) <= 5  # 4 distinct + 1 shared "other" gray
    assert colors["solver_4"] == colors["solver_5"] == "#898781"
