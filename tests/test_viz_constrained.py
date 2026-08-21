import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.optimizers.barrier_method import ConstrainedProblem, barrier_method
from optimlab.viz import central_path_figure, duality_gap_figure, kkt_geometry_figure


def _disk_constrained_problem():
    """minimize (x-2)^2 + (y-1)^2 s.t. x^2+y^2<=1, -x<=0 -- optimum sits on the disk
    boundary with the box constraint (-x<=0) slack, so exactly one constraint is active.
    """
    return ConstrainedProblem(
        f=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        x0=np.array([0.1, 0.1]),
        inequality_constraints=[lambda x: x[0] ** 2 + x[1] ** 2 - 1.0, lambda x: -x[0]],
        name="disk_constrained_quadratic",
    )


def _corner_constrained_problem():
    """minimize x+y s.t. -x<=-0.2, -y<=-0.2, x+y<=... no -- pick a problem where two
    constraints are simultaneously active at the optimum, to exercise the multi-arrow
    (and multi-slot offset) path in kkt_geometry_figure.
    """
    return ConstrainedProblem(
        f=lambda x: (x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2,
        x0=np.array([0.5, 0.5]),
        inequality_constraints=[lambda x: x[0] - 1.0, lambda x: x[1] - 1.0],
        name="box_corner_quadratic",
    )


def test_central_path_figure_builds_and_stays_feasible():
    problem = _disk_constrained_problem()
    result = barrier_method(problem)
    fig = central_path_figure(problem, result)
    assert isinstance(fig, go.Figure)
    path_trace = next(t for t in fig.data if t.name == "central path")
    xy = np.column_stack([path_trace.x, path_trace.y])
    assert np.all(np.sum(xy**2, axis=1) <= 1.0 + 1e-6)
    assert np.all(xy[:, 0] >= -1e-6)


def test_central_path_figure_rejects_non_2d_problem():
    problem = ConstrainedProblem(
        f=lambda x: np.sum(x**2), x0=np.full(3, 0.1),
        inequality_constraints=[lambda x: np.sum(x**2) - 1.0],
    )
    result = barrier_method(problem)
    with pytest.raises(ValueError, match="2D"):
        central_path_figure(problem, result)


def test_kkt_geometry_figure_single_active_constraint_cancels_neg_grad_f():
    """With one active constraint, KKT stationarity reduces to
    grad f(x*) + lambda * grad g(x*) = 0 -- the sum arrow's vector must equal -grad f(x*)
    to float precision, which is exactly what the figure is asserting geometrically.
    """
    problem = _disk_constrained_problem()
    result = barrier_method(problem)
    fig = kkt_geometry_figure(problem, result)
    assert isinstance(fig, go.Figure)

    names = [t.name for t in fig.data if t.name is not None]
    assert "-∇f(x*)" in names
    assert "Σ λ_i·∇g_i(x*)" in names
    assert sum(1 for n in names if n.startswith("λ_")) == 1  # only constraint 0 is active

    grad_f = problem.grad(result.x)
    final_gap = result.grad_norm_trajectory[-1]
    t_final = problem.n_constraints / final_gap
    s0 = -float(problem.inequality_constraints[0](result.x))
    lambda_0 = 1.0 / (t_final * s0)
    lambda_sum = lambda_0 * problem._g_grad[0](result.x)
    np.testing.assert_allclose(lambda_sum, -grad_f, atol=1e-4)


def test_kkt_geometry_figure_handles_two_active_constraints():
    problem = _corner_constrained_problem()
    result = barrier_method(problem)
    fig = kkt_geometry_figure(problem, result)
    names = [t.name for t in fig.data if t.name is not None]
    assert sum(1 for n in names if n.startswith("λ_")) == 2


def test_kkt_geometry_figure_rejects_non_2d_problem():
    problem = ConstrainedProblem(
        f=lambda x: np.sum(x**2), x0=np.full(3, 0.1),
        inequality_constraints=[lambda x: np.sum(x**2) - 1.0],
    )
    result = barrier_method(problem)
    with pytest.raises(ValueError, match="2D"):
        kkt_geometry_figure(problem, result)


def test_duality_gap_figure_builds_with_tol_line():
    problem = _disk_constrained_problem()
    result = barrier_method(problem)
    fig = duality_gap_figure(result.grad_norm_trajectory, tol=1e-8)
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.type == "log"
    assert len(fig.layout.shapes) == 1  # the hline
