import numpy as np
import pytest
from scipy.optimize import minimize

from optimlab.core import Problem
from optimlab.optimizers import projected_gradient


def _box_qp_problem(P, q, x0):
    def f(x):
        return 0.5 * x @ P @ x + q @ x

    def grad(x):
        return P @ x + q

    return Problem(f=f, x0=np.asarray(x0, dtype=float), grad=grad)


def test_clips_to_the_boundary_when_unconstrained_optimum_is_outside_the_box():
    problem = _box_qp_problem(np.diag([2.0, 2.0]), np.array([-6.0, -6.0]), [0.0, 0.0])
    result = projected_gradient(problem, lower=0.0, upper=2.0, lr=0.1, max_iter=500)
    assert result.converged
    np.testing.assert_allclose(result.x, [2.0, 2.0], atol=1e-4)


def test_matches_unconstrained_optimum_when_it_already_lies_inside_the_box():
    problem = _box_qp_problem(np.diag([2.0, 2.0]), np.array([-2.0, -2.0]), [0.0, 0.0])
    result = projected_gradient(problem, lower=-10.0, upper=10.0, lr=0.1, max_iter=500)
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)


def test_matches_scipy_minimize_with_bounds_on_a_random_box_qp():
    rng = np.random.default_rng(0)
    n = 5
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    q = rng.uniform(-5, 5, size=n)
    lower, upper = -1.0, 1.0

    problem = _box_qp_problem(P, q, np.zeros(n))
    result = projected_gradient(problem, lower=lower, upper=upper, lr=1.0 / np.linalg.eigvalsh(P).max(), max_iter=5000)

    ref = minimize(
        lambda x: 0.5 * x @ P @ x + q @ x, x0=np.zeros(n),
        bounds=[(lower, upper)] * n, method="L-BFGS-B",
    )
    assert result.converged
    np.testing.assert_allclose(result.x, ref.x, atol=1e-3)
    assert result.f == pytest.approx(ref.fun, abs=1e-4)


def test_per_coordinate_bounds():
    problem = _box_qp_problem(np.diag([2.0, 2.0]), np.array([-6.0, -6.0]), [0.0, 0.0])
    result = projected_gradient(
        problem, lower=[0.0, 0.0], upper=[1.0, 5.0], lr=0.1, max_iter=1000
    )
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 3.0], atol=1e-4)
