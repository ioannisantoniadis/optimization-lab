import numpy as np
import pytest
from scipy.optimize import minimize

from optimlab.landscapes import get
from optimlab.optimizers.nelder_mead import nelder_mead


def test_converges_on_convex_sphere():
    bf = get("sphere")
    problem = bf.problem(n_dim=3, seed=0)
    result = nelder_mead(problem)
    assert result.converged
    np.testing.assert_allclose(result.x, np.zeros(3), atol=1e-3)


def test_converges_on_rosenbrock_valley():
    bf = get("rosenbrock")
    problem = bf.problem(x0=np.array([-1.2, 1.0]))
    result = nelder_mead(problem, max_iter=1000)
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-2)


def test_lands_on_one_of_himmelblaus_four_known_minima():
    bf = get("himmelblau")
    problem = bf.problem(x0=np.array([0.0, 0.0]))
    result = nelder_mead(problem)
    assert result.converged
    distances = [np.linalg.norm(result.x - m) for m in bf.minima]
    assert min(distances) < 1e-2


def test_never_calls_the_gradient():
    """The whole point of a derivative-free method -- assert it directly rather than
    just trusting the docstring.
    """
    bf = get("sphere")
    problem = bf.problem(n_dim=2, seed=0)

    def exploding_grad(_x):
        raise AssertionError("nelder_mead must never call problem.grad")

    problem.grad = exploding_grad
    result = nelder_mead(problem, max_iter=50)
    assert result.n_iter > 0


def test_matches_scipy_nelder_mead_objective_value():
    bf = get("rosenbrock")
    x0 = np.array([-1.2, 1.0])
    result = nelder_mead(bf.problem(x0=x0))
    ref = minimize(bf.f, x0, method="Nelder-Mead")
    assert result.f == pytest.approx(ref.fun, abs=1e-4)


def test_spread_of_simplex_shrinks_monotonically_near_the_end():
    """g_norm_trajectory (std of f across the simplex) should trend toward zero as the
    simplex collapses onto the optimum -- not strictly monotonic every step (a
    reflection can briefly worsen the spread), but the final value should be far below
    the initial one.
    """
    bf = get("sphere")
    result = nelder_mead(bf.problem(n_dim=2, seed=1))
    assert result.grad_norm_trajectory[-1] < result.grad_norm_trajectory[0] * 1e-3
