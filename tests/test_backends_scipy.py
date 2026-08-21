import jax.numpy as jnp
import numpy as np
import pytest

from optimlab.backends import scipy_linprog, scipy_nonlinear_least_squares
from optimlab.optimizers import LinearProgram, NonlinearLeastSquaresProblem, gauss_newton, simplex

_CLASSIC_LP = LinearProgram(c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])


def test_scipy_linprog_matches_our_simplex():
    ours = simplex(_CLASSIC_LP)
    oracle = scipy_linprog(_CLASSIC_LP)
    assert oracle.status == "optimal"
    np.testing.assert_allclose(ours.x, oracle.x, atol=1e-6)
    assert ours.objective == pytest.approx(oracle.objective, abs=1e-6)


def test_scipy_least_squares_matches_our_gauss_newton():
    t = np.linspace(0.0, 5.0, 30)
    y = 2.5 * np.exp(-0.7 * t)

    def residual(params):
        a, b = params[0], params[1]
        return a * jnp.exp(-b * t) - y

    problem = NonlinearLeastSquaresProblem(residual=residual, x0=np.array([1.0, 1.0]))
    ours = gauss_newton(problem)
    oracle = scipy_nonlinear_least_squares(problem)

    assert ours.converged and oracle.converged
    np.testing.assert_allclose(ours.x, oracle.x, atol=1e-5)
