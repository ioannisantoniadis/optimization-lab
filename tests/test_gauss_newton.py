import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares

from optimlab.optimizers import NonlinearLeastSquaresProblem, gauss_newton


def test_recovers_exact_params_from_noiseless_exponential_decay_data():
    t = np.linspace(0.0, 5.0, 30)
    a_true, b_true = 2.5, 0.7
    y = a_true * np.exp(-b_true * t)

    def residual(params):
        a, b = params[0], params[1]
        return a * jnp.exp(-b * t) - y

    problem = NonlinearLeastSquaresProblem(residual=residual, x0=np.array([1.0, 1.0]))
    result = gauss_newton(problem)

    assert result.converged
    np.testing.assert_allclose(result.x, [a_true, b_true], atol=1e-6)
    assert result.n_iter < 20  # Gauss-Newton on a clean fit should be fast


def test_matches_scipy_least_squares_on_noisy_data():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 5.0, 40)
    a_true, b_true = 3.0, 0.5
    y = a_true * np.exp(-b_true * t) + rng.normal(scale=0.01, size=t.size)

    def residual_np(params):
        a, b = params[0], params[1]
        return a * np.exp(-b * t) - y

    def residual_jax(params):
        a, b = params[0], params[1]
        return a * jnp.exp(-b * t) - y

    problem = NonlinearLeastSquaresProblem(residual=residual_jax, x0=np.array([1.0, 1.0]))
    result = gauss_newton(problem)
    ref = scipy_least_squares(residual_np, x0=np.array([1.0, 1.0]))

    assert result.converged
    np.testing.assert_allclose(result.x, ref.x, atol=1e-4)


def test_uses_explicit_jacobian_when_provided():
    """A linear residual r(x) = A x - b makes Gauss-Newton exact in one step -- and its
    Jacobian is just A, constant everywhere, the simplest possible hand-derived case.
    """
    A = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    b = np.array([4.0, 9.0, 5.0])

    def residual(x):
        return A @ x - b

    def jacobian(_x):
        return A

    problem = NonlinearLeastSquaresProblem(residual=residual, x0=np.zeros(2), jacobian=jacobian)
    result = gauss_newton(problem, max_iter=5)
    assert result.converged
    assert result.n_iter == 1

    ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(result.x, ref, atol=1e-8)


def test_finite_difference_jacobian_matches_jax_jacobian():
    """`_finite_diff_jacobian` only actually gets used if JAX itself is unavailable
    (mirroring `optimlab.core.Problem`'s autodiff fallback) -- a plain-numpy residual
    doesn't trigger it, it just breaks under `jax.jacfwd`'s tracing, so exercise the
    finite-difference path directly and check it agrees with the JAX one on the same
    residual, rather than relying on some indirect trigger.
    """
    from optimlab.optimizers.gauss_newton import _finite_diff_jacobian

    def residual_np(x):
        return np.array([x[0] ** 2 - 1.0, x[0] * x[1] - 2.0])

    def residual_jax(x):
        return jnp.array([x[0] ** 2 - 1.0, x[0] * x[1] - 2.0])

    x = np.array([2.0, 3.0])
    fd_jacobian = _finite_diff_jacobian(residual_np, x)

    jax_problem = NonlinearLeastSquaresProblem(residual=residual_jax, x0=x)
    jax_jacobian = jax_problem.jacobian(x)

    np.testing.assert_allclose(fd_jacobian, jax_jacobian, atol=1e-6)


@pytest.mark.parametrize("x0", [[1.0, 1.0], [0.5, 3.0]])
def test_converges_on_a_second_landscape_benchmark(x0):
    """Rosenbrock is a sum of squares (100(y-x^2)^2 + (1-x)^2), so it doubles as a
    nonlinear-least-squares residual problem -- ties this solver back to the same
    benchmark used throughout optimlab.landscapes.
    """
    def residual(x):
        return jnp.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

    problem = NonlinearLeastSquaresProblem(residual=residual, x0=np.array(x0))
    result = gauss_newton(problem, max_iter=50)
    assert result.converged
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)
