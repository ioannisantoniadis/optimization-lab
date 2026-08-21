import numpy as np
import pytest

from optimlab.optimizers import conjugate_gradient


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)  # guarantees positive-definiteness


@pytest.mark.parametrize("n", [2, 5, 20, 50])
def test_converges_within_n_steps_and_matches_direct_solve(n):
    rng = np.random.default_rng(n)
    A = _random_spd(n, seed=n)
    x_true = rng.standard_normal(n)
    b = A @ x_true

    result = conjugate_gradient(A, b, tol=1e-10)

    assert result.converged
    assert result.n_iter <= n  # the textbook exact-arithmetic bound
    np.testing.assert_allclose(result.x, x_true, atol=1e-6)
    np.testing.assert_allclose(A @ result.x, b, atol=1e-6)


def test_matches_numpy_solve_on_sphere_problem():
    """A = I reduces the quadratic to the sphere function -- CG should reach the exact
    minimum (x = b) in a single step, since every direction already has identical
    curvature (no need to build up conjugate directions).
    """
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    result = conjugate_gradient(A, b)
    assert result.n_iter == 1
    np.testing.assert_allclose(result.x, b, atol=1e-10)


def test_respects_custom_starting_point():
    A = _random_spd(6, seed=42)
    x_true = np.ones(6)
    b = A @ x_true
    result = conjugate_gradient(A, b, x0=x_true.copy(), tol=1e-10)
    assert result.n_iter == 0
    assert result.converged
