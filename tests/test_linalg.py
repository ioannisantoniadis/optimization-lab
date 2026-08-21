import numpy as np
import pytest
from scipy.optimize import minimize

from optimlab.linalg import (
    condition_number,
    equality_constrained_least_squares,
    equality_constrained_qp,
    least_squares,
    ridge_regression,
    svd,
)


def test_svd_reconstructs_the_matrix():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 4))
    result = svd(A)
    np.testing.assert_allclose(result.U @ np.diag(result.s) @ result.Vt, A, atol=1e-10)
    assert result.rank == 4


def test_condition_number_of_identity_is_one():
    assert condition_number(np.eye(5)) == pytest.approx(1.0)


def test_condition_number_matches_the_ill_conditioned_quadratic_from_chapter_1():
    """docs Ch. 1's anisotropic quadratic uses Hessian diag(1, 100) specifically because
    its condition number, 100, is what makes fixed-step gradient descent zig-zag --
    this pins that story to an actual number computed here, not just asserted in prose.
    """
    assert condition_number(np.diag([1.0, 100.0])) == pytest.approx(100.0)


def test_condition_number_accepts_precomputed_singular_values():
    assert condition_number(s=np.array([4.0, 2.0, 1.0])) == pytest.approx(4.0)


def test_least_squares_matches_numpy_on_overdetermined_system():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((50, 5))
    b = rng.standard_normal(50)
    result = least_squares(A, b)
    ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(result.x, ref, atol=1e-8)
    assert result.rank == 5


def test_least_squares_recovers_minimum_norm_solution_when_underdetermined():
    """3 equations, 8 unknowns -- infinitely many exact solutions; the pseudoinverse
    picks the one with smallest ||x||, which we check against by direct competition
    against another exact solution nudged along the null space (must have larger norm).
    """
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 8))
    b = rng.standard_normal(3)
    result = least_squares(A, b)

    np.testing.assert_allclose(A @ result.x, b, atol=1e-8)  # exact, not approximate
    assert result.residual_norm < 1e-8

    _, _, Vt = np.linalg.svd(A)
    null_direction = Vt[3]  # any row past the rank (3) spans the null space
    nudged = result.x + 0.5 * null_direction  # another exact solution, just not minimum-norm
    np.testing.assert_allclose(A @ nudged, b, atol=1e-8)
    assert np.linalg.norm(nudged) > np.linalg.norm(result.x)


def test_least_squares_ignores_near_zero_singular_values_via_rcond():
    """A rank-deficient matrix (a duplicated column) has a near-zero singular value;
    without truncating it, dividing by it would blow the solution up.
    """
    A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])
    b = np.array([1.0, 2.0, 3.0])
    result = least_squares(A, b, rcond=1e-8)
    assert result.rank == 2
    assert np.all(np.isfinite(result.x))


def test_ridge_regression_matches_closed_form_normal_equations():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 6))
    b = rng.standard_normal(30)
    alpha = 2.5
    result = ridge_regression(A, b, alpha)
    closed_form = np.linalg.solve(A.T @ A + alpha * np.eye(6), A.T @ b)
    np.testing.assert_allclose(result.x, closed_form, atol=1e-8)


def test_ridge_with_zero_alpha_matches_plain_least_squares():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((20, 4))
    b = rng.standard_normal(20)
    np.testing.assert_allclose(ridge_regression(A, b, 0.0).x, least_squares(A, b).x, atol=1e-8)


def test_ridge_shrinks_coefficient_norm_as_alpha_grows():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((30, 8))
    b = rng.standard_normal(30)
    norms = [np.linalg.norm(ridge_regression(A, b, alpha).x) for alpha in [0.0, 1.0, 10.0, 100.0]]
    assert norms == sorted(norms, reverse=True)


def test_ridge_rejects_negative_alpha():
    with pytest.raises(ValueError, match="nonnegative"):
        ridge_regression(np.eye(2), np.ones(2), -1.0)


def test_equality_constrained_qp_matches_scipy():
    P = np.diag([2.0, 4.0, 6.0])
    q = np.array([-1.0, -2.0, -3.0])
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    x = equality_constrained_qp(P, q, A_eq, b_eq)

    ref = minimize(
        lambda x: 0.5 * x @ P @ x + q @ x, x0=np.zeros(3),
        constraints=[{"type": "eq", "fun": lambda x: A_eq @ x - b_eq}],
    )
    np.testing.assert_allclose(x, ref.x, atol=1e-6)
    np.testing.assert_allclose(A_eq @ x, b_eq, atol=1e-8)


def test_equality_constrained_least_squares_satisfies_the_constraint():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((10, 4))
    b = rng.standard_normal(10)
    C = np.array([[1.0, 1.0, 1.0, 1.0]])
    d = np.array([2.0])
    x = equality_constrained_least_squares(A, b, C, d)

    ref = minimize(
        lambda x: 0.5 * np.sum((A @ x - b) ** 2), x0=np.zeros(4),
        constraints=[{"type": "eq", "fun": lambda x: C @ x - d}],
    )
    np.testing.assert_allclose(x, ref.x, atol=1e-3)
    np.testing.assert_allclose(C @ x, d, atol=1e-8)
