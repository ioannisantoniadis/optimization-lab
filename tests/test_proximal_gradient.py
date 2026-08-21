import numpy as np
import pytest

from optimlab.optimizers.proximal_gradient import (
    CompositeProblem,
    proximal_gradient,
    soft_threshold,
)


def test_soft_threshold_zeros_a_dead_zone_around_the_origin():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(soft_threshold(x, 1.0), [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_soft_threshold_is_the_identity_at_zero_threshold():
    x = np.array([-3.0, 0.2, 5.0])
    np.testing.assert_allclose(soft_threshold(x, 0.0), x)


def _lasso_problem(A, b, alpha, x0):
    return CompositeProblem(
        grad_smooth=lambda x: A.T @ (A @ x - b),
        prox_nonsmooth=lambda v, t: soft_threshold(v, alpha * t),
        x0=x0,
        f_smooth=lambda x: 0.5 * np.sum((A @ x - b) ** 2),
        f_nonsmooth=lambda x: alpha * np.sum(np.abs(x)),
    )


def test_lasso_matches_cvxpy():
    cvxpy = pytest.importorskip("cvxpy", reason="requires the 'backends' extra")

    rng = np.random.default_rng(0)
    m, n = 50, 20
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[[2, 5, 9]] = [3.0, -2.0, 1.5]
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    alpha = 1.0

    L = np.linalg.eigvalsh(A.T @ A).max()
    problem = _lasso_problem(A, b, alpha, np.zeros(n))
    result = proximal_gradient(problem, lr=1.0 / L, max_iter=5000, tol=1e-10)

    x = cvxpy.Variable(n)
    cvx_problem = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.sum_squares(A @ x - b) + alpha * cvxpy.norm1(x)))
    cvx_problem.solve()

    assert result.converged
    np.testing.assert_allclose(result.x, x.value, atol=1e-4)


def test_lasso_recovers_a_sparse_ground_truth():
    rng = np.random.default_rng(1)
    m, n = 60, 25
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    nonzero_indices = [3, 8, 15]
    x_true[nonzero_indices] = [2.0, -1.5, 1.0]
    b = A @ x_true + 0.005 * rng.standard_normal(m)

    L = np.linalg.eigvalsh(A.T @ A).max()
    problem = _lasso_problem(A, b, alpha=0.5, x0=np.zeros(n))
    result = proximal_gradient(problem, lr=1.0 / L, max_iter=5000, tol=1e-10)

    recovered_nonzero = set(np.flatnonzero(np.abs(result.x) > 1e-6))
    assert recovered_nonzero == set(nonzero_indices)


def test_proximal_gradient_reduces_to_plain_gradient_descent_without_a_penalty():
    """h identically zero -- prox of zero-lr-scaled nothing is the identity, so this
    should behave exactly like unconstrained gradient descent on f_smooth alone.
    """
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    b = np.array([4.0, 9.0])
    problem = CompositeProblem(
        grad_smooth=lambda x: A @ x - b,
        prox_nonsmooth=lambda v, t: v,  # h = 0, prox is the identity
        x0=np.zeros(2),
        f_smooth=lambda x: 0.5 * x @ A @ x - b @ x,
    )
    result = proximal_gradient(problem, lr=0.3, max_iter=200, tol=1e-10)
    assert result.converged
    np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-4)


def test_larger_alpha_produces_a_sparser_solution():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 15))
    b = rng.standard_normal(40)
    L = np.linalg.eigvalsh(A.T @ A).max()

    sparsities = []
    for alpha in [0.01, 1.0, 10.0]:
        problem = _lasso_problem(A, b, alpha, np.zeros(15))
        result = proximal_gradient(problem, lr=1.0 / L, max_iter=3000, tol=1e-10)
        sparsities.append(int(np.sum(np.abs(result.x) > 1e-6)))
    assert sparsities[0] >= sparsities[1] >= sparsities[2]
