import numpy as np
import pytest

from optimlab.optimizers.admm import ADMMProblem, admm
from optimlab.optimizers.proximal_gradient import soft_threshold


def _lasso_admm_problem(A, b, alpha, rho, x0):
    AtA = A.T @ A
    Atb = A.T @ b
    n = A.shape[1]

    def prox_f(v, t):
        r = 1.0 / t
        return np.linalg.solve(AtA + r * np.eye(n), Atb + r * v)

    def prox_g(v, t):
        return soft_threshold(v, alpha * t)

    return ADMMProblem(
        prox_f=prox_f, prox_g=prox_g, x0=x0,
        f_obj=lambda x: 0.5 * np.sum((A @ x - b) ** 2),
        g_obj=lambda x: alpha * np.sum(np.abs(x)),
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

    problem = _lasso_admm_problem(A, b, alpha, rho=20.0, x0=np.zeros(n))
    result = admm(problem, rho=20.0, max_iter=2000, tol=1e-8)

    x = cvxpy.Variable(n)
    cvx_problem = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.sum_squares(A @ x - b) + alpha * cvxpy.norm1(x)))
    cvx_problem.solve()

    assert result.converged
    np.testing.assert_allclose(result.x, x.value, atol=1e-4)


def test_lasso_matches_proximal_gradient_on_the_same_problem():
    """Two entirely different algorithms (alternating proximal steps + a dual variable
    vs. a single gradient+proximal step) solving the identical LASSO problem should
    land in the same place -- a second independent check beyond the cvxpy oracle.
    """
    from optimlab.optimizers.proximal_gradient import CompositeProblem, proximal_gradient

    rng = np.random.default_rng(1)
    m, n = 60, 25
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[[3, 8, 15]] = [2.0, -1.5, 1.0]
    b = A @ x_true + 0.005 * rng.standard_normal(m)
    alpha = 0.5

    admm_problem = _lasso_admm_problem(A, b, alpha, rho=20.0, x0=np.zeros(n))
    admm_result = admm(admm_problem, rho=20.0, max_iter=2000, tol=1e-8)

    L = np.linalg.eigvalsh(A.T @ A).max()
    pg_problem = CompositeProblem(
        grad_smooth=lambda x: A.T @ (A @ x - b),
        prox_nonsmooth=lambda v, t: soft_threshold(v, alpha * t),
        x0=np.zeros(n),
    )
    pg_result = proximal_gradient(pg_problem, lr=1.0 / L, max_iter=5000, tol=1e-10)

    np.testing.assert_allclose(admm_result.x, pg_result.x, atol=1e-3)


def test_always_runs_at_least_one_iteration():
    """x0 and z0 start identical, so the naive primal residual at iteration 0 is
    trivially zero -- must not be mistaken for convergence before any work is done.
    """
    A = np.eye(2)
    b = np.array([5.0, -3.0])
    problem = _lasso_admm_problem(A, b, alpha=0.1, rho=1.0, x0=np.zeros(2))
    result = admm(problem, rho=1.0, max_iter=500, tol=1e-10)
    assert result.n_iter > 0
    assert result.converged


def test_primal_residual_trends_toward_zero():
    """residuals[0] is trivially 0 (x0 == z0 by construction, before any real work) --
    the meaningful check is that the residual becomes nonzero once ADMM actually starts
    reconciling x and z, then settles back near zero once it converges.
    """
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 10))
    b = rng.standard_normal(30)
    problem = _lasso_admm_problem(A, b, alpha=0.2, rho=10.0, x0=np.zeros(10))
    result = admm(problem, rho=10.0, max_iter=1000, tol=1e-10)
    residuals = np.asarray(result.grad_norm_trajectory)
    assert residuals[0] == 0.0
    assert residuals[1] > 1e-6  # real disagreement appears after the first real step
    assert residuals[-1] < 1e-6  # and is resolved by the end
