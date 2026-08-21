"""The Gauss-Newton method for nonlinear least squares (book §4.7): fit a nonlinear
model by linearizing its *residuals* (not the objective) at each step and solving the
resulting linear least-squares problem — Newton's method's curvature idea, specialized
to the very common case where the objective is a sum of squared residuals.

`optimlab.core.Problem` doesn't fit here: Gauss-Newton needs the residual vector `r(x)`
and its Jacobian, not just the scalar `f(x)=0.5||r(x)||^2` and its gradient — the
Jacobian carries first-order information Gauss-Newton uses as a curvature *approximation*
(`J^T J ≈ hessian of f`, exact only when the residuals are small or the model is linear
in `x`) without ever computing the true, more expensive Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, GradFn, Objective, OptimizeResult, track_iterations
from optimlab.optimizers.line_search import backtracking_armijo

ResidualFn = Objective  # x -> residual vector r(x), reusing Objective's x -> array shape
JacobianFn = GradFn  # x -> Jacobian matrix dr_i/dx_j


@dataclass
class NonlinearLeastSquaresProblem:
    """Minimize `0.5 * ||residual(x)||^2` over `x`. Leave `jacobian=None` to get it via
    JAX automatic differentiation (`jax.jacfwd`), exactly as `optimlab.core.Problem`
    derives gradients — `residual` should then be written with `jax.numpy` ops.
    """

    residual: ResidualFn
    x0: ArrayLike
    jacobian: JacobianFn | None = None
    name: str = "nonlinear_least_squares"

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.jacobian is None:
            self.jacobian = _autojacobian(self.residual)

    @property
    def n_dim(self) -> int:
        return int(self.x0.size)

    def f(self, x: ArrayLike) -> float:
        r = np.asarray(self.residual(x))
        return 0.5 * float(r @ r)


def _autojacobian(residual: ResidualFn) -> JacobianFn:
    try:
        import jax
        import jax.numpy as jnp

        jac_fn = jax.jit(jax.jacfwd(lambda x: residual(x)))

        def jacobian(x: ArrayLike) -> ArrayLike:
            return np.asarray(jac_fn(jnp.asarray(x, dtype=jnp.float64)))

        return jacobian
    except ImportError:
        return lambda x: _finite_diff_jacobian(residual, x)


def _finite_diff_jacobian(residual: ResidualFn, x: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(residual(x))
    J = np.zeros((r0.size, x.size))
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = eps
        J[:, j] = (np.asarray(residual(x + step)) - np.asarray(residual(x - step))) / (2 * eps)
    return J


def gauss_newton(
    problem: NonlinearLeastSquaresProblem,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    line_search: bool = True,
) -> OptimizeResult:
    """Each step solves `min_delta ||J(x) delta + r(x)||^2` (a *linear* least squares
    problem in `delta`, via `numpy.linalg.lstsq` rather than forming and inverting the
    normal equations `J^T J` explicitly — the same "avoid squaring the condition number"
    reasoning as `optimlab.linalg.least_squares`) and steps to `x + delta`. Optional
    backtracking keeps a step that would otherwise overshoot (Gauss-Newton, unlike
    genuine Newton, has no guarantee its quadratic model is accurate far from `x`) from
    increasing the objective.
    """
    x = problem.x0.copy()
    r = np.asarray(problem.residual(x))
    f_x = 0.5 * float(r @ r)
    J = problem.jacobian(x)
    grad = J.T @ r
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(grad))]

    def f_of(point: ArrayLike) -> float:
        rp = np.asarray(problem.residual(point))
        return 0.5 * float(rp @ rp)

    def grad_of(point: ArrayLike) -> ArrayLike:
        return problem.jacobian(point).T @ np.asarray(problem.residual(point))

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(grad) < tol:
            converged = True
            break

        delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        alpha = (
            backtracking_armijo(f_of, grad_of, x, delta, f_x=f_x, grad_x=grad, alpha0=1.0)
            if line_search
            else 1.0
        )

        x = x + alpha * delta
        r = np.asarray(problem.residual(x))
        f_x = 0.5 * float(r @ r)
        J = problem.jacobian(x)
        grad = J.T @ r

        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(grad)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="gauss_newton",
        message="gradient norm below tol" if converged else "max_iter reached",
    )
