"""Projected gradient descent: gradient descent's fix for box constraints (`lower <= x
<= upper`) — take an ordinary gradient step, then snap back onto the feasible box by
clipping. Solves the "easy half" of general quadratic programming (book §4.6): boxes are
the inequality constraints simple enough that no active-set bookkeeping is needed, unlike
general linear inequalities. `optimlab.linalg.qp.equality_constrained_qp` covers the
complementary easy half (equalities, no inequalities) — the general case with both,
worked out via full KKT conditions, is Phase 4.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike, OptimizeResult, Problem, track_iterations


def _project(x: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> ArrayLike:
    return np.clip(x, lower, upper)


def projected_gradient(
    problem: Problem,
    *,
    lower: ArrayLike | float,
    upper: ArrayLike | float,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Minimize `problem.f` subject to `lower <= x <= upper` (each may be a scalar,
    broadcast to every coordinate, or a per-coordinate array).

    At a constrained optimum sitting *on* the boundary, the raw gradient generally isn't
    zero (the constraint is what's holding `x` there) — so convergence is checked via
    the *projected*-gradient norm `||x - Proj(x - lr*grad(x))|| / lr` instead of the raw
    gradient norm every unconstrained solver in this repo uses. That quantity is exactly
    zero when there's no feasible direction left that would decrease `f`, whether or not
    `x` is on the boundary, which is the right generalization of "gradient is zero" once
    a constraint can hold you in place.
    """
    n = problem.n_dim
    lower_arr = np.broadcast_to(np.asarray(lower, dtype=float), (n,))
    upper_arr = np.broadcast_to(np.asarray(upper, dtype=float), (n,))

    x = _project(problem.x0.copy(), lower_arr, upper_arr)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        g = problem.grad(x)
        x_next = _project(x - lr * g, lower_arr, upper_arr)
        projected_grad_norm = float(np.linalg.norm(x - x_next) / lr)

        x = x_next
        f_x = float(problem.f(x))
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(projected_grad_norm)

        if projected_grad_norm < tol:
            converged = True
            break

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="projected_gradient",
        message="projected gradient norm below tol" if converged else "max_iter reached",
    )
