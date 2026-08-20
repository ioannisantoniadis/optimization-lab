"""Vanilla gradient descent (book §2.1): x_{k+1} = x_k - alpha * grad f(x_k)."""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations
from optimlab.optimizers.line_search import backtracking_armijo


def gradient_descent(
    problem: Problem,
    *,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    line_search: bool = False,
) -> OptimizeResult:
    """Fixed-step gradient descent, or Armijo-backtracking gradient descent if
    `line_search=True` (using `lr` as the initial step-length guess each iteration).

    Fixed-step GD is the right first algorithm precisely because it's fragile: on an
    ill-conditioned quadratic (see `tests/test_optimizers.py::test_gd_zigzags_on_ill_conditioned`)
    it visibly zig-zags across the narrow valley, which is the intuition every later
    method (momentum, Newton, quasi-Newton) is built to fix.
    """
    x = problem.x0.copy()
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        if line_search:
            alpha = backtracking_armijo(problem.f, problem.grad, x, -g, f_x=f_x, grad_x=g, alpha0=lr)
        else:
            alpha = lr
        x = x - alpha * g
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist,
        f_hist,
        g_hist,
        converged=converged,
        solver_name="gradient_descent" + ("_armijo" if line_search else ""),
        message="gradient norm below tol" if converged else "max_iter reached",
    )
