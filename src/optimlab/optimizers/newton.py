"""Newton's method (book §2.4): use curvature (the Hessian), not just slope, to jump
straight toward a stationary point. Converges quadratically near a strict local min —
dramatically fewer iterations than gradient descent — but each iteration costs an O(n^3)
linear solve, which is exactly why this doesn't scale to million/billion-parameter
problems and why quasi-Newton methods (`quasi_newton.py`) exist.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations
from optimlab.optimizers.line_search import backtracking_armijo


def newton_method(
    problem: Problem,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    line_search: bool = True,
    min_eigenvalue: float = 1e-6,
) -> OptimizeResult:
    """Damped Newton's method: solve H @ p = -g for the step, then take a full or
    line-searched step along `p`.

    Away from a convex bowl the Hessian can be indefinite, in which case the raw Newton
    step points toward a saddle point rather than a minimum — it's solving "where does
    the local quadratic model have zero gradient," which for an indefinite Hessian is a
    saddle. We fix this the standard way (a small piece of the Levenberg-Marquardt idea):
    check the Hessian's eigenvalues, and if the smallest one isn't safely positive, add
    `(min_eigenvalue - lambda_min) * I` before solving. This is also a direct, hands-on
    instance of the "critical-point index depends on the sign pattern of the Hessian
    eigenvalues" story from the high-dimensional non-convexity module (ROADMAP Phase 6).
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
        H = problem.hess(x)
        eigvals = np.linalg.eigvalsh(H)
        lambda_min = eigvals.min()
        if lambda_min < min_eigenvalue:
            H = H + (min_eigenvalue - lambda_min) * np.eye(H.shape[0])
        direction = np.linalg.solve(H, -g)

        if line_search:
            alpha = backtracking_armijo(problem.f, problem.grad, x, direction, f_x=f_x, grad_x=g, alpha0=1.0)
        else:
            alpha = 1.0

        x = x + alpha * direction
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="newton",
        message="gradient norm below tol" if converged else "max_iter reached",
    )
