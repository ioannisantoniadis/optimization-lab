"""Quasi-Newton methods (book §2.6): get Newton's curvature-awareness without ever
forming or inverting a real Hessian. Both methods here build up an *approximate* inverse
Hessian purely from the gradients already computed along the way (the "secant"
information s=x_{k+1}-x_k, y=g_{k+1}-g_k), which is what makes them the practical default
for large-scale smooth optimization (L-BFGS in particular is the backbone of, e.g.,
`scipy.optimize.minimize(method="L-BFGS-B")`).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations
from optimlab.optimizers.line_search import strong_wolfe_line_search


def bfgs(
    problem: Problem,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> OptimizeResult:
    """BFGS: maintain a dense n x n approximate inverse Hessian `H`, updated each step by
    the rank-2 BFGS formula so that `H` satisfies the secant equation `H @ y = s`. O(n^2)
    per iteration (vs. Newton's O(n^3) solve) but still O(n^2) memory — fine up to a few
    thousand parameters, not to millions (see `lbfgs` below).
    """
    x = problem.x0.copy()
    n = problem.n_dim
    H = np.eye(n)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        p = -H @ g
        alpha = strong_wolfe_line_search(problem.f, problem.grad, x, p, f_x=f_x, grad_x=g)
        x_new = x + alpha * p
        g_new = problem.grad(x_new)

        s = x_new - x
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:  # skip the update if curvature condition s^T y > 0 fails
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x, g = x_new, g_new
        f_x = float(problem.f(x))
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="bfgs",
        message="gradient norm below tol" if converged else "max_iter reached",
    )


def lbfgs(
    problem: Problem,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    memory: int = 10,
) -> OptimizeResult:
    """L-BFGS: never form `H` at all. Instead keep only the last `memory` (s, y) pairs and
    reconstruct the effect of `H @ g` on demand via the two-loop recursion (Nocedal &
    Wright Algorithm 7.4). Memory drops from O(n^2) to O(memory * n) — the difference
    between "unusable past a few thousand parameters" and "the thing PyTorch/SciPy
    actually ship," which matters directly for the million/billion-parameter framing
    this repo cares about.
    """
    x = problem.x0.copy()
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    s_hist: deque[np.ndarray] = deque(maxlen=memory)
    y_hist: deque[np.ndarray] = deque(maxlen=memory)
    rho_hist: deque[float] = deque(maxlen=memory)

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break

        q = g.copy()
        alphas = []
        for s, y, rho in reversed(list(zip(s_hist, y_hist, rho_hist, strict=True))):
            a = rho * (s @ q)
            q = q - a * y
            alphas.append(a)
        alphas.reverse()

        if y_hist:
            gamma = (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
        else:
            gamma = 1.0
        r = gamma * q

        for (s, y, rho), a in zip(zip(s_hist, y_hist, rho_hist, strict=True), alphas, strict=True):
            beta = rho * (y @ r)
            r = r + s * (a - beta)

        p = -r
        alpha = strong_wolfe_line_search(problem.f, problem.grad, x, p, f_x=f_x, grad_x=g)
        x_new = x + alpha * p
        g_new = problem.grad(x_new)

        s = x_new - x
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)

        x, g = x_new, g_new
        f_x = float(problem.f(x))
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="lbfgs",
        message="gradient norm below tol" if converged else "max_iter reached",
    )
