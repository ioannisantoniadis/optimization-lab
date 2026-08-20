"""Momentum methods (book §2.2): accumulate a velocity term so the optimizer keeps
moving through directions of persistent gradient signal and damps oscillation across
directions where the gradient keeps flipping sign — exactly the ill-conditioned-valley
failure mode `gradient_descent` demonstrates.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations


def heavy_ball(
    problem: Problem,
    *,
    lr: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Polyak's heavy-ball method: v_{k+1} = beta*v_k - lr*grad(x_k); x_{k+1} = x_k + v_{k+1}.

    Like a ball with mass rolling downhill: `beta` controls how much of the previous
    velocity survives, so the optimizer builds up speed along consistent-gradient
    directions and the accumulated momentum partially cancels across
    oscillating (sign-flipping) directions.
    """
    x = problem.x0.copy()
    v = np.zeros_like(x)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        v = beta * v - lr * g
        x = x + v
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="heavy_ball",
        message="gradient norm below tol" if converged else "max_iter reached",
    )


def nesterov(
    problem: Problem,
    *,
    lr: float = 0.01,
    beta: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Nesterov accelerated gradient: evaluate the gradient at a "lookahead" point
    `x_k + beta*v_k` rather than at `x_k` itself, then update velocity/position from
    there. That lookahead is the entire difference from heavy-ball, and it's what gives
    Nesterov's method its provably faster O(1/k^2) convergence rate on convex problems
    (vs. heavy-ball's O(1/k)) — the gradient gets a chance to "correct" the momentum step
    before it's committed.
    """
    x = problem.x0.copy()
    v = np.zeros_like(x)
    f_x = float(problem.f(x))
    g = problem.grad(x)
    x_hist, f_hist, g_hist = [x.copy()], [f_x], [float(np.linalg.norm(g))]

    converged = False
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        lookahead = x + beta * v
        g_lookahead = problem.grad(lookahead)
        v = beta * v - lr * g_lookahead
        x = x + v
        f_x = float(problem.f(x))
        g = problem.grad(x)
        x_hist.append(x.copy())
        f_hist.append(f_x)
        g_hist.append(float(np.linalg.norm(g)))

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="nesterov",
        message="gradient norm below tol" if converged else "max_iter reached",
    )
