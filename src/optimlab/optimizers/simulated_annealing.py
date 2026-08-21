"""Simulated annealing (book §5.3): a random-walk global optimizer that — unlike every
method so far in this repo — sometimes deliberately accepts a *worse* point, so it can
walk out of a local minimum that would trap gradient descent, Newton, or Nelder-Mead
permanently. The trick is the acceptance rule and its schedule: early on (high
"temperature"), a worse move is accepted often; as the temperature cools, the walk
gradually turns into pure greedy descent. The name and the physics metaphor are literal —
this mirrors how slowly cooling a metal lets its atoms settle into a low-energy
crystalline structure instead of freezing into a higher-energy disordered one.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import OptimizeResult, Problem, track_iterations


def simulated_annealing(
    problem: Problem,
    *,
    max_iter: int = 2000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.995,
    step_scale: float = 1.0,
    tol: float = 1e-10,
    seed: int = 0,
) -> OptimizeResult:
    """At each step, propose `x + step_scale * temperature * N(0, I)` (a
    temperature-scaled random jump — large and exploratory while hot, small and local
    once cooled) and accept it outright if it improves `f`; otherwise accept it anyway
    with probability `exp(-(f_new - f_current) / temperature)` (the Metropolis
    criterion). `temperature *= cooling_rate` every step. Tracks the *best point found
    so far*, not the current (possibly worse, deliberately-accepted) point — the
    algorithm's whole mechanism relies on wandering somewhere worse sometimes, so
    "current" and "best" are genuinely different quantities here, unlike every
    monotonically-improving method elsewhere in `optimlab.optimizers`.
    """
    rng = np.random.default_rng(seed)
    x = problem.x0.copy()
    f_x = float(problem.f(x))
    best_x, best_f = x.copy(), f_x
    temperature = initial_temp

    x_hist, f_hist, g_hist = [best_x.copy()], [best_f], [temperature]

    converged = False
    for _ in range(max_iter):
        proposal = x + step_scale * temperature * rng.standard_normal(problem.n_dim)
        f_proposal = float(problem.f(proposal))
        delta = f_proposal - f_x

        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
            x, f_x = proposal, f_proposal
            if f_x < best_f:
                best_x, best_f = x.copy(), f_x

        temperature *= cooling_rate
        x_hist.append(best_x.copy())
        f_hist.append(best_f)
        g_hist.append(temperature)

        if temperature < tol:
            converged = True
            break

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="simulated_annealing",
        message="temperature below tol" if converged else "max_iter reached",
    )
