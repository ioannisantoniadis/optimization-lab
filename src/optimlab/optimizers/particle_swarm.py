"""Particle swarm optimization (book §5.5): another population-based, gradient-free
search, but where `optimlab.optimizers.genetic_algorithm` explores through selection and
recombination, PSO gives every candidate ("particle") its own *velocity* and nudges that
velocity toward two pulls — the best point *that particle* has personally seen, and the
best point *any* particle in the swarm has seen. No crossover, no mutation, no explicit
fitness-based selection at all: exploration and exploitation both fall out of ordinary
momentum-like dynamics, which is what makes PSO closer in spirit to
`optimlab.optimizers.momentum` (nudge a velocity, don't jump) than to
`genetic_algorithm`, despite both being population methods aimed at the same class of
problem.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike, OptimizeResult, Problem, track_iterations


def particle_swarm(
    problem: Problem,
    *,
    bounds: tuple[float, float] | None = None,
    n_particles: int = 40,
    max_iter: int = 200,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    tol: float = 1e-10,
    seed: int = 0,
) -> OptimizeResult:
    """Velocity update for particle `i`:
    `v_i <- inertia*v_i + cognitive*r1*(personal_best_i - x_i) + social*r2*(global_best - x_i)`,
    with independent `r1, r2 ~ Uniform(0,1)` per coordinate — `inertia` controls how much
    of the existing heading survives (too high and the swarm never settles; too low and
    it collapses onto the first decent point it finds), while `cognitive`/`social`
    balance "trust my own best find" against "trust the swarm's best find." Particles
    are clamped back into `bounds` (defaulting to `problem.domain`, as in
    `genetic_algorithm`) after each step, velocity zeroed on the clamped axis so a
    particle doesn't immediately try to fly back out.
    """
    if bounds is None:
        if problem.domain is None:
            raise ValueError("particle_swarm needs `bounds` (problem.domain is unset)")
        bounds = problem.domain
    lower, upper = bounds
    n = problem.n_dim
    rng = np.random.default_rng(seed)

    positions = rng.uniform(lower, upper, size=(n_particles, n))
    span = upper - lower
    velocities = rng.uniform(-span, span, size=(n_particles, n)) * 0.1

    def evaluate(pop: ArrayLike) -> ArrayLike:
        return np.array([float(problem.f(ind)) for ind in pop])

    fitness = evaluate(positions)
    personal_best_pos = positions.copy()
    personal_best_fit = fitness.copy()
    global_best_idx = int(np.argmin(fitness))
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = float(personal_best_fit[global_best_idx])

    x_hist, f_hist, g_hist = [global_best_pos.copy()], [global_best_fit], [float(fitness.std())]
    converged = g_hist[0] < tol

    n_iter = 0
    while not converged and n_iter < max_iter:
        r1 = rng.random((n_particles, n))
        r2 = rng.random((n_particles, n))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_pos - positions)
            + social * r2 * (global_best_pos - positions)
        )
        positions = positions + velocities

        out_of_bounds = (positions < lower) | (positions > upper)
        positions = np.clip(positions, lower, upper)
        velocities = np.where(out_of_bounds, 0.0, velocities)

        fitness = evaluate(positions)
        improved = fitness < personal_best_fit
        personal_best_pos[improved] = positions[improved]
        personal_best_fit[improved] = fitness[improved]

        global_best_idx = int(np.argmin(personal_best_fit))
        if personal_best_fit[global_best_idx] < global_best_fit:
            global_best_fit = float(personal_best_fit[global_best_idx])
            global_best_pos = personal_best_pos[global_best_idx].copy()

        n_iter += 1
        x_hist.append(global_best_pos.copy())
        f_hist.append(global_best_fit)
        g_hist.append(float(fitness.std()))
        if g_hist[-1] < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="particle_swarm",
        message="swarm converged (fitness std below tol)" if converged else "max_iter reached",
    )
