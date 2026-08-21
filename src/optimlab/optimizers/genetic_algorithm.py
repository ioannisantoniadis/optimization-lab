"""A from-scratch genetic algorithm (book §5.4): maintain an entire *population* of
candidate points rather than one, and evolve it — keep the fittest, recombine pairs
(crossover), and occasionally perturb (mutate) — generation after generation. Simulated
annealing explores by letting one point wander with a cooling tolerance for bad moves;
a genetic algorithm explores by keeping many simultaneous guesses alive and letting
competition between them do the work, which trades per-evaluation efficiency for
robustness against getting the whole search stuck in one basin.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike, OptimizeResult, Problem, track_iterations


def genetic_algorithm(
    problem: Problem,
    *,
    bounds: tuple[float, float] | None = None,
    population_size: int = 60,
    max_generations: int = 200,
    elite_fraction: float = 0.1,
    mutation_scale: float = 0.1,
    mutation_rate: float = 0.2,
    tol: float = 1e-10,
    seed: int = 0,
) -> OptimizeResult:
    """Each generation: rank the population by `f`, keep the top `elite_fraction`
    unchanged, then fill the rest by picking two parents (tournament selection — draw 3
    at random, keep the fittest) and **blend crossover** (the child is a random point on
    the line segment between its parents, `child = t*parent1 + (1-t)*parent2`, not a
    literal splice — blending is the natural choice for continuous variables), then
    mutating each gene independently with probability `mutation_rate` by a Gaussian jump
    scaled by `mutation_scale * (upper - lower)`.

    `bounds` defaults to `problem.domain` (set on every `optimlab.landscapes` benchmark)
    since a population needs a region to be initialized *over*, not just a starting
    point — pass it explicitly for a `Problem` with no `domain` set.
    """
    if bounds is None:
        if problem.domain is None:
            raise ValueError("genetic_algorithm needs `bounds` (problem.domain is unset)")
        bounds = problem.domain
    lower, upper = bounds
    n = problem.n_dim
    rng = np.random.default_rng(seed)

    population = rng.uniform(lower, upper, size=(population_size, n))
    n_elite = max(1, int(elite_fraction * population_size))

    def evaluate(pop: ArrayLike) -> ArrayLike:
        return np.array([float(problem.f(ind)) for ind in pop])

    fitness = evaluate(population)
    order = np.argsort(fitness)
    population, fitness = population[order], fitness[order]

    x_hist, f_hist, g_hist = [population[0].copy()], [float(fitness[0])], [float(fitness.std())]
    converged = g_hist[0] < tol

    def tournament_pick() -> ArrayLike:
        contenders = rng.integers(0, population_size, size=3)
        return population[contenders[np.argmin(fitness[contenders])]]

    generation = 0
    while not converged and generation < max_generations:
        next_population = list(population[:n_elite])
        while len(next_population) < population_size:
            parent1, parent2 = tournament_pick(), tournament_pick()
            t = rng.random()
            child = t * parent1 + (1 - t) * parent2
            mutate_mask = rng.random(n) < mutation_rate
            child = np.where(
                mutate_mask, child + rng.standard_normal(n) * mutation_scale * (upper - lower), child
            )
            next_population.append(np.clip(child, lower, upper))

        population = np.array(next_population)
        fitness = evaluate(population)
        order = np.argsort(fitness)
        population, fitness = population[order], fitness[order]

        generation += 1
        x_hist.append(population[0].copy())
        f_hist.append(float(fitness[0]))
        g_hist.append(float(fitness.std()))
        if g_hist[-1] < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="genetic_algorithm",
        message="population converged (fitness std below tol)" if converged else "max_generations reached",
    )
