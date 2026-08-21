import numpy as np
import pytest

from optimlab.core import Problem
from optimlab.landscapes import get
from optimlab.optimizers.genetic_algorithm import genetic_algorithm


@pytest.mark.parametrize("name", ["sphere", "rastrigin", "ackley", "himmelblau"])
def test_finds_the_global_optimum_on_multimodal_benchmarks(name):
    """Rastrigin and Ackley are exactly the landscapes a purely local method (gradient
    descent, Nelder-Mead from a bad start) gets stuck on -- a population-based search
    should reliably reach the global optimum's neighborhood regardless.
    """
    bf = get(name)
    problem = bf.problem(n_dim=2, seed=0)
    result = genetic_algorithm(problem, seed=0)
    assert result.f < 1e-2


def test_requires_bounds_when_problem_has_no_domain():
    problem = Problem(f=lambda x: float(x @ x), x0=np.zeros(2))
    with pytest.raises(ValueError, match="bounds"):
        genetic_algorithm(problem)


def test_explicit_bounds_override_problem_domain():
    bf = get("sphere")
    problem = bf.problem(n_dim=2, seed=0)
    result = genetic_algorithm(problem, bounds=(-1.0, 1.0), seed=0)
    assert np.all(np.abs(result.x) <= 1.0 + 1e-8)


def test_population_never_leaves_the_bounds():
    bf = get("rastrigin")
    problem = bf.problem(n_dim=2, seed=0)
    result = genetic_algorithm(problem, bounds=(-2.0, 2.0), seed=1, max_generations=30)
    for x in result.trajectory:
        assert np.all(x >= -2.0 - 1e-8) and np.all(x <= 2.0 + 1e-8)


def test_best_fitness_never_gets_worse_across_generations():
    bf = get("ackley")
    problem = bf.problem(n_dim=2, seed=0)
    result = genetic_algorithm(problem, seed=3)
    f_traj = np.asarray(result.f_trajectory)
    assert np.all(np.diff(f_traj) <= 1e-12)  # elitism guarantees this


def test_is_reproducible_given_the_same_seed():
    bf = get("rastrigin")
    result_a = genetic_algorithm(bf.problem(n_dim=2, seed=0), seed=42, max_generations=20)
    result_b = genetic_algorithm(bf.problem(n_dim=2, seed=0), seed=42, max_generations=20)
    np.testing.assert_allclose(result_a.x, result_b.x)
