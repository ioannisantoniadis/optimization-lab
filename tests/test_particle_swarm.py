import numpy as np
import pytest

from optimlab.core import Problem
from optimlab.landscapes import get
from optimlab.optimizers.particle_swarm import particle_swarm


@pytest.mark.parametrize("name", ["sphere", "rastrigin", "ackley", "himmelblau"])
def test_finds_the_global_optimum_on_multimodal_benchmarks(name):
    bf = get(name)
    problem = bf.problem(n_dim=2, seed=0)
    result = particle_swarm(problem, seed=0)
    assert result.f < 1e-2


def test_requires_bounds_when_problem_has_no_domain():
    problem = Problem(f=lambda x: float(x @ x), x0=np.zeros(2))
    with pytest.raises(ValueError, match="bounds"):
        particle_swarm(problem)


def test_particles_stay_clamped_within_bounds():
    bf = get("rastrigin")
    problem = bf.problem(n_dim=2, seed=0)
    result = particle_swarm(problem, bounds=(-2.0, 2.0), seed=1, max_iter=50)
    for x in result.trajectory:
        assert np.all(x >= -2.0 - 1e-8) and np.all(x <= 2.0 + 1e-8)


def test_global_best_never_gets_worse_across_iterations():
    bf = get("ackley")
    problem = bf.problem(n_dim=2, seed=0)
    result = particle_swarm(problem, seed=3)
    f_traj = np.asarray(result.f_trajectory)
    assert np.all(np.diff(f_traj) <= 1e-12)


def test_is_reproducible_given_the_same_seed():
    bf = get("rastrigin")
    result_a = particle_swarm(bf.problem(n_dim=2, seed=0), seed=42, max_iter=50)
    result_b = particle_swarm(bf.problem(n_dim=2, seed=0), seed=42, max_iter=50)
    np.testing.assert_allclose(result_a.x, result_b.x)
