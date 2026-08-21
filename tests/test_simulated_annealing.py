import numpy as np

from optimlab.landscapes import get
from optimlab.optimizers import gradient_descent
from optimlab.optimizers.simulated_annealing import simulated_annealing


def test_converges_on_convex_sphere():
    bf = get("sphere")
    problem = bf.problem(n_dim=2, seed=0)
    result = simulated_annealing(problem, max_iter=3000, initial_temp=2.0, cooling_rate=0.995, seed=0)
    assert result.f < 1e-2


def test_escapes_a_local_minimum_that_traps_gradient_descent():
    """Rastrigin, started near a local minimum well away from the global one at the
    origin: gradient descent should get stuck there (it always does -- that's the
    whole point of a local method on a multimodal landscape), while simulated
    annealing's willingness to accept worse moves should let it wander out.
    """
    bf = get("rastrigin")
    x0 = np.array([4.3, -3.7])

    gd_result = gradient_descent(bf.problem(x0=x0.copy()), lr=0.01, max_iter=500)
    assert gd_result.f > 10.0  # stuck near a local minimum, nowhere close to the global one

    sa_result = simulated_annealing(
        bf.problem(x0=x0.copy()), max_iter=5000, initial_temp=5.0, cooling_rate=0.998, seed=0
    )
    assert sa_result.f < gd_result.f
    assert sa_result.f < 2.0  # meaningfully close to the true global minimum (f=0 at the origin)


def test_tracks_best_so_far_not_current_possibly_worse_point():
    """f_trajectory must be monotonically non-increasing even though the underlying
    random walk itself is not -- it's recording the best point found by each step, a
    genuinely different quantity from "the current point" for this algorithm.
    """
    bf = get("rastrigin")
    result = simulated_annealing(bf.problem(x0=np.array([3.0, 3.0])), max_iter=500, seed=2)
    f_traj = np.asarray(result.f_trajectory)
    assert np.all(np.diff(f_traj) <= 1e-12)


def test_temperature_decays_monotonically():
    bf = get("sphere")
    result = simulated_annealing(bf.problem(n_dim=2, seed=0), max_iter=200, cooling_rate=0.99)
    temps = np.asarray(result.grad_norm_trajectory)
    assert np.all(np.diff(temps) < 0)


def test_is_reproducible_given_the_same_seed():
    bf = get("rastrigin")
    problem_a = bf.problem(x0=np.array([2.0, -2.0]))
    problem_b = bf.problem(x0=np.array([2.0, -2.0]))
    result_a = simulated_annealing(problem_a, max_iter=200, seed=7)
    result_b = simulated_annealing(problem_b, max_iter=200, seed=7)
    np.testing.assert_allclose(result_a.x, result_b.x)
    assert result_a.f == result_b.f
