import numpy as np
import pytest

from optimlab.core import Problem
from optimlab.optimizers.bayesian_optimization import (
    bayesian_optimize,
    expected_improvement,
    gp_posterior,
)


def test_gp_posterior_interpolates_training_points_almost_exactly():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0.0, 1.0, 4.0, 9.0])
    posterior = gp_posterior(X_train, y_train, X_train, length_scale=1.0, variance=10.0, noise=1e-8)

    np.testing.assert_allclose(posterior.mean, y_train, atol=1e-6)
    assert np.all(posterior.std < 1e-3)


def test_gp_posterior_reverts_to_the_prior_far_from_any_data():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    posterior = gp_posterior(X_train, y_train, np.array([[1000.0]]), length_scale=1.0, variance=10.0)
    assert posterior.std[0] == pytest.approx(np.sqrt(10.0), rel=1e-3)


def test_expected_improvement_favors_high_uncertainty_over_a_similar_mean():
    mean = np.array([0.0, 0.0])
    std = np.array([0.01, 5.0])
    ei = expected_improvement(mean, std, best_so_far=0.0)
    assert ei[1] > ei[0]


def test_expected_improvement_is_zero_at_zero_uncertainty():
    ei = expected_improvement(mean=np.array([5.0]), std=np.array([0.0]), best_so_far=0.0)
    assert ei[0] == 0.0


def test_bayesian_optimize_finds_a_simple_quadratics_minimum():
    problem = Problem(f=lambda x: (x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2, x0=np.zeros(2), name="quad")
    result = bayesian_optimize(problem, bounds=(-3.0, 3.0), n_init=5, n_iter=25, length_scale=1.0, seed=0)

    assert result.f < 0.05
    np.testing.assert_allclose(result.x, [1.5, -0.5], atol=0.2)


def test_bayesian_optimize_f_trajectory_is_the_running_best_and_non_increasing():
    problem = Problem(f=lambda x: (x[0] - 1.5) ** 2, x0=np.zeros(1), name="quad_1d")
    result = bayesian_optimize(problem, bounds=(-3.0, 3.0), n_init=3, n_iter=10, seed=0)

    f_traj = np.asarray(result.f_trajectory)
    assert np.all(np.diff(f_traj) <= 0)
    assert f_traj[-1] == result.f
