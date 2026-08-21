import numpy as np
import pytest

from optimlab.inference.em import em_gmm


def _three_blob_data(seed=0):
    rng = np.random.default_rng(seed)
    true_means = np.array([[0.0, 0.0], [8.0, 8.0], [8.0, -8.0]])
    X = np.concatenate([rng.multivariate_normal(m, 0.5 * np.eye(2), size=100) for m in true_means])
    return X, true_means


def test_log_likelihood_is_monotonically_non_decreasing():
    """The one guarantee EM actually carries: each E-step + M-step cycle maximizes a
    surrogate that lower-bounds the true log-likelihood and touches it at the current
    parameters, so the true log-likelihood can never decrease from one iteration to the
    next (Jensen's inequality) -- unlike gradient descent, this holds with no step-size
    condition to get right.
    """
    X, _ = _three_blob_data()
    result = em_gmm(X, n_components=3, seed=0)
    ll = np.asarray(result.log_likelihood_trajectory)
    assert np.all(np.diff(ll) >= -1e-8)


def test_recovers_well_separated_cluster_means():
    X, true_means = _three_blob_data()
    result = em_gmm(X, n_components=3, seed=0)
    assert result.converged
    for true_mean in true_means:
        nearest_dist = np.min(np.linalg.norm(result.means - true_mean, axis=1))
        assert nearest_dist < 0.5


def test_weights_sum_to_one_and_responsibilities_are_a_proper_distribution():
    X, _ = _three_blob_data()
    result = em_gmm(X, n_components=3, seed=0)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), 1.0, atol=1e-8)
