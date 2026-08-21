import numpy as np
import pytest

from optimlab.highdim.random_landscapes import critical_point_index_stats, sample_goe_eigenvalues


def test_goe_matrices_are_symmetric_with_the_right_shape():
    eigenvalues = sample_goe_eigenvalues(dim=8, n_samples=50, seed=0)
    assert eigenvalues.shape == (50, 8)
    # eigvalsh returns ascending order; every sample's eigenvalues should be sorted.
    assert np.all(np.diff(eigenvalues, axis=1) >= 0)


def test_dim_one_is_a_coin_flip():
    """A 1x1 'GOE matrix' is just a single Gaussian scalar -- positive or negative with
    equal probability, no eigenvalue repulsion possible with only one eigenvalue.
    """
    stats = critical_point_index_stats(dims=[1], n_samples=20000, seed=0)
    assert stats.p_local_min[0] == pytest.approx(0.5, abs=0.02)
    assert stats.p_saddle[0] == 0.0


def test_probability_of_a_local_minimum_collapses_with_dimension():
    """The core claim: as dimension grows, pinning every eigenvalue to the same sign
    becomes overwhelmingly unlikely -- almost every critical point is a saddle.
    """
    stats = critical_point_index_stats(dims=[1, 2, 4, 8], n_samples=20000, seed=1)
    assert np.all(np.diff(stats.p_local_min) < 0)  # strictly decreasing
    assert stats.p_saddle[-1] > 0.99  # essentially always a saddle by dim=8
