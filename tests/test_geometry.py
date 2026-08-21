import numpy as np
import pytest

from optimlab.highdim.geometry import ball_shell_volume_fraction, pairwise_cosine_similarities


def test_cosine_similarities_concentrate_toward_zero_as_dimension_grows():
    low_dim = pairwise_cosine_similarities(dim=3, n_vectors=300, seed=0)
    high_dim = pairwise_cosine_similarities(dim=500, n_vectors=300, seed=0)
    assert np.std(high_dim) < np.std(low_dim)
    assert np.mean(np.abs(high_dim)) < 0.1


def test_cosine_similarities_are_bounded():
    sims = pairwise_cosine_similarities(dim=10, n_vectors=100, seed=0)
    assert np.all(sims >= -1.0 - 1e-8)
    assert np.all(sims <= 1.0 + 1e-8)


def test_ball_shell_fraction_matches_the_closed_form():
    assert ball_shell_volume_fraction(dim=1, shell_thickness=0.1) == pytest.approx(0.1)
    assert ball_shell_volume_fraction(dim=2, shell_thickness=0.1) == pytest.approx(1 - 0.9**2)


def test_ball_shell_fraction_approaches_one_as_dimension_grows():
    assert ball_shell_volume_fraction(dim=1000, shell_thickness=0.1) > 0.999


def test_ball_shell_fraction_rejects_invalid_thickness():
    with pytest.raises(ValueError, match="shell_thickness"):
        ball_shell_volume_fraction(dim=5, shell_thickness=1.5)
