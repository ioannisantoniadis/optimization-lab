import numpy as np

from optimlab.highdim.nets import MLPShape, init_params
from optimlab.highdim.ntk import (
    ntk_concentration_experiment,
    ntk_matrix,
    relative_frobenius_difference,
)


def test_ntk_matrix_is_symmetric_and_positive_semidefinite():
    shape = MLPShape(layer_sizes=[2, 16, 1])
    params = init_params(shape, seed=0)
    X = np.random.default_rng(0).uniform(-1, 1, size=(10, 2))

    K = ntk_matrix(params, shape, X)
    np.testing.assert_allclose(K, K.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() > -1e-6  # K = J J^T is PSD by construction


def test_relative_frobenius_difference_is_zero_for_identical_matrices():
    A = np.random.default_rng(0).standard_normal((5, 5))
    assert relative_frobenius_difference(A, A) == 0.0


def test_ntk_concentrates_as_width_grows():
    """The core NTK claim: independent random initializations' empirical tangent
    kernels should become more similar to each other as the network gets wider.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(20, 2))
    result = ntk_concentration_experiment(widths=[4, 64, 1024], X=X, n_in=2, n_seed_pairs=5, seed=0)

    assert result.mean_relative_diff[0] > result.mean_relative_diff[1] > result.mean_relative_diff[2]
    assert result.mean_relative_diff[-1] < 0.15
