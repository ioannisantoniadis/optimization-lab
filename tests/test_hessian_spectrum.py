import numpy as np

from optimlab.core import _autohess
from optimlab.highdim.hessian_spectrum import hessian_vector_product, lanczos_eigenvalues
from optimlab.highdim.nets import MLPShape, mlp_training_problem
from optimlab.optimizers import adam


def _trained_small_network():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(100, 1))
    y = (np.sin(X) + 0.05 * rng.standard_normal(X.shape)).reshape(-1, 1)
    shape = MLPShape(layer_sizes=[1, 8, 1])
    problem = mlp_training_problem(shape, X, y, seed=0)
    result = adam(problem, lr=0.01, max_iter=2000)
    return problem, result.x


def test_hessian_vector_product_matches_the_dense_hessian():
    problem, x_star = _trained_small_network()
    H_dense = _autohess(problem.f)(x_star)
    v = np.random.default_rng(1).standard_normal(x_star.size)

    hv_dense = H_dense @ v
    hv_fast = hessian_vector_product(problem.f, x_star, v)
    np.testing.assert_allclose(hv_fast, hv_dense, atol=1e-8)


def test_lanczos_recovers_extreme_eigenvalues_of_a_small_dense_hessian():
    """With n_iter comparable to the parameter count, Lanczos's Krylov subspace spans
    (nearly) the whole space, so its Ritz values should match the true dense
    eigendecomposition's extremes almost exactly -- the ground-truth check that the
    from-scratch Lanczos implementation is actually correct, not just plausible-looking.
    """
    problem, x_star = _trained_small_network()
    H_dense = _autohess(problem.f)(x_star)
    true_eigs = np.linalg.eigvalsh(H_dense)

    result = lanczos_eigenvalues(problem.f, x_star, n_iter=60, seed=0)

    assert abs(result.ritz_values.max() - true_eigs.max()) < 1e-6
    assert abs(result.ritz_values.min() - true_eigs.min()) < 1e-6


def test_lanczos_ritz_values_never_exceed_the_true_spectrums_range():
    """Ritz values are eigenvalues of the Hessian restricted to a subspace -- by the
    Rayleigh quotient / min-max theorem they can never fall outside [true_min, true_max],
    regardless of how few Lanczos iterations are run.
    """
    problem, x_star = _trained_small_network()
    H_dense = _autohess(problem.f)(x_star)
    true_eigs = np.linalg.eigvalsh(H_dense)

    result = lanczos_eigenvalues(problem.f, x_star, n_iter=10, seed=2)
    assert result.ritz_values.min() >= true_eigs.min() - 1e-8
    assert result.ritz_values.max() <= true_eigs.max() + 1e-8
