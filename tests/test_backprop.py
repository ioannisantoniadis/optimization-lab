import numpy as np

from optimlab.highdim.nets import MLPShape, init_params, mlp_training_problem
from optimlab.ml.backprop import manual_mlp_gradient


def test_manual_gradient_matches_autodiff_exactly():
    """The actual correctness check: two independent derivations of the same
    gradient -- hand-implemented backprop and JAX's automatic differentiation --
    landing at essentially machine precision of each other.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(30, 3))
    y = rng.uniform(-1, 1, size=(30, 2))

    shape = MLPShape(layer_sizes=[3, 8, 5, 2])
    params = init_params(shape, seed=0)

    problem = mlp_training_problem(shape, X, y, seed=0)
    autodiff_grad = problem.grad(params)
    manual_grad = manual_mlp_gradient(params, shape, X, y)

    np.testing.assert_allclose(manual_grad, autodiff_grad, atol=1e-10)


def test_manual_gradient_matches_autodiff_for_a_single_hidden_layer():
    """Same check, a different (shallower) architecture -- not a coincidence specific
    to one layer count.
    """
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(20, 2))
    y = rng.uniform(-1, 1, size=(20, 1))

    shape = MLPShape(layer_sizes=[2, 10, 1])
    params = init_params(shape, seed=1)

    problem = mlp_training_problem(shape, X, y, seed=1)
    np.testing.assert_allclose(
        manual_mlp_gradient(params, shape, X, y), problem.grad(params), atol=1e-10
    )


def test_manual_gradient_drives_loss_down_via_plain_gradient_descent():
    """Not just numerically matching autodiff -- actually usable to train: a bare
    gradient-descent loop using only manual_mlp_gradient (no optimlab.optimizers, no
    autodiff) should still reduce the loss substantially.
    """
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(50, 1))
    y = np.sin(X)

    shape = MLPShape(layer_sizes=[1, 12, 1])
    problem = mlp_training_problem(shape, X, y, seed=2)
    params = problem.x0.copy()
    initial_loss = problem.f(params)

    for _ in range(2000):
        grad = manual_mlp_gradient(params, shape, X, y)
        params = params - 0.05 * grad

    assert problem.f(params) < 0.1 * initial_loss
