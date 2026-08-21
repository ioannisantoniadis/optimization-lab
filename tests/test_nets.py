import numpy as np
import pytest

from optimlab.highdim.nets import MLPShape, forward, init_params, mlp_training_problem, unflatten
from optimlab.optimizers import adam, bfgs


def test_n_params_matches_a_hand_counted_architecture():
    # [2, 3, 1]: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
    shape = MLPShape(layer_sizes=[2, 3, 1])
    assert shape.n_params == 13


def test_unflatten_round_trips_through_forward_with_correct_shapes():
    shape = MLPShape(layer_sizes=[3, 5, 2])
    flat = init_params(shape, seed=0)
    layers = unflatten(flat, shape)
    assert len(layers) == 2
    assert layers[0][0].shape == (3, 5) and layers[0][1].shape == (5,)
    assert layers[1][0].shape == (5, 2) and layers[1][1].shape == (2,)

    X = np.random.default_rng(0).standard_normal((10, 3))
    preds = forward(flat, shape, X)
    assert preds.shape == (10, 2)


def test_forward_is_linear_output_no_final_activation():
    """A single-layer 'MLP' (no hidden layer) is exactly linear regression -- checks
    that forward doesn't apply tanh to the output layer, only hidden ones.
    """
    shape = MLPShape(layer_sizes=[2, 1])
    flat = init_params(shape, seed=0)
    W, b = unflatten(flat, shape)[0]
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    expected = np.asarray(X) @ np.asarray(W) + np.asarray(b)
    np.testing.assert_allclose(np.asarray(forward(flat, shape, X)), expected, atol=1e-10)


@pytest.mark.parametrize("solver", [adam, bfgs])
def test_existing_solvers_train_the_network_below_the_noise_floor(solver):
    """The whole point of wrapping an MLP as an ordinary Problem: no new optimizer is
    needed to train it. Fits noisy sin(x) data and checks the loss drops close to the
    injected noise variance (0.05^2 = 0.0025) -- i.e. it actually learned the function,
    not just decreased loss a little.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = (np.sin(X) + 0.05 * rng.standard_normal(X.shape)).reshape(-1, 1)

    shape = MLPShape(layer_sizes=[1, 16, 16, 1])
    problem = mlp_training_problem(shape, X, y, seed=0)
    initial_loss = problem.f(problem.x0)

    kwargs = {"lr": 0.01, "max_iter": 3000} if solver is adam else {"max_iter": 500}
    result = solver(problem, **kwargs)

    assert result.f < initial_loss
    assert result.f < 0.01
