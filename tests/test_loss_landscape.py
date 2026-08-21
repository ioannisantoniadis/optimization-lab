import numpy as np

from optimlab.highdim.loss_landscape import (
    filter_normalized_direction,
    linear_interpolation_loss,
    loss_landscape_slice,
)
from optimlab.highdim.nets import MLPShape, init_params, mlp_training_problem, unflatten
from optimlab.optimizers import adam


def _trained_network():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = (np.sin(X) + 0.05 * rng.standard_normal(X.shape)).reshape(-1, 1)
    shape = MLPShape(layer_sizes=[1, 16, 16, 1])
    problem = mlp_training_problem(shape, X, y, seed=0)
    result = adam(problem, lr=0.01, max_iter=1500)
    return problem, shape, result.x


def test_filter_normalized_direction_matches_each_layers_own_norm():
    shape = MLPShape(layer_sizes=[1, 16, 16, 1])
    base = init_params(shape, seed=0)
    direction = filter_normalized_direction(base, shape, seed=1)

    for (W, _b), (dW, _db) in zip(unflatten(base, shape), unflatten(direction, shape), strict=True):
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(dW)), np.linalg.norm(np.asarray(W)), rtol=1e-5
        )


def test_loss_slice_center_matches_the_base_points_own_loss():
    problem, shape, theta_star = _trained_network()
    resolution = 11
    sl = loss_landscape_slice(problem.f, theta_star, shape, span=1.0, resolution=resolution, seed=0)
    center = resolution // 2
    assert sl.Z[center, center] == problem.f(theta_star)


def test_linear_interpolation_between_a_point_and_itself_is_flat():
    problem, _shape, theta_star = _trained_network()
    _alphas, losses = linear_interpolation_loss(problem.f, theta_star, theta_star, n_points=10)
    np.testing.assert_allclose(losses, losses[0], atol=1e-10)


def test_linear_interpolation_endpoints_match_the_two_input_losses():
    problem, shape, theta_star = _trained_network()
    theta_other = np.random.default_rng(5).standard_normal(shape.n_params) * 0.5
    alphas, losses = linear_interpolation_loss(problem.f, theta_star, theta_other, n_points=20)
    assert alphas[0] == 0.0 and alphas[-1] == 1.0
    assert losses[0] == problem.f(theta_star)
    assert losses[-1] == problem.f(theta_other)
