import numpy as np

from optimlab.highdim.loss_landscape import linear_interpolation_loss
from optimlab.highdim.mode_connectivity import (
    bezier_curve_problem,
    bezier_point,
    evaluate_curve_loss,
)
from optimlab.highdim.nets import MLPShape, mlp_training_problem
from optimlab.optimizers import adam, bfgs


def _two_trained_minima():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = (np.sin(X) + 0.05 * rng.standard_normal(X.shape)).reshape(-1, 1)
    shape = MLPShape(layer_sizes=[1, 16, 16, 1])

    problem_a = mlp_training_problem(shape, X, y, seed=0)
    theta_a = adam(problem_a, lr=0.01, max_iter=2000).x
    problem_b = mlp_training_problem(shape, X, y, seed=1)
    theta_b = adam(problem_b, lr=0.01, max_iter=2000).x
    return problem_a, theta_a, theta_b


def test_bezier_point_reduces_to_the_endpoints_at_t_zero_and_one():
    theta_a = np.array([1.0, 2.0])
    theta_c = np.array([10.0, 10.0])
    theta_b = np.array([3.0, 4.0])
    np.testing.assert_allclose(bezier_point(0.0, theta_a, theta_c, theta_b), theta_a)
    np.testing.assert_allclose(bezier_point(1.0, theta_a, theta_c, theta_b), theta_b)


def test_bezier_curve_with_control_point_at_the_midpoint_is_the_straight_line():
    theta_a = np.array([0.0, 0.0])
    theta_b = np.array([4.0, 8.0])
    midpoint = 0.5 * (theta_a + theta_b)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        expected = (1 - t) * theta_a + t * theta_b
        np.testing.assert_allclose(bezier_point(t, theta_a, midpoint, theta_b), expected, atol=1e-10)


def test_optimized_curve_has_much_lower_max_loss_than_the_straight_line():
    """The actual mode-connectivity claim: a straight line between two independently
    trained minima crosses a real loss barrier, but a curve chosen to minimize the
    average loss along it stays close to both endpoints' own loss the whole way.
    """
    problem, theta_a, theta_b = _two_trained_minima()

    _alphas, straight_losses = linear_interpolation_loss(problem.f, theta_a, theta_b, n_points=21)

    curve_problem = bezier_curve_problem(problem.f, theta_a, theta_b, n_samples=8)
    curve_result = bfgs(curve_problem, max_iter=300)
    _ts, curve_losses = evaluate_curve_loss(problem.f, theta_a, curve_result.x, theta_b, n_points=21)

    assert straight_losses.max() > 10 * max(straight_losses[0], straight_losses[-1])
    assert curve_losses.max() < 2 * max(curve_losses[0], curve_losses[-1])
    assert curve_losses.max() < straight_losses.max()
