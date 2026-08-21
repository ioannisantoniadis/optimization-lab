import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.linalg import least_squares
from optimlab.viz import (
    lasso_path_figure,
    regression_fit_figure,
    residuals_figure,
    ridge_path_figure,
    svd_conditioning_figure,
)


def test_regression_fit_figure_needs_single_feature_A():
    A = np.ones((5, 2))
    with pytest.raises(ValueError, match="single-feature"):
        regression_fit_figure(A, np.ones(5), np.ones(2))


def test_regression_fit_figure_draws_data_and_fit_line():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 20).reshape(-1, 1)
    A = np.hstack([t, np.ones_like(t)])
    y = 2.0 * t[:, 0] + 1.0 + rng.normal(scale=0.1, size=20)
    result = least_squares(A, y)

    fig = regression_fit_figure(t, y, result.x)
    assert isinstance(fig, go.Figure)
    names = [trace.name for trace in fig.data]
    assert "data" in names and "least squares fit" in names


def test_residuals_figure_centers_on_zero_for_a_good_fit():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 3))
    x_true = np.array([1.0, -2.0, 0.5])
    b = A @ x_true
    result = least_squares(A, b)
    fig = residuals_figure(A, b, result.x)
    residuals = fig.data[0].y
    np.testing.assert_allclose(residuals, 0.0, atol=1e-8)


def test_svd_conditioning_figure_needs_2x2():
    with pytest.raises(ValueError, match="2x2"):
        svd_conditioning_figure(np.eye(3))


def test_svd_conditioning_figure_ellipse_semi_axes_match_singular_values():
    A = np.array([[3.0, 1.0], [0.0, 0.5]])
    fig = svd_conditioning_figure(A)
    ellipse = next(t for t in fig.data if t.name == "A · (unit circle)")
    radii = np.hypot(ellipse.x, ellipse.y)
    s = np.linalg.svd(A, compute_uv=False)
    # loose tolerances: the ellipse is a 200-point sampled curve, not the analytic
    # ellipse, so its discrete max/min radius only approximates the true semi-axes
    np.testing.assert_allclose(radii.max(), s.max(), atol=1e-4)
    np.testing.assert_allclose(radii.min(), s.min(), atol=1e-2)


def test_svd_conditioning_figure_title_reports_condition_number():
    A = np.diag([4.0, 2.0])
    fig = svd_conditioning_figure(A)
    assert "2" in fig.layout.title.text  # condition number = 4/2 = 2


def test_ridge_path_figure_has_one_line_per_feature():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 4))
    b = rng.standard_normal(30)
    alphas = np.logspace(-2, 2, 10)
    fig = ridge_path_figure(A, b, alphas)
    assert len(fig.data) == 4
    for trace in fig.data:
        np.testing.assert_allclose(trace.x, alphas)


def test_ridge_path_figure_coefficients_shrink_to_zero_at_large_alpha():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 3))
    b = rng.standard_normal(30)
    alphas = np.array([1e-3, 1e8])
    fig = ridge_path_figure(A, b, alphas)
    for trace in fig.data:
        assert abs(trace.y[-1]) < abs(trace.y[0])
        assert abs(trace.y[-1]) < 1e-4


def test_lasso_path_figure_has_one_line_per_feature():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((30, 4))
    b = rng.standard_normal(30)
    alphas = np.logspace(-2, 2, 8)
    fig = lasso_path_figure(A, b, alphas)
    assert len(fig.data) == 4
    for trace in fig.data:
        np.testing.assert_allclose(trace.x, alphas)


def test_lasso_path_figure_hits_exact_zero_unlike_ridge():
    """The qualitative point of putting LASSO next to ridge: LASSO coefficients reach
    *exactly* zero at large-enough alpha and stay there, where ridge's only approach it
    in the limit.
    """
    rng = np.random.default_rng(4)
    A = rng.standard_normal((30, 3))
    x_true = np.array([2.0, -1.5, 0.0])
    b = A @ x_true + 0.01 * rng.standard_normal(30)
    alphas = np.logspace(-2, 3, 30)

    fig = lasso_path_figure(A, b, alphas)
    # the truly-irrelevant feature (x2) should be exactly zero for large alpha
    assert fig.data[2].y[-1] == pytest.approx(0.0, abs=1e-8)

    ridge_fig = ridge_path_figure(A, b, alphas)
    # ridge shrinks the same feature but never hits exactly zero at a finite alpha
    assert abs(ridge_fig.data[2].y[-2]) > 1e-8
