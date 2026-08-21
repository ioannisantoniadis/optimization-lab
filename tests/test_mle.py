import jax.numpy as jnp
import numpy as np
import pytest

from optimlab.inference.mle import map_fit, mle_fit
from optimlab.linalg import least_squares, ridge_regression
from optimlab.optimizers.projected_gradient import projected_gradient


def test_mle_recovers_sample_mean_for_gaussian_data():
    """MLE for a Gaussian's mean, known variance, is exactly the sample mean -- one of
    the few MLEs with a clean closed form to check against.
    """
    rng = np.random.default_rng(0)
    sigma = 2.0
    data = rng.normal(3.0, sigma, size=200)

    def log_likelihood(params):
        mu = params[0]
        return jnp.sum(-0.5 * ((data - mu) / sigma) ** 2)

    result = mle_fit(log_likelihood, x0=np.array([0.0]))
    assert result.converged
    assert result.x[0] == pytest.approx(data.mean(), abs=1e-4)


def test_map_matches_closed_form_gaussian_gaussian_posterior_mean():
    """Gaussian likelihood + Gaussian prior is the one conjugate case with a fully
    closed-form posterior mean/variance -- the MAP estimate (the posterior mode) must
    equal that closed-form mean exactly, since a Gaussian's mode and mean coincide.
    """
    rng = np.random.default_rng(1)
    sigma, n = 2.0, 30
    data = rng.normal(3.0, sigma, size=n)
    mu0, tau0 = 0.0, 5.0

    def log_likelihood(params):
        mu = params[0]
        return jnp.sum(-0.5 * ((data - mu) / sigma) ** 2)

    def log_prior(params):
        mu = params[0]
        return -0.5 * ((mu - mu0) / tau0) ** 2

    post_var = 1.0 / (n / sigma**2 + 1.0 / tau0**2)
    post_mean = post_var * (np.sum(data) / sigma**2 + mu0 / tau0**2)

    result = map_fit(log_likelihood, log_prior, x0=np.array([0.0]))
    assert result.converged
    assert result.x[0] == pytest.approx(post_mean, abs=1e-4)


def test_map_matches_closed_form_beta_binomial_mode_with_a_bounded_solver():
    """theta in (0,1) has genuinely unbounded curvature near the boundary -- plain BFGS
    overshoots wildly there, so this needs the bounded `projected_gradient` (Chapter 3)
    passed in as `solver`. The closed-form Beta(alpha,beta) posterior's mode is
    (alpha-1)/(alpha+beta-2).
    """
    n_trials, n_success = 8, 7
    alpha0, beta0 = 1.0, 1.0

    def log_likelihood(params):
        theta = params[0]
        return n_success * jnp.log(theta) + (n_trials - n_success) * jnp.log(1 - theta)

    def log_prior(params):
        theta = params[0]
        return (alpha0 - 1) * jnp.log(theta) + (beta0 - 1) * jnp.log(1 - theta)

    post_alpha, post_beta = alpha0 + n_success, beta0 + n_trials - n_success
    true_mode = (post_alpha - 1) / (post_alpha + post_beta - 2)

    result = map_fit(
        log_likelihood, log_prior, x0=np.array([0.5]),
        solver=projected_gradient, lower=0.02, upper=0.98, lr=0.002, max_iter=5000,
    )
    assert result.converged
    assert result.x[0] == pytest.approx(true_mode, abs=1e-3)


def test_ordinary_least_squares_is_exactly_gaussian_mle():
    """optimlab.linalg.least_squares (Chapter 3) minimizes ||Ax-b||^2 via the SVD --
    a purely geometric derivation with no mention of probability anywhere. Framed as
    MLE instead (data ~ N(Ax, sigma^2 I)), the log-likelihood is -0.5/sigma^2 times
    that exact same sum of squared residuals, so the two derivations' optima must
    coincide -- checked here by fitting the identical data both ways.
    """
    rng = np.random.default_rng(4)
    m, n = 60, 3
    A = rng.standard_normal((m, n))
    x_true = np.array([2.0, -1.5, 0.5])
    b = A @ x_true + rng.standard_normal(m)

    ols_result = least_squares(A, b)

    def log_likelihood(params):
        residual = jnp.asarray(A) @ params - jnp.asarray(b)
        return -0.5 * jnp.sum(residual**2)

    mle_result = mle_fit(log_likelihood, x0=np.zeros(n))
    np.testing.assert_allclose(ols_result.x, mle_result.x, atol=1e-6)


def test_ridge_regression_is_exactly_gaussian_map():
    """Ridge's penalty alpha*||x||^2 (Chapter 3) is, up to an additive constant, the
    negative log of a Gaussian prior x ~ N(0, I/alpha) -- so ridge_regression's
    closed-form SVD solution and map_fit's iterative optimum should coincide exactly
    for matching alpha, the same cross-check as ordinary least squares vs. MLE above.
    """
    rng = np.random.default_rng(5)
    m, n = 60, 3
    A = rng.standard_normal((m, n))
    x_true = np.array([2.0, -1.5, 0.5])
    b = A @ x_true + rng.standard_normal(m)
    alpha = 5.0

    ridge_result = ridge_regression(A, b, alpha)

    def log_likelihood(params):
        residual = jnp.asarray(A) @ params - jnp.asarray(b)
        return -0.5 * jnp.sum(residual**2)

    def log_prior(params):
        return -0.5 * alpha * jnp.sum(params**2)

    map_result = map_fit(log_likelihood, log_prior, x0=np.zeros(n))
    np.testing.assert_allclose(ridge_result.x, map_result.x, atol=1e-6)
