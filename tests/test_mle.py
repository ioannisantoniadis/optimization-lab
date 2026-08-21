import jax.numpy as jnp
import numpy as np
import pytest

from optimlab.inference.mle import map_fit, mle_fit
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
