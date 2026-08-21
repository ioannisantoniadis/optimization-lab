import jax.numpy as jnp
import numpy as np
import pytest

from optimlab.inference.laplace import laplace_approximation
from optimlab.inference.mle import map_fit


def test_laplace_is_exact_for_a_conjugate_gaussian_gaussian_posterior():
    """The one case where the Laplace approximation isn't an approximation at all: if
    the true posterior is already Gaussian, a second-order Taylor expansion of its log
    at the mode reproduces it exactly, mean and variance both.
    """
    rng = np.random.default_rng(2)
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

    map_result = map_fit(log_likelihood, log_prior, x0=np.array([0.0]))
    approx = laplace_approximation(log_likelihood, log_prior, map_result.x)

    assert approx.mean[0] == pytest.approx(post_mean, abs=1e-4)
    assert approx.cov[0, 0] == pytest.approx(post_var, abs=1e-4)


def test_laplace_mean_diverges_from_true_posterior_mean_under_skew():
    """Beta(8,2) is left-skewed: mode (0.875) != mean (0.8). Laplace centers on the
    mode by construction, so its `mean` field -- despite the name -- lands on 0.875, not
    the true posterior mean 0.8. This is the concrete failure `optimlab.inference.mcmc`
    exists to avoid.
    """
    from optimlab.optimizers.projected_gradient import projected_gradient

    n_trials, n_success = 8, 7

    def log_likelihood(params):
        theta = params[0]
        return n_success * jnp.log(theta) + (n_trials - n_success) * jnp.log(1 - theta)

    def log_prior(params):
        return 0.0 * params[0]  # uniform (Beta(1,1)) prior

    map_result = map_fit(
        log_likelihood, log_prior, x0=np.array([0.5]),
        solver=projected_gradient, lower=0.02, upper=0.98, lr=0.002, max_iter=5000,
    )
    approx = laplace_approximation(log_likelihood, log_prior, map_result.x)

    true_posterior_mean = 8.0 / (8.0 + 2.0)  # Beta(8,2) mean = alpha/(alpha+beta)
    assert abs(approx.mean[0] - true_posterior_mean) > 0.05
