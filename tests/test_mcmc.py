import numpy as np
import pytest

from optimlab.inference.mcmc import metropolis_hastings


def test_mcmc_recovers_closed_form_gaussian_gaussian_posterior():
    rng = np.random.default_rng(3)
    sigma, n = 2.0, 30
    data = rng.normal(3.0, sigma, size=n)
    mu0, tau0 = 0.0, 5.0

    post_var = 1.0 / (n / sigma**2 + 1.0 / tau0**2)
    post_mean = post_var * (np.sum(data) / sigma**2 + mu0 / tau0**2)

    def log_posterior(params):
        mu = params[0]
        log_lik = np.sum(-0.5 * ((data - mu) / sigma) ** 2)
        log_prior = -0.5 * ((mu - mu0) / tau0) ** 2
        return float(log_lik + log_prior)

    result = metropolis_hastings(
        log_posterior, x0=np.array([0.0]), n_samples=20000, proposal_std=0.5, burn_in=2000, seed=0
    )
    assert 0.1 < result.acceptance_rate < 0.9
    assert result.samples.mean() == pytest.approx(post_mean, abs=0.05)
    assert result.samples.std() == pytest.approx(np.sqrt(post_var), abs=0.05)


def test_mcmc_recovers_true_mean_of_a_skewed_beta_posterior_where_laplace_would_not():
    """The true posterior here is Beta(8,2): mean 0.8, mode 0.875. Unlike a Gaussian
    approximation centered at the mode, sampling doesn't need the posterior to be
    symmetric to get its mean right.
    """
    n_trials, n_success = 8, 7

    def log_posterior(params):
        theta = params[0]
        if theta <= 0.0 or theta >= 1.0:
            return -np.inf
        return n_success * np.log(theta) + (n_trials - n_success) * np.log(1 - theta)

    result = metropolis_hastings(
        log_posterior, x0=np.array([0.7]), n_samples=20000, proposal_std=0.1, burn_in=2000, seed=0
    )
    true_posterior_mean = 8.0 / (8.0 + 2.0)
    assert result.samples.mean() == pytest.approx(true_posterior_mean, abs=0.02)


def test_burn_in_samples_are_not_included_in_the_output():
    result = metropolis_hastings(
        lambda params: -0.5 * params[0] ** 2, x0=np.array([10.0]),
        n_samples=100, burn_in=50, proposal_std=1.0, seed=0,
    )
    assert result.samples.shape == (100, 1)
