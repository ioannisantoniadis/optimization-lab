# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Bayesian Posterior Explorer")


@app.cell
def _():
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    from scipy import stats

    from optimlab.inference import laplace_approximation, map_fit, metropolis_hastings
    from optimlab.optimizers.projected_gradient import projected_gradient
    from optimlab.viz import posterior_figure

    return (
        jnp,
        laplace_approximation,
        map_fit,
        metropolis_hastings,
        mo,
        np,
        posterior_figure,
        projected_gradient,
        stats,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Posterior Explorer

    A coin flipped `n_trials` times, `n_success` of them heads, a flat `Beta(1,1)`
    prior on the success rate `theta`. Drag the trial counts and watch three views
    of the posterior move together — the true `Beta` posterior (closed form), the
    Laplace approximation (a Gaussian centered at the MAP estimate), and a
    Metropolis-Hastings chain's samples.

    A few things worth trying:

    - Set `n_trials` and `n_success` far apart from 50/50 (e.g. 8 and 7) — the
      posterior skews hard toward 1, and Laplace's necessarily-symmetric Gaussian
      visibly misses the shape MCMC's histogram still tracks.
    - Push `n_trials` up into the hundreds with a similar success rate — the
      posterior narrows and becomes more symmetric, and Laplace's approximation
      should visibly improve.
    - Watch the printed MAP/Laplace-mean/MCMC-mean numbers converge to each other
      as the posterior becomes more Gaussian-shaped.
    """)
    return


@app.cell
def _(mo):
    n_trials_slider = mo.ui.slider(2, 200, value=8, step=1, label="n_trials")
    n_success_slider = mo.ui.slider(0, 200, value=7, step=1, label="n_success")
    mo.vstack([n_trials_slider, n_success_slider])
    return n_success_slider, n_trials_slider


@app.cell
def _(n_success_slider, n_trials_slider):
    n_trials = n_trials_slider.value
    n_success = min(n_success_slider.value, n_trials)  # can't exceed n_trials
    return n_success, n_trials


@app.cell
def _(jnp, laplace_approximation, map_fit, metropolis_hastings, n_success, n_trials, np, projected_gradient):
    def log_likelihood(params):
        theta = params[0]
        return n_success * jnp.log(theta) + (n_trials - n_success) * jnp.log(1 - theta)

    def flat_log_prior(params):
        return 0.0 * params[0]

    map_result = map_fit(
        log_likelihood, flat_log_prior, x0=np.array([0.5]),
        solver=projected_gradient, lower=0.01, upper=0.99, lr=0.002, max_iter=5000,
    )
    laplace = laplace_approximation(log_likelihood, flat_log_prior, map_result.x)

    def log_posterior(params):
        theta = params[0]
        if theta <= 0.0 or theta >= 1.0:
            return -np.inf
        return n_success * np.log(theta) + (n_trials - n_success) * np.log(1 - theta)

    mcmc = metropolis_hastings(
        log_posterior, x0=np.array([map_result.x[0]]), n_samples=8000, proposal_std=0.1, burn_in=2000, seed=0
    )
    return laplace, map_result, mcmc


@app.cell
def _(laplace, map_result, mcmc, mo, n_success, n_trials, posterior_figure, stats):
    post_alpha, post_beta = 1.0 + n_success, 1.0 + (n_trials - n_success)
    true_mean = post_alpha / (post_alpha + post_beta)

    mo.vstack(
        [
            mo.md(
                f"**MAP**: {map_result.x[0]:.4f}  |  **Laplace mean**: {laplace.mean[0]:.4f}  |  "
                f"**MCMC mean**: {mcmc.samples.mean():.4f}  |  **true posterior mean**: {true_mean:.4f}  |  "
                f"**MCMC acceptance**: {mcmc.acceptance_rate:.2f}"
            ),
            posterior_figure(
                x_range=(0.001, 0.999),
                true_pdf=lambda xs: stats.beta.pdf(xs, post_alpha, post_beta),
                laplace=laplace,
                mcmc_samples=mcmc.samples,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
