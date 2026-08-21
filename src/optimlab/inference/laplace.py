"""The Laplace approximation: approximate a (possibly non-Gaussian) posterior with a
Gaussian centered at its mode — a second-order Taylor expansion of the log-posterior
around its peak, kept only to quadratic order. Cheap (one MAP fit, one Hessian) and
exact whenever the true posterior already is Gaussian (a conjugate Gaussian-Gaussian
model); increasingly wrong the more the true posterior's shape deviates from a bell
curve — skewed, multimodal, or heavy-tailed — which `optimlab.inference.mcmc` exists to
handle instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, Objective, _autohess


@dataclass
class GaussianApprox:
    mean: ArrayLike
    cov: ArrayLike


def laplace_approximation(log_likelihood: Objective, log_prior: Objective, map_estimate: ArrayLike) -> GaussianApprox:
    """`mean` is just the supplied MAP estimate; `cov` is the inverse Hessian of the
    negative log-posterior *at* that point — the curvature of a log-density's peak sets
    the width of the Gaussian that best matches it there, sharper curvature (a more
    confident peak) giving a narrower approximation. Uses the exact same
    `_autohess` autodiff path `optimlab.core.Problem` uses for `hess`, so
    `log_likelihood`/`log_prior` need the same `jax.numpy`-compatible authoring as any
    other `Problem` objective.
    """
    map_estimate = np.asarray(map_estimate, dtype=float)

    def neg_log_posterior(params: ArrayLike) -> float:
        return -(log_likelihood(params) + log_prior(params))

    hessian = _autohess(neg_log_posterior)(map_estimate)
    cov = np.linalg.inv(hessian)
    return GaussianApprox(mean=map_estimate, cov=cov)
