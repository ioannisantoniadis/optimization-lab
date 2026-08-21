"""Metropolis-Hastings: a from-scratch random-walk MCMC sampler that draws samples from
a (possibly unnormalized) posterior without ever needing its normalizing constant —
only the *ratio* of densities at a proposed vs. current point ever appears in the
acceptance rule, and that constant cancels out of any ratio. Where `optimlab.inference
.laplace` approximates the posterior's *shape* with a single Gaussian guess, MCMC
approximates it with *samples* — slower, but faithful to skew, multimodality, or heavy
tails a Gaussian can't represent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, Objective


@dataclass
class MCMCResult:
    samples: ArrayLike
    acceptance_rate: float


def metropolis_hastings(
    log_posterior: Objective,
    x0: ArrayLike,
    *,
    n_samples: int = 5000,
    proposal_std: float = 1.0,
    burn_in: int = 1000,
    seed: int = 0,
) -> MCMCResult:
    """Propose `x + Normal(0, proposal_std)`, accept it with probability
    `min(1, exp(log_posterior(proposal) - log_posterior(current)))`. A *symmetric*
    proposal (a Gaussian centered at the current point) is what makes the proposal
    density's own contribution cancel exactly out of the Metropolis-Hastings
    acceptance ratio, leaving nothing but the posterior ratio coded below — this
    specific symmetric case is sometimes called plain "Metropolis." `burn_in` samples
    are drawn and discarded before keeping any, so the chain has a chance to wander
    away from `x0` into the posterior's actual high-density region first.
    """
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    log_p_x = float(log_posterior(x))

    samples = np.empty((n_samples, x.size))
    n_accepted = 0
    total = n_samples + burn_in
    kept = 0
    for i in range(total):
        proposal = x + proposal_std * rng.standard_normal(x.size)
        log_p_proposal = float(log_posterior(proposal))
        if np.log(rng.uniform()) < log_p_proposal - log_p_x:
            x, log_p_x = proposal, log_p_proposal
            n_accepted += 1
        if i >= burn_in:
            samples[kept] = x
            kept += 1

    return MCMCResult(samples=samples, acceptance_rate=n_accepted / total)
