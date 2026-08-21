"""Maximum likelihood and maximum a posteriori estimation: fitting a probability
model's parameters by turning "find the most likely parameters" into an ordinary
`optimlab.core.Problem` this repo's existing gradient-based solvers already handle.
No new optimization machinery here — just a specific, named choice of objective
(negative log-likelihood, optionally plus a negative log-prior) and BFGS.
"""

from __future__ import annotations

from collections.abc import Callable

from optimlab.core import ArrayLike, Objective, OptimizeResult, Problem
from optimlab.optimizers.quasi_newton import bfgs

Solver = Callable[..., OptimizeResult]


def mle_fit(
    log_likelihood: Objective, x0: ArrayLike, *, name: str = "mle", solver: Solver = bfgs, **solver_kwargs
) -> OptimizeResult:
    """`argmax_params log_likelihood(params, data)` is exactly
    `argmin_params -log_likelihood(params, data)` — flip the sign and it's an
    unconstrained minimization, solved by default with BFGS. `log_likelihood` should
    already have the data baked in (e.g. a closure over a fixed dataset) since
    `Problem` only ever sees a function of `params` alone; write it with
    `jax.numpy` ops if you want the autodiff gradient the default solver needs (see
    `optimlab.core.Problem`). Pass a different `solver` (e.g.
    `optimlab.optimizers.projected_gradient`, with its required `lower`/`upper` in
    `solver_kwargs`) for a parameter with a hard domain boundary — a bare log-likelihood
    for a bounded parameter (a probability, a variance) is usually badly conditioned
    right at that boundary, which plain BFGS has no notion of respecting.
    """
    problem = Problem(f=lambda params: -log_likelihood(params), x0=x0, name=name)
    return solver(problem, **solver_kwargs)


def map_fit(
    log_likelihood: Objective,
    log_prior: Objective,
    x0: ArrayLike,
    *,
    name: str = "map",
    solver: Solver = bfgs,
    **solver_kwargs,
) -> OptimizeResult:
    """MAP adds a log-prior term to MLE's objective — Bayes' rule
    (posterior ∝ likelihood × prior) taken in log space turns a product into a sum, so
    maximizing `log_likelihood + log_prior` maximizes the (unnormalized) log-posterior.
    The normalizing constant `p(data)` never appears: it doesn't depend on `params`, so
    it can't affect where the maximum sits. See `mle_fit` for the `solver`/
    `solver_kwargs` escape hatch a boundary-constrained parameter needs.
    """

    def neg_log_posterior(params: ArrayLike) -> float:
        return -(log_likelihood(params) + log_prior(params))

    problem = Problem(f=neg_log_posterior, x0=x0, name=name)
    return solver(problem, **solver_kwargs)
