"""Fair resource allocation via proportional fairness (Kelly 1997; Network Utility
Maximization): allocate a shared, capacity-limited resource (bandwidth on a network
link, say) across competing users to maximize `sum(log(x_i))` rather than raw
throughput `sum(x_i)`. Log's diminishing returns are exactly what makes "fair"
different from "efficient": maximizing `sum(x_i)` alone would give everything to
whichever user's constraints allow the most usage, while `log`'s steep slope near zero
means starving any one user — even a little — costs the objective a lot, so every user
ends up with a strictly positive share. Solved via `optimlab.optimizers.barrier_method`
(Phase 4): proportional fairness is a genuine inequality-constrained convex problem —
`x > 0` (implicit in `log`) plus each resource's capacity constraint.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike
from optimlab.optimizers.barrier_method import ConstrainedProblem, barrier_method


def proportional_fairness_problem(A: ArrayLike, capacities: ArrayLike) -> ConstrainedProblem:
    """`n` users share `m` capacity-limited resources; `A[j, i]` is how much of
    resource `j` one unit of user `i`'s allocation consumes, `capacities[j]` is
    resource `j`'s total capacity. `f = -sum(log(x_i))` (minimizing the negative is
    maximizing `sum(log(x_i))`) subject to `A @ x <= capacities`.
    """
    A = np.asarray(A, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n = A.shape[1]

    def f(x: ArrayLike) -> float:
        return -jnp.sum(jnp.log(x))

    constraints = [
        (lambda x, row=A[j], cap=capacities[j]: row @ x - cap) for j in range(A.shape[0])
    ]

    # a small, uniformly-strictly-feasible starting allocation: for every resource j,
    # n * x0 * A[j, :].sum() must stay comfortably under capacities[j].
    per_resource_bound = capacities / A.sum(axis=1)
    x0 = np.full(n, 0.5 * per_resource_bound.min())

    return ConstrainedProblem(f=f, x0=x0, inequality_constraints=constraints, name="proportional_fairness")


def solve_fair_allocation(A: ArrayLike, capacities: ArrayLike):
    return barrier_method(proportional_fairness_problem(A, capacities))
