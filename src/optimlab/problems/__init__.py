"""Cross-domain problem library: physics, economics, sociology/networks, machine
learning, and a personal "life as optimization" case study. Each problem exposes a
`Problem`-shaped interface (see `optimlab.core`) plus a short write-up of its
parameterization and modeling assumptions.

Physics already has its worked problem — the pendulum swing-up in
`optimlab.control.trajectory_optimization` (Phase 7) — rather than a duplicate entry
here. Machine learning's domain problem (hyperparameter search, Optuna vs. a
from-scratch Bayesian optimizer) lives in `optimlab.optimizers.bayesian_optimization`
for the same reason: it's an optimizer comparison, not a new problem shape.
"""

from optimlab.problems.economics import (
    EfficientFrontier,
    efficient_frontier,
    minimum_variance_portfolio,
)
from optimlab.problems.sociology import proportional_fairness_problem, solve_fair_allocation

__all__ = [
    "EfficientFrontier",
    "efficient_frontier",
    "minimum_variance_portfolio",
    "proportional_fairness_problem",
    "solve_fair_allocation",
]
