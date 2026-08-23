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

import importlib

from optimlab.problems.economics import (
    EfficientFrontier,
    efficient_frontier,
    minimum_variance_portfolio,
)

__all__ = [
    "EfficientFrontier",
    "efficient_frontier",
    "minimum_variance_portfolio",
    "proportional_fairness_problem",
    "solve_fair_allocation",
]

#: `optimlab.problems.sociology` hard-imports jax at module level (jax has no
#: WebAssembly/Pyodide build), so importing it eagerly here would break
#: `optimlab.problems` for every caller — including economics-only callers like the
#: WASM-exported marimo notebooks. Resolved lazily via PEP 562.
_LAZY = {
    "proportional_fairness_problem": "optimlab.problems.sociology",
    "solve_fair_allocation": "optimlab.problems.sociology",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
