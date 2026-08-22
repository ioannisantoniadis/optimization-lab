"""Portfolio optimization (Markowitz 1952): the minimum-variance portfolio achieving a
given target expected return, subject to the two equality constraints
`optimlab.linalg.qp.equality_constrained_qp` already solves in closed form — weights
sum to 1 (fully invested) and expected return matches the target. Sweeping the target
return traces the **efficient frontier**: for a given amount of risk, this is literally
the best (lowest-variance) reward achievable, and vice versa. Short sales are allowed
(weights can be negative) — the simplest version of the model, where the problem stays
a pure equality-constrained QP rather than needing the general inequality machinery a
no-short-selling constraint (`w >= 0`) would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike
from optimlab.linalg.qp import equality_constrained_qp


def minimum_variance_portfolio(cov: ArrayLike, expected_returns: ArrayLike, target_return: float) -> np.ndarray:
    """Portfolio weights minimizing `0.5 w^T cov w` subject to `sum(w) = 1` and
    `w @ expected_returns = target_return`.
    """
    cov = np.asarray(cov, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    n = cov.shape[0]
    A_eq = np.vstack([np.ones(n), expected_returns])
    b_eq = np.array([1.0, target_return])
    return equality_constrained_qp(cov, np.zeros(n), A_eq, b_eq)


@dataclass
class EfficientFrontier:
    target_returns: np.ndarray
    risks: np.ndarray  # portfolio std dev at each target return
    weights: np.ndarray  # (n_targets, n_assets)


def efficient_frontier(cov: ArrayLike, expected_returns: ArrayLike, target_returns: ArrayLike) -> EfficientFrontier:
    """`minimum_variance_portfolio` swept across every return in `target_returns` —
    the classic Markowitz hyperbola: risk is minimized at the "global minimum variance"
    portfolio and rises on *either* side of it, since demanding either a much higher or
    a much lower return than that vertex both force more concentrated (less
    diversified) weights.
    """
    cov = np.asarray(cov, dtype=float)
    target_returns = np.asarray(target_returns, dtype=float)
    weights = np.array([minimum_variance_portfolio(cov, expected_returns, r) for r in target_returns])
    risks = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    return EfficientFrontier(target_returns=target_returns, risks=risks, weights=weights)
