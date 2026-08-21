"""Expectation-Maximization for a Gaussian Mixture Model: alternate computing each
point's soft cluster assignment (E-step — a closed-form posterior over which component
generated it, given the current parameters) with re-fitting each component's
mean/covariance/weight to those soft assignments (M-step — a closed-form weighted MLE).
Unlike every other solver in this repo, EM never takes a gradient step at all: each
step is itself an exact maximization, just of a *surrogate* objective (the expected
complete-data log-likelihood) rather than the true marginal log-likelihood directly —
which is exactly what guarantees the true log-likelihood never decreases from one
iteration to the next (Jensen's inequality applied to the surrogate), the property
`tests/test_em.py` checks directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike


@dataclass
class GMMResult:
    means: ArrayLike
    covariances: ArrayLike
    weights: ArrayLike
    responsibilities: ArrayLike
    log_likelihood_trajectory: list[float]
    n_iter: int
    converged: bool


def _gaussian_density(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = mean.size
    diff = X - mean
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    return norm_const * np.exp(exponent)


def em_gmm(
    X: ArrayLike, n_components: int, *, max_iter: int = 200, tol: float = 1e-6, seed: int = 0
) -> GMMResult:
    """`X` is `(n_points, n_dim)`. Initializes means at `n_components` randomly chosen
    data points (identity covariances, uniform weights) — a real EM run is only ever a
    *local* maximizer of the likelihood, so which local optimum it lands in genuinely
    depends on this random start, the same "no global guarantee" caveat every
    gradient-free method in Chapter 4 carries.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)

    means = X[rng.choice(n, n_components, replace=False)].copy()
    covariances = np.array([np.eye(d) for _ in range(n_components)])
    weights = np.full(n_components, 1.0 / n_components)

    ll_hist: list[float] = []
    converged = False
    n_iter = 0
    prev_ll = -np.inf
    responsibilities = np.zeros((n, n_components))

    for it in range(max_iter):
        # E-step: soft-assign every point to every component under the *current* params.
        densities = np.stack([_gaussian_density(X, means[k], covariances[k]) for k in range(n_components)], axis=1)
        weighted = densities * weights
        total = weighted.sum(axis=1, keepdims=True)
        responsibilities = weighted / total
        log_likelihood = float(np.sum(np.log(total)))
        ll_hist.append(log_likelihood)

        if abs(log_likelihood - prev_ll) < tol:
            converged = True
            n_iter = it + 1
            break
        prev_ll = log_likelihood

        # M-step: re-fit each component to its (soft-)weighted share of the data.
        n_k = responsibilities.sum(axis=0)
        weights = n_k / n
        means = (responsibilities.T @ X) / n_k[:, None]
        covariances = np.zeros((n_components, d, d))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.einsum("n,ni,nj->ij", responsibilities[:, k], diff, diff) / n_k[k]

        n_iter = it + 1

    return GMMResult(
        means=means, covariances=covariances, weights=weights, responsibilities=responsibilities,
        log_likelihood_trajectory=ll_hist, n_iter=n_iter, converged=converged,
    )
