"""Bayesian optimization from scratch: a genuinely different way to minimize a
black-box function than anything else in this repo. Rather than following a gradient
or evolving a population, maintain a probabilistic surrogate model — a Gaussian
process, built from every point evaluated so far — of the objective, and pick the next
point by maximizing an acquisition function (Expected Improvement) that explicitly
trades exploring uncertain regions against exploiting the surrogate's current best
guess. The point of the extra machinery: this is the right tool when each evaluation is
assumed *expensive* (a real hyperparameter-tuning run, not a cheap benchmark
function) — the surrogate model earns back its own overhead by using every previous
evaluation to choose the next one as informedly as possible, unlike a population method
that needs many cheap evaluations to make progress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from optimlab.core import ArrayLike, OptimizeResult, Problem


def rbf_kernel(X1: ArrayLike, X2: ArrayLike, *, length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    """The squared-exponential covariance `variance * exp(-0.5 ||x-x'||^2 / length_scale^2)`
    between every pair of rows in `X1` and `X2` — points close together (relative to
    `length_scale`) are assumed to have similar function values, points far apart are
    treated as nearly independent. This one assumption is the entire content of the
    Gaussian process below.
    """
    X1, X2 = np.asarray(X1, dtype=float), np.asarray(X2, dtype=float)
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
    return variance * np.exp(-0.5 * np.maximum(sq_dists, 0.0) / length_scale**2)


@dataclass
class GPPosterior:
    mean: np.ndarray
    std: np.ndarray


def gp_posterior(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike,
    *, length_scale: float = 1.0, variance: float = 1.0, noise: float = 1e-6,
) -> GPPosterior:
    """The standard closed-form Gaussian process regression posterior — conditioning a
    jointly-Gaussian prior on the observed `(X_train, y_train)` pairs gives another
    Gaussian at every test point, mean and variance both available in closed form (no
    iteration needed, unlike almost every other estimate in this repo). `noise` is
    added to the training kernel's diagonal both to model genuine observation noise and
    to keep the matrix safely invertible when two training points sit very close
    together.
    """
    X_train, y_train, X_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test)
    K = rbf_kernel(X_train, X_train, length_scale=length_scale, variance=variance) + noise * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale=length_scale, variance=variance)
    K_ss = rbf_kernel(X_test, X_test, length_scale=length_scale, variance=variance)

    K_inv = np.linalg.inv(K)
    mean = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return GPPosterior(mean=mean, std=std)


def expected_improvement(mean: ArrayLike, std: ArrayLike, best_so_far: float, *, xi: float = 0.01) -> np.ndarray:
    """How much a candidate point is expected to improve on `best_so_far` (the best
    objective value seen yet), in closed form for a Gaussian belief about that point's
    value — large where the surrogate's mean is promising *or* where its uncertainty is
    high (a point that's never been sampled always has some chance of being great),
    exactly the explore/exploit balance Bayesian optimization is named for. `xi` is a
    small margin that keeps EI from collapsing to zero right at already-sampled points.
    """
    mean, std = np.asarray(mean), np.asarray(std)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (best_so_far - mean - xi) / std
        ei = (best_so_far - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 1e-9, ei, 0.0)


def bayesian_optimize(
    problem: Problem,
    *,
    bounds: tuple[float, float] | None = None,
    n_init: int = 5,
    n_iter: int = 20,
    length_scale: float = 1.0,
    variance: float = 1.0,
    seed: int = 0,
) -> OptimizeResult:
    """`n_init` random points, then `n_iter` rounds of (fit GP on everything evaluated
    so far -> maximize Expected Improvement over a random candidate set -> evaluate the
    winner). `bounds` defaults to `problem.domain`, the same convention
    `genetic_algorithm`/`particle_swarm` use, so this plugs into `optimlab.arena` the
    identical way every other gradient-free method does.
    """
    if bounds is None and problem.domain is None:
        raise ValueError("bayesian_optimize needs `bounds` (problem.domain is unset)")
    low, high = bounds if bounds is not None else problem.domain
    n_dim = problem.n_dim
    rng = np.random.default_rng(seed)

    X = low + (high - low) * rng.random((n_init, n_dim))
    y = np.array([float(problem.f(x)) for x in X])

    for _ in range(n_iter):
        candidates = low + (high - low) * rng.random((1000, n_dim))
        posterior = gp_posterior(X, y, candidates, length_scale=length_scale, variance=variance)
        ei = expected_improvement(posterior.mean, posterior.std, float(y.min()))
        next_x = candidates[np.argmax(ei)]
        next_y = float(problem.f(next_x))
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)

    best_idx = int(np.argmin(y))
    running_best = np.minimum.accumulate(y)
    return OptimizeResult(
        x=X[best_idx], f=float(y[best_idx]), n_iter=n_iter, converged=True,
        solver_name="bayesian_optimize", message="n_iter completed",
        trajectory=list(X), f_trajectory=list(running_best),
    )
