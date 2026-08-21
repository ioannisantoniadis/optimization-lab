"""Saddle points dominate as dimension grows, not bad local minima (Dauphin et al. 2014;
the spin-glass argument in Choromanska et al. 2015): a local landscape near any critical
point is described, to second order, by its Hessian — a local *minimum* needs every
eigenvalue positive, a local *maximum* needs every eigenvalue negative, anything mixed is
a saddle. Model "a random critical point's Hessian" as a random symmetric matrix (a GOE
matrix — a standard random-matrix-theory model with no special structure assumed) and ask
how likely an all-one-sign eigenvalue spectrum is as the dimension grows. It collapses
fast: eigenvalues of a random symmetric matrix repel each other (Wigner-Dyson statistics),
so pinning all `n` of them to the same sign becomes exponentially unlikely in `n`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_goe_eigenvalues(dim: int, n_samples: int, *, seed: int = 0) -> np.ndarray:
    """`n_samples` draws of a `dim x dim` GOE (Gaussian Orthogonal Ensemble) matrix's
    eigenvalues, one row per sample. `H = (M + M^T) / sqrt(2*dim)`, `M` iid standard
    normal, is the standard symmetric random matrix with no assumed structure — the
    `1/sqrt(2*dim)` normalization keeps the eigenvalue spread `O(1)` regardless of `dim`
    (Wigner's semicircle law), so eigenvalue *signs* are comparable across dimensions.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.empty((n_samples, dim))
    for i in range(n_samples):
        M = rng.standard_normal((dim, dim))
        H = (M + M.T) / np.sqrt(2.0 * dim)
        eigenvalues[i] = np.linalg.eigvalsh(H)
    return eigenvalues


@dataclass
class CriticalPointStats:
    dims: np.ndarray
    p_local_min: np.ndarray
    p_local_max: np.ndarray
    p_saddle: np.ndarray


def critical_point_index_stats(dims: list[int], n_samples: int, *, seed: int = 0) -> CriticalPointStats:
    """For each dimension in `dims`, the fraction of `n_samples` random GOE Hessians
    that are a local min (every eigenvalue `> 0`), a local max (every eigenvalue `< 0`),
    or a saddle (mixed signs) — a direct, from-scratch instance of the random-matrix
    argument behind "saddle points, not bad local minima, dominate high-dimensional
    non-convex landscapes."
    """
    p_min, p_max, p_saddle = [], [], []
    for i, dim in enumerate(dims):
        eigenvalues = sample_goe_eigenvalues(dim, n_samples, seed=seed + i)
        all_positive = np.all(eigenvalues > 0, axis=1)
        all_negative = np.all(eigenvalues < 0, axis=1)
        p_min.append(float(np.mean(all_positive)))
        p_max.append(float(np.mean(all_negative)))
        p_saddle.append(float(np.mean(~(all_positive | all_negative))))
    return CriticalPointStats(
        dims=np.asarray(dims), p_local_min=np.asarray(p_min),
        p_local_max=np.asarray(p_max), p_saddle=np.asarray(p_saddle),
    )
