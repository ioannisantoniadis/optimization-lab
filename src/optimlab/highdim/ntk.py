"""Overparameterization makes training easier, not harder (the Neural Tangent Kernel;
Jacot et al. 2018, Du et al. 2019 for the "gradient descent provably converges" result
this enables). A network's *tangent kernel* — how the network's own output would change
in response to an infinitesimal parameter step, `K = J @ J^T` where `J` is the Jacobian
of outputs with respect to parameters — is, in the infinite-width limit, a *fixed,
deterministic* object independent of the random initialization: training becomes
equivalent to kernel regression against that fixed kernel ("lazy training"), which is
exactly the ingredient the convergence proofs need. Measured here the direct way: draw
several independent random initializations at each width and see how much their
empirical NTKs actually differ from each other — Jacot et al.'s claim predicts that
difference should shrink toward zero as width grows, checked directly rather than
derived from the infinite-width limit itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike
from optimlab.highdim.nets import MLPShape, forward, init_params


def ntk_matrix(flat_params: ArrayLike, shape: MLPShape, X: ArrayLike) -> np.ndarray:
    """The empirical (finite-width) neural tangent kernel `K = J @ J^T` at
    `flat_params`, evaluated on samples `X` — `J[i]` is the gradient of the network's
    output at `X[i]` with respect to every parameter, flattened over any multi-output
    dimension, computed in one `jax.jacobian` call rather than one gradient per sample.
    """

    def output_fn(params: ArrayLike) -> ArrayLike:
        return forward(params, shape, X).reshape(-1)

    J = np.asarray(jax.jacobian(output_fn)(jnp.asarray(flat_params, dtype=jnp.float64)))
    return J @ J.T


def relative_frobenius_difference(A: np.ndarray, B: np.ndarray) -> float:
    """`||A - B||_F / ||A||_F` — how different two matrices are, relative to `A`'s own
    scale.
    """
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


@dataclass
class NTKConcentrationResult:
    widths: np.ndarray
    mean_relative_diff: np.ndarray
    std_relative_diff: np.ndarray


def ntk_concentration_experiment(
    widths: list[int], X: ArrayLike, *, n_in: int, n_out: int = 1, n_seed_pairs: int = 5, seed: int = 0
) -> NTKConcentrationResult:
    """For each width, draw `n_seed_pairs` independent pairs of random initializations
    of an otherwise-identical architecture, compute each pair's empirical NTK, and
    measure how different the two NTKs in a pair are (`relative_frobenius_difference`)
    — averaged over pairs to smooth out the high variance any single pair has at small
    width. Widths must be swept with `X` fixed so every width's NTK is directly
    comparable (a bigger sample set alone would also shrink relative noise, which isn't
    the effect being measured here).
    """
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for width in widths:
        arch = MLPShape(layer_sizes=[n_in, width, n_out])
        diffs = []
        for _ in range(n_seed_pairs):
            seed_a, seed_b = (int(s) for s in rng.integers(0, 2**31 - 1, size=2))
            K_a = ntk_matrix(init_params(arch, seed=seed_a), arch, X)
            K_b = ntk_matrix(init_params(arch, seed=seed_b), arch, X)
            diffs.append(relative_frobenius_difference(K_a, K_b))
        means.append(float(np.mean(diffs)))
        stds.append(float(np.std(diffs)))
    return NTKConcentrationResult(
        widths=np.asarray(widths), mean_relative_diff=np.asarray(means), std_relative_diff=np.asarray(stds)
    )
