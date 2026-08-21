"""Estimating a Hessian's eigenspectrum without ever forming the full matrix: Sagun et
al. 2016 and Ghorbani et al. 2019 found a small number of large *outlier* eigenvalues
sitting on a large near-zero *bulk* in real trained networks' loss Hessians — a shape
computable here via a Hessian-vector product (the Hessian is only ever touched through
matvecs, never materialized as an `n x n` array) fed into the Lanczos algorithm, the
standard iterative method for a symmetric matrix's extreme eigenvalues from matvecs
alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike, Objective


def hessian_vector_product(f: Objective, x: ArrayLike, v: ArrayLike) -> np.ndarray:
    """`H @ v` at `x`, `H = hess(f)(x)` — computed as the directional derivative of
    `grad(f)` along `v` ("forward-over-reverse" autodiff via `jax.jvp` of `jax.grad`),
    at roughly the cost of one extra gradient evaluation rather than forming the full
    `n x n` Hessian and multiplying.
    """
    grad_f = jax.grad(lambda z: f(z))
    x_j = jnp.asarray(x, dtype=jnp.float64)
    v_j = jnp.asarray(v, dtype=jnp.float64)
    _, hv = jax.jvp(grad_f, (x_j,), (v_j,))
    return np.asarray(hv)


@dataclass
class LanczosResult:
    ritz_values: np.ndarray
    n_iter: int


def lanczos_eigenvalues(f: Objective, x: ArrayLike, *, n_iter: int = 50, seed: int = 0) -> LanczosResult:
    """Ritz values (approximate eigenvalues) of `hess(f)(x)` from `n_iter`
    Hessian-vector products alone. Lanczos tridiagonalizes the Hessian's restriction to
    the Krylov subspace spanned by repeated `H @ v` products from a random start
    vector, and that small tridiagonal matrix's *own* eigenvalues converge to the true
    Hessian's *extreme* eigenvalues fastest — long before the full spectrum is
    resolved. That's exactly the "few large outliers" regime this function targets: a
    modest `n_iter` reliably reveals the largest eigenvalues even though the bulk near
    zero would need many more iterations to fill in accurately.

    Full reorthogonalization (checking every new Krylov vector against every previous
    one) trades away Lanczos's usual `O(n)`-memory advantage for numerical stability at
    the iteration counts used here — floating-point rounding makes the bare three-term
    recursion lose orthogonality after only a few dozen steps otherwise, corrupting the
    Ritz values it's supposed to compute.
    """
    x = np.asarray(x, dtype=float)
    dim = x.size
    rng = np.random.default_rng(seed)

    def hvp(v: np.ndarray) -> np.ndarray:
        return hessian_vector_product(f, x, v)

    v0 = rng.standard_normal(dim)
    v0 /= np.linalg.norm(v0)
    basis = [v0]
    alphas: list[float] = []
    betas: list[float] = []
    v_prev = np.zeros(dim)
    beta_prev = 0.0

    for j in range(n_iter):
        w = hvp(basis[j])
        alpha = float(w @ basis[j])
        alphas.append(alpha)
        w = w - alpha * basis[j] - beta_prev * v_prev
        for vi in basis:
            w -= (w @ vi) * vi  # full reorthogonalization

        if j == n_iter - 1:
            break
        beta = float(np.linalg.norm(w))
        if beta < 1e-10:
            w = rng.standard_normal(dim)
            for vi in basis:
                w -= (w @ vi) * vi
            beta = float(np.linalg.norm(w))
        betas.append(beta)
        v_prev = basis[j]
        beta_prev = beta
        basis.append(w / beta)

    T = np.diag(alphas)
    if betas:
        T += np.diag(betas, k=1) + np.diag(betas, k=-1)
    return LanczosResult(ritz_values=np.linalg.eigvalsh(T), n_iter=len(alphas))
