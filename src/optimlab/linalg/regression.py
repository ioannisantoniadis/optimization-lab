"""Least-squares regression (book §4.1), minimum-norm solutions for underdetermined
systems (book §4.3), and ridge / Tikhonov regularization (book §4.4) — implemented as
three variations on the exact same operation: divide by the singular values of `A`, then
map back through `U`/`V`. Nonlinear least squares (book §4.7) needs actual iteration
instead and lives in `optimlab.optimizers.gauss_newton`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike
from optimlab.linalg.svd import SVDResult, svd


@dataclass
class LeastSquaresResult:
    x: ArrayLike
    residual_norm: float
    rank: int
    singular_values: ArrayLike
    condition_number: float


def least_squares(A: ArrayLike, b: ArrayLike, *, rcond: float = 1e-12) -> LeastSquaresResult:
    """Solve `min_x ||Ax - b||_2` via the Moore-Penrose pseudoinverse, built directly
    from `A`'s SVD: `x = V @ diag(1/s_i, or 0 if s_i is ~0) @ U^T @ b`.

    This one formula quietly covers three cases the book treats separately:

    - **Overdetermined, full rank** (book §4.1, the usual regression setup, `m > n`):
      the unique least-squares solution, identical to solving the normal equations
      `A^T A x = A^T b` but without ever forming the (worse-conditioned,
      condition-number-*squared*) matrix `A^T A`.
    - **Underdetermined** (book §4.3, `m < n`, infinitely many exact solutions): zeroing
      the near-zero singular directions' contribution automatically selects the
      *minimum-norm* one among them — there's no separate "minimum norm" algorithm,
      it's a free consequence of the pseudoinverse.
    - **Rank-deficient** (`A` doesn't have full column rank, over- or under-determined):
      singular values below `rcond * s.max()` are treated as zero rather than divided
      by, which is what keeps a near-zero singular value from blowing up the solution
      with noise amplified by `1/s_i` — see `condition_number()` for why that blow-up
      would otherwise happen.
    """
    result = svd(A)
    x = _pinv_apply(result, np.asarray(b, dtype=float), rcond=rcond)
    residual = np.asarray(A, dtype=float) @ x - np.asarray(b, dtype=float)
    return LeastSquaresResult(
        x=x, residual_norm=float(np.linalg.norm(residual)),
        rank=result.rank, singular_values=result.s, condition_number=result.condition_number,
    )


def ridge_regression(A: ArrayLike, b: ArrayLike, alpha: float) -> LeastSquaresResult:
    """Tikhonov-regularized least squares: `min_x ||Ax - b||_2^2 + alpha ||x||_2^2`.

    Same SVD, different divisor: ordinary least squares divides each singular
    direction's contribution by `s_i`; ridge divides by `s_i + alpha/s_i` instead
    (equivalently `s_i / (s_i^2 + alpha)`), which barely changes a large, well-trusted
    singular value but aggressively shrinks a small one — exactly the directions
    `condition_number()` identifies as the ones that amplify noise. `alpha=0` recovers
    plain `least_squares` exactly (no rank truncation, since ridge's smooth shrinkage
    makes an `rcond` cutoff unnecessary: a true zero singular value just contributes
    zero either way).
    """
    if alpha < 0:
        raise ValueError(f"alpha must be nonnegative, got {alpha}")
    result = svd(A)
    b = np.asarray(b, dtype=float)
    shrinkage = result.s / (result.s**2 + alpha)
    x = result.Vt.T @ (shrinkage * (result.U.T @ b))
    residual = np.asarray(A, dtype=float) @ x - b
    return LeastSquaresResult(
        x=x, residual_norm=float(np.linalg.norm(residual)),
        rank=result.rank, singular_values=result.s, condition_number=result.condition_number,
    )


def equality_constrained_least_squares(A: ArrayLike, b: ArrayLike, C: ArrayLike, d: ArrayLike) -> ArrayLike:
    """Solve `min_x ||Ax - b||_2` subject to the hard linear constraint `Cx = d` (book
    §4.5). Setting the Lagrangian's gradient to zero turns this into one linear KKT
    system — `optimlab.linalg.qp.equality_constrained_qp` solves the identical system
    for the general quadratic case; this is that function specialized to a sum-of-squares
    objective (`P = A^T A`, `q = -A^T b`), kept separate because "least squares with an
    equality constraint" reads better in the regression module than a `P`/`q` QP call.
    """
    from optimlab.linalg.qp import equality_constrained_qp

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    return equality_constrained_qp(P=A.T @ A, q=-A.T @ b, A_eq=C, b_eq=d)


def _pinv_apply(result: SVDResult, b: ArrayLike, *, rcond: float) -> ArrayLike:
    threshold = rcond * result.s[0] if result.s.size else 0.0
    inv_s = np.where(result.s > threshold, 1.0 / np.where(result.s > 0, result.s, 1.0), 0.0)
    return result.Vt.T @ (inv_s * (result.U.T @ b))
