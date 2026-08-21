"""The singular value decomposition (book §4.2) and what it says about how well- or
ill-posed a linear system is (book §4.4). Every other function in `optimlab.linalg`
builds directly on this — least squares, minimum-norm solutions, and ridge regression
are all just different ways of dividing by the singular values in `SVDResult.s`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike


@dataclass
class SVDResult:
    """`A = U @ diag(s) @ Vt`, with `s` sorted descending (numpy's convention)."""

    U: ArrayLike
    s: ArrayLike
    Vt: ArrayLike

    @property
    def rank(self) -> int:
        """Numerical rank: singular values within `rtol` of the largest one are treated
        as nonzero; anything smaller is noise-dominated and doesn't count.
        """
        return int(np.sum(self.s > self.s[0] * 1e-12)) if self.s.size and self.s[0] > 0 else 0

    @property
    def condition_number(self) -> float:
        """`s.max() / s.min()`: how much a worst-case input perturbation gets amplified
        in the solution. A condition number of `10^k` costs roughly `k` digits of
        precision — see `condition_number()` below for the full explanation this
        property is a shortcut for.
        """
        return condition_number(s=self.s)


def svd(A: ArrayLike) -> SVDResult:
    """Thin (economy-size) SVD of `A`, via `numpy.linalg.svd(..., full_matrices=False)`
    — the from-scratch value here is in the *use* of the decomposition (see
    `regression.py`), not in re-deriving an SVD algorithm itself, which is squarely a
    numerical-linear-algebra topic in its own right rather than an optimization one.
    """
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return SVDResult(U=U, s=s, Vt=Vt)


def condition_number(A: ArrayLike | None = None, *, s: ArrayLike | None = None) -> float:
    """The ratio of the largest to smallest singular value — pass either a matrix `A`
    (its SVD is computed for you) or an already-computed singular-value array `s`.

    Geometrically: `A` maps the unit sphere to an ellipsoid whose semi-axis lengths
    *are* the singular values, so the condition number is exactly how squashed that
    ellipsoid is — `optimlab.viz` visualizes this directly (`svd_conditioning_figure`).
    A well-conditioned `A` (condition number near 1) maps the sphere to something
    close to another sphere: every input direction gets amplified about equally, and a
    small change in `b` produces a correspondingly small change in the solution `x` of
    `Ax=b`. An ill-conditioned `A` maps the sphere to a needle-thin ellipsoid: a tiny
    perturbation along the short axis barely moves `Ax`, so recovering `x` from a
    perturbed `Ax` amplifies that perturbation by (large axis)/(short axis) — the
    condition number. This is the same quantity that made `optimlab.optimizers`'s
    ill-conditioned-quadratic demo (docs Ch. 1) force gradient descent into hundreds of
    zig-zagging steps: that demo's Hessian, `diag(1, 100)`, has condition number 100.
    """
    if s is None:
        if A is None:
            raise ValueError("condition_number needs either A or s")
        s = svd(A).s
    s = np.asarray(s, dtype=float)
    nonzero = s[s > 0]
    if nonzero.size == 0:
        return float("inf")
    return float(nonzero.max() / nonzero.min())
