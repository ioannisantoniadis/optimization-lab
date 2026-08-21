"""The linear conjugate gradient method (book §4.8): solves `A x = b` for symmetric
positive-definite `A` without ever forming `A^{-1}`, using only matrix-vector products.

Framed here as *minimizing* the quadratic `f(x) = 0.5 x^T A x - b^T x` rather than
"solving a linear system" — the two are the same problem (its unique stationary point,
`grad f(x) = A x - b = 0`, is exactly the linear system's solution), and phrasing it this
way is what makes CG belong next to the other from-scratch solvers in
`optimlab.optimizers`, sharing their `OptimizeResult` output. Textbook-remarkable
property: for exact arithmetic, CG reaches the exact solution in at most `n` steps
(`n` = dimension) — very different from gradient descent's asymptotic convergence,
because each step is constructed to be exactly orthogonal (in the `A`-inner-product
sense) to every previous step, so it can't waste an iteration re-covering old ground.
"""

from __future__ import annotations

import numpy as np

from optimlab.core import ArrayLike, OptimizeResult, track_iterations


def conjugate_gradient(
    A: ArrayLike,
    b: ArrayLike,
    *,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> OptimizeResult:
    """Solve `A x = b` for symmetric positive-definite `A`. `max_iter` defaults to `2n`
    (comfortably above the `n`-step exact-arithmetic bound, to leave room for the
    rounding error real floating-point CG accumulates over many steps).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    max_iter = 2 * n if max_iter is None else max_iter

    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)

    def objective(x: ArrayLike) -> float:
        return float(0.5 * x @ A @ x - b @ x)

    x_hist = [x.copy()]
    f_hist = [objective(x)]
    g_hist = [float(np.sqrt(rs_old))]

    converged = g_hist[0] < tol
    n_iter = 0
    while not converged and n_iter < max_iter:
        Ap = A @ p
        alpha = rs_old / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)

        x_hist.append(x.copy())
        f_hist.append(objective(x))
        g_hist.append(float(np.sqrt(rs_new)))
        n_iter += 1

        if np.sqrt(rs_new) < tol:
            converged = True
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="conjugate_gradient",
        message="residual norm below tol" if converged else "max_iter reached",
    )
