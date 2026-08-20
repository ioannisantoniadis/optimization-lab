"""Step-length selection (book §2.5). Every optimizer that isn't fixed-step calls into
one of these rather than re-implementing its own line search.
"""

from __future__ import annotations

from optimlab.core import ArrayLike, GradFn, Objective


def backtracking_armijo(
    f: Objective,
    grad_f: GradFn,
    x: ArrayLike,
    direction: ArrayLike,
    *,
    f_x: float | None = None,
    grad_x: ArrayLike | None = None,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 50,
) -> float:
    """Shrink `alpha` by `rho` until the Armijo sufficient-decrease condition holds:
    f(x + alpha*d) <= f(x) + c1*alpha*grad(x)@d. Cheap (no curvature/derivative checks
    at each trial point) and the workhorse behind damped Newton and line-searched GD.
    """
    f_x = f(x) if f_x is None else f_x
    grad_x = grad_f(x) if grad_x is None else grad_x
    directional_deriv = grad_x @ direction

    alpha = alpha0
    for _ in range(max_iter):
        if f(x + alpha * direction) <= f_x + c1 * alpha * directional_deriv:
            return alpha
        alpha *= rho
    return alpha


def strong_wolfe_line_search(
    f: Objective,
    grad_f: GradFn,
    x: ArrayLike,
    direction: ArrayLike,
    *,
    f_x: float | None = None,
    grad_x: ArrayLike | None = None,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_max: float = 10.0,
    max_iter: int = 25,
) -> float:
    """Bracket-and-bisect search for a step length satisfying the strong Wolfe conditions
    (sufficient decrease + curvature). BFGS/L-BFGS need this, not just Armijo: Wolfe's
    curvature condition is what guarantees the s^T y > 0 curvature-pair used in the BFGS
    update stays valid, so a plain backtracking search can silently break BFGS.

    Simplified relative to Nocedal & Wright Algorithm 3.5: bisection instead of cubic
    interpolation in the "zoom" phase. Costs a few extra function evaluations in
    practice; easier to read and verify, which is the point of a from-scratch version.
    """
    f_x = f(x) if f_x is None else f_x
    grad_x = grad_f(x) if grad_x is None else grad_x
    dphi0 = grad_x @ direction

    def phi(alpha: float) -> float:
        return f(x + alpha * direction)

    def dphi(alpha: float) -> float:
        return grad_f(x + alpha * direction) @ direction

    alpha_lo, alpha_hi = 0.0, alpha_max
    alpha = 1.0
    for _ in range(max_iter):
        f_alpha = phi(alpha)
        if f_alpha > f_x + c1 * alpha * dphi0 or (alpha_lo > 0 and f_alpha >= phi(alpha_lo)):
            alpha_hi = alpha
        else:
            d_alpha = dphi(alpha)
            if abs(d_alpha) <= -c2 * dphi0:
                return alpha
            if d_alpha * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
        alpha = 0.5 * (alpha_lo + alpha_hi)
    return alpha


__all__ = ["backtracking_armijo", "strong_wolfe_line_search"]
