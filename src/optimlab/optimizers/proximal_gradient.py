"""Proximal gradient descent (book §5.1): minimize `g(x) + h(x)` where `g` is smooth (has
a gradient) but `h` need not be — the workhorse for objectives like LASSO
(`h = alpha * ||x||_1`) that are only *piecewise* smooth. Ordinary gradient descent has
no answer for the kink in `|x_i|` at zero; proximal gradient sidesteps it by alternating
an ordinary gradient step on `g` with a **proximal step** that handles `h` exactly:

    x_{k+1} = prox_{lr*h}( x_k - lr * grad(g)(x_k) ),    prox_{t*h}(v) = argmin_x  h(x) + (1/2t)||x-v||^2

`optimlab.optimizers.projected_gradient` is the special case `h` = the indicator
function of a box (0 inside, infinity outside) — its proximal operator is exactly
clipping to the box, which is why that solver's update looks like ordinary projected
gradient rather than anything exotic: it *is* proximal gradient, just with the one prox
operator that happens to look like clipping. `soft_threshold` below is `h`'s proximal
operator for the LASSO case, `h(x) = alpha||x||_1`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, GradFn, Objective, OptimizeResult, track_iterations


def soft_threshold(x: ArrayLike, threshold: float) -> ArrayLike:
    """The proximal operator of `alpha * ||.||_1`, with `threshold = alpha * lr`: shrink
    every coordinate toward zero by `threshold`, clamping anything that would cross zero
    to exactly zero. This is *why* LASSO produces exactly-sparse solutions where ridge
    regression (Chapter 3) only ever shrinks toward, never to, zero — ridge's proximal
    step is linear rescaling (`x / (1 + t*alpha)`, never hits zero exactly for finite
    `alpha`); soft-thresholding has a flat dead zone `[-threshold, threshold]` that maps
    to exactly zero.
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@dataclass
class CompositeProblem:
    """Minimize `f_smooth(x) + f_nonsmooth(x)`. `grad_smooth` is `f_smooth`'s gradient
    (no autodiff fallback here, unlike `optimlab.core.Problem` — proximal methods are
    usually reached for exactly when `f_nonsmooth` isn't differentiable, so autodiff
    through the *whole* objective isn't on the table anyway); `prox_nonsmooth(v, t)`
    must return `argmin_x f_nonsmooth(x) + ||x-v||^2/(2t)`. `f_smooth`/`f_nonsmooth`
    themselves are optional and only used to report the objective value in
    `OptimizeResult.f_trajectory` — the algorithm itself never evaluates them.
    """

    grad_smooth: GradFn
    prox_nonsmooth: Callable[[ArrayLike, float], ArrayLike]
    x0: ArrayLike
    f_smooth: Objective | None = None
    f_nonsmooth: Objective | None = None
    name: str = "composite_problem"

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)

    @property
    def n_dim(self) -> int:
        return int(self.x0.size)

    def f(self, x: ArrayLike) -> float:
        total = 0.0
        if self.f_smooth is not None:
            total += float(self.f_smooth(x))
        if self.f_nonsmooth is not None:
            total += float(self.f_nonsmooth(x))
        return total


def proximal_gradient(
    problem: CompositeProblem,
    *,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizeResult:
    """`lr` should be at most `1 / L`, where `L` is `f_smooth`'s Lipschitz-continuous-
    gradient constant (its largest curvature) — the same fixed-step stability bound
    `optimlab.optimizers.gradient_descent` needs, since a proximal step degenerates to
    ordinary gradient descent whenever `f_nonsmooth` is identically zero. Convergence is
    checked via the same "generalized" gradient norm `projected_gradient` uses,
    `||x - x_next|| / lr` — zero exactly when neither the smooth step nor the proximal
    step wants to move `x` any further.
    """
    x = problem.x0.copy()
    grad = problem.grad_smooth(x)
    x_next = problem.prox_nonsmooth(x - lr * grad, lr)
    step_norm = float(np.linalg.norm(x - x_next) / lr)

    x_hist, f_hist, g_hist = [x.copy()], [problem.f(x)], [step_norm]

    converged = step_norm < tol
    n_iter = 0
    while not converged and n_iter < max_iter:
        x = x_next
        grad = problem.grad_smooth(x)
        x_next = problem.prox_nonsmooth(x - lr * grad, lr)
        step_norm = float(np.linalg.norm(x - x_next) / lr)

        n_iter += 1
        x_hist.append(x.copy())
        f_hist.append(problem.f(x))
        g_hist.append(step_norm)
        if step_norm < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="proximal_gradient",
        message="generalized gradient norm below tol" if converged else "max_iter reached",
    )
