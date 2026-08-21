"""ADMM — the Alternating Direction Method of Multipliers (book §6.5): another way to
minimize `f(x) + g(z)` for a smooth `f` and possibly-nonsmooth `g`, alongside
`optimlab.optimizers.proximal_gradient`. Where proximal gradient takes an *explicit*
gradient step on `f` and a proximal step on `g`, ADMM never touches a gradient at all —
it needs only the two pieces' proximal operators, alternating a proximal step on each
against the constraint `x = z` tying them together, with a running "dual" variable `u`
(a scaled Lagrange multiplier for that constraint) nudging the two into agreement:

    x_{k+1} = prox_{f/rho}(z_k - u_k)
    z_{k+1} = prox_{g/rho}(x_{k+1} + u_k)
    u_{k+1} = u_k + x_{k+1} - z_{k+1}

`x` and `z` both solve the same problem at convergence (`x_{k+1} - z_{k+1} -> 0`, the
**primal residual** this module uses as its stopping criterion); they differ during the
run because each is mid-way through its own proximal step, not yet reconciled with the
other's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike, Objective, OptimizeResult, track_iterations


@dataclass
class ADMMProblem:
    """Minimize `f_obj(x) + g_obj(z)` subject to `x = z`, given each piece's proximal
    operator (`prox_f(v, t)` returns `argmin_x f_obj(x) + ||x-v||^2/(2t)`, likewise
    `prox_g`) rather than a gradient — the whole appeal of ADMM is that neither `f_obj`
    nor `g_obj` needs to be differentiable, so there's no autodiff fallback to offer the
    way `optimlab.core.Problem` has one. `f_obj`/`g_obj` themselves are optional, used
    only to report `OptimizeResult.f_trajectory`.
    """

    prox_f: Callable[[ArrayLike, float], ArrayLike]
    prox_g: Callable[[ArrayLike, float], ArrayLike]
    x0: ArrayLike
    f_obj: Objective | None = None
    g_obj: Objective | None = None
    name: str = "admm_problem"

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)

    @property
    def n_dim(self) -> int:
        return int(self.x0.size)

    def f(self, x: ArrayLike) -> float:
        total = 0.0
        if self.f_obj is not None:
            total += float(self.f_obj(x))
        if self.g_obj is not None:
            total += float(self.g_obj(x))
        return total


def admm(
    problem: ADMMProblem,
    *,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> OptimizeResult:
    """`rho` trades off how strictly `x = z` is enforced at each step against how far
    each proximal step is allowed to move — unlike proximal gradient's step size, `rho`
    doesn't need to respect any Lipschitz bound (ADMM converges for any `rho > 0`), so
    it's mostly a practical convergence-speed tuning knob rather than a stability
    requirement. Convergence is checked via the **primal residual**
    `||x_{k+1} - z_{k+1}||` — the two blocks agreeing is exactly the constraint `x = z`
    being satisfied.
    """
    x = problem.x0.copy()
    z = problem.x0.copy()
    u = np.zeros_like(x)

    x_hist, f_hist, g_hist = [x.copy()], [problem.f(x)], [float(np.linalg.norm(x - z))]

    # Unlike every other from-scratch solver here, the *initial* residual (x0 - z0) is
    # always exactly zero by construction (both start at problem.x0) -- it says nothing
    # about whether the problem is solved, so, unlike e.g. gradient_descent checking
    # grad_norm before its first step, ADMM must always run at least one iteration.
    converged = False
    n_iter = 0
    while not converged and n_iter < max_iter:
        x = problem.prox_f(z - u, 1.0 / rho)
        z = problem.prox_g(x + u, 1.0 / rho)
        u = u + x - z

        n_iter += 1
        primal_residual = float(np.linalg.norm(x - z))
        x_hist.append(x.copy())
        f_hist.append(problem.f(x))
        g_hist.append(primal_residual)

        if primal_residual < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="admm",
        message="primal residual below tol" if converged else "max_iter reached",
    )
