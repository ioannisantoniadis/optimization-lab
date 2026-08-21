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
**primal residual**); they differ during the run because each is mid-way through its own
proximal step, not yet reconciled with the other's.

The primal residual alone is *not* a safe stopping criterion, though it might look like
one: it can hit (near-)zero long before the run has actually converged, particularly at
a large `rho` -- confirmed directly on this module's own LASSO test problem, where at
`rho=50` the primal residual transiently dips to ~1e-17 at iteration 5 while the
objective is still ~50% off the true minimum, then *rises* again before properly
decaying. `x` and `z` agreeing with each other says nothing about whether they've also
converged to the right value. The standard fix (Boyd et al. 2011, §3.3.1) also tracks
the **dual residual** `rho * ||z_{k+1} - z_k||` -- `z` barely moving between steps is
what actually signals the dual variable `u` has stopped needing to correct anything --
and requires both below `tol` before declaring convergence.
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
    requirement. Convergence requires *both* the primal residual `||x-z||` and the dual
    residual `rho * ||z_{k+1} - z_k||` below `tol` (see the module docstring for why the
    primal residual by itself isn't safe to stop on). `OptimizeResult
    .grad_norm_trajectory` holds `max(primal_residual, dual_residual)` at each step —
    the single conservative number driving that stopping decision.
    """
    x = problem.x0.copy()
    z = problem.x0.copy()
    u = np.zeros_like(x)

    x_hist, f_hist, g_hist = [x.copy()], [problem.f(x)], [0.0]

    # Unlike every other from-scratch solver here, the *initial* residual (x0 - z0) is
    # always exactly zero by construction (both start at problem.x0) -- it says nothing
    # about whether the problem is solved, so, unlike e.g. gradient_descent checking
    # grad_norm before its first step, ADMM must always run at least one iteration.
    converged = False
    n_iter = 0
    while not converged and n_iter < max_iter:
        x = problem.prox_f(z - u, 1.0 / rho)
        z_prev = z
        z = problem.prox_g(x + u, 1.0 / rho)
        u = u + x - z

        n_iter += 1
        primal_residual = float(np.linalg.norm(x - z))
        dual_residual = rho * float(np.linalg.norm(z - z_prev))
        x_hist.append(x.copy())
        f_hist.append(problem.f(x))
        g_hist.append(max(primal_residual, dual_residual))

        if primal_residual < tol and dual_residual < tol:
            converged = True

    return track_iterations(
        x_hist, f_hist, g_hist, converged=converged, solver_name="admm",
        message="primal and dual residuals below tol" if converged else "max_iter reached",
    )
