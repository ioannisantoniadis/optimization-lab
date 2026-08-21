"""The barrier (interior point) method (book §6.4): solve a convex inequality-constrained
problem

    minimize    f(x)
    subject to  g_i(x) <= 0,  i = 1..m

by replacing each constraint with a logarithmic penalty that blows up as `x` approaches
the constraint boundary, then solving a *sequence* of unconstrained problems with that
penalty's weight driven toward zero:

    minimize    t * f(x) - sum_i log(-g_i(x))

For any fixed `t`, this is smooth and unconstrained — solvable with the exact
`optimlab.optimizers.newton` machinery this repo already has. As `t -> infinity`, the
log term's influence vanishes and the minimizer converges to the true constrained
optimum. The sequence of minimizers traced out along the way is the **central path** —
`optimlab.viz.constrained.central_path_figure` draws it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import (
    ArrayLike,
    GradFn,
    HessFn,
    Objective,
    OptimizeResult,
    _autograd,
    _autohess,
    track_iterations,
)


@dataclass
class ConstrainedProblem:
    """Minimize `f(x)` subject to `inequality_constraints[i](x) <= 0` for every `i`.
    `x0` **must** be strictly feasible (`g_i(x0) < 0` for every `i`) — the barrier
    `-log(-g_i(x))` is undefined the moment a constraint is only just satisfied, let
    alone violated, so there's no "start anywhere and get pulled in" the way an
    unconstrained `Problem` allows. `grad`/`hess` (of `f`) and each constraint's own
    gradient/Hessian are all filled in via JAX automatic differentiation when left as
    `None`, exactly like `optimlab.core.Problem`.
    """

    f: Objective
    x0: ArrayLike
    inequality_constraints: list[Objective]
    grad: GradFn | None = None
    hess: HessFn | None = None
    name: str = "constrained_problem"

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.grad is None:
            self.grad = _autograd(self.f)
        if self.hess is None:
            self.hess = _autohess(self.f)
        self._g_grad = [_autograd(g) for g in self.inequality_constraints]
        self._g_hess = [_autohess(g) for g in self.inequality_constraints]

        infeasible = [i for i, g in enumerate(self.inequality_constraints) if g(self.x0) >= 0]
        if infeasible:
            raise ValueError(
                f"x0 must be strictly feasible for every constraint; "
                f"constraint(s) {infeasible} are violated or exactly active at x0"
            )

    @property
    def n_dim(self) -> int:
        return int(self.x0.size)

    @property
    def n_constraints(self) -> int:
        return len(self.inequality_constraints)

    def is_strictly_feasible(self, x: ArrayLike) -> bool:
        return all(float(g(x)) < 0 for g in self.inequality_constraints)

    def barrier_grad_hess(self, x: ArrayLike, t: float) -> tuple[ArrayLike, ArrayLike]:
        """`grad`/`hess` of `t*f(x) - sum_i log(-g_i(x))` at `x` — the standard
        log-barrier formulas, derived once in the module docstring's neighborhood: for
        `s_i = -g_i(x) > 0`, `grad[-log(s_i)] = grad_g_i / s_i` and
        `hess[-log(s_i)] = hess_g_i / s_i + outer(grad_g_i, grad_g_i) / s_i^2`.
        """
        grad = t * self.grad(x)
        hess = t * self.hess(x)
        for g, g_grad, g_hess in zip(self.inequality_constraints, self._g_grad, self._g_hess, strict=True):
            s = -float(g(x))
            gi_grad = g_grad(x)
            grad = grad + gi_grad / s
            hess = hess + g_hess(x) / s + np.outer(gi_grad, gi_grad) / s**2
        return grad, hess


def _feasible_backtracking(
    problem: ConstrainedProblem, phi: callable, x: ArrayLike, direction: ArrayLike,
    phi_x: float, grad_phi_x: ArrayLike, *, c1: float = 1e-4, rho: float = 0.5, max_iter: int = 60,
) -> float:
    """Ordinary Armijo backtracking (`optimlab.optimizers.line_search`), plus the one
    thing a barrier method needs that an unconstrained line search doesn't: a trial step
    is rejected outright, before even checking Armijo, unless it lands strictly inside
    every constraint — the barrier value isn't just large near the boundary, it's
    undefined past it.
    """
    directional_deriv = grad_phi_x @ direction
    alpha = 1.0
    for _ in range(max_iter):
        x_new = x + alpha * direction
        if problem.is_strictly_feasible(x_new) and phi(x_new) <= phi_x + c1 * alpha * directional_deriv:
            return alpha
        alpha *= rho
    return alpha


def barrier_method(
    problem: ConstrainedProblem,
    *,
    t0: float = 1.0,
    mu: float = 10.0,
    tol: float = 1e-8,
    newton_tol: float = 1e-10,
    max_outer: int = 50,
    max_newton: int = 50,
) -> OptimizeResult:
    """Standard two-loop barrier method (Boyd & Vandenberghe, Algorithm 11.1): for each
    `t` in a geometric sequence (`t *= mu` every outer step), run damped Newton on the
    barrier-augmented objective until it's converged, then increase `t` and continue
    from there (a warm start — each Newton solve starts very close to its answer,
    because the previous `t`'s optimum is a good guess for the next). Stops once
    `n_constraints / t < tol`: a standard duality-gap estimate for the log barrier that
    upper-bounds how far the current point is from the true optimum. `OptimizeResult
    .trajectory` holds one point per **outer** step — the central path — not one per
    Newton step, since the inner Newton iterations are an implementation detail of
    solving a single point on that path, not new information about the constrained
    problem itself.
    """
    x = problem.x0.copy()
    t = t0
    x_hist = [x.copy()]
    f_hist = [float(problem.f(x))]
    gap_hist = [problem.n_constraints / t]

    converged = False
    outer = 0
    while not converged and outer < max_outer:

        def phi(point: ArrayLike, t: float = t) -> float:
            barrier = sum(-np.log(-g(point)) for g in problem.inequality_constraints)
            return t * float(problem.f(point)) + float(barrier)

        for _ in range(max_newton):
            grad_phi_x, hess_phi_x = problem.barrier_grad_hess(x, t)
            if np.linalg.norm(grad_phi_x) < newton_tol:
                break
            direction = np.linalg.solve(hess_phi_x, -grad_phi_x)
            alpha = _feasible_backtracking(problem, phi, x, direction, phi(x), grad_phi_x)
            x = x + alpha * direction

        outer += 1
        x_hist.append(x.copy())
        f_hist.append(float(problem.f(x)))
        duality_gap = problem.n_constraints / t
        gap_hist.append(duality_gap)

        if duality_gap < tol:
            converged = True
        else:
            t *= mu

    return track_iterations(
        x_hist, f_hist, gap_hist, converged=converged, solver_name="barrier_method",
        message="duality gap below tol" if converged else "max_outer reached",
    )
