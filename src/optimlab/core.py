"""Core abstractions shared by every algorithm and every problem in optimlab.

A :class:`Problem` describes *what* to minimize. A solver (any callable with the
signature ``solver(problem, **kwargs) -> OptimizeResult``, from-scratch or a backend
adapter) describes *how*. Because both sides only ever touch this interface, any solver
in the repo can run against any problem in the repo — that's the whole "port a problem
in, get solvers for free" mechanism.

Gradients and Hessians are supplied by the problem when known in closed form (worth
doing by hand at least once, for the pedagogy), and computed automatically via JAX
otherwise, falling back to finite differences if JAX isn't available. Every solver
therefore always has a `grad` and `hess` to call, regardless of what the problem
author provided.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

try:
    # JAX defaults to float32, which is nowhere near precise enough for the tight
    # convergence tolerances and Hessian eigenvalue work this repo cares about — must be
    # set before any JAX arrays are created, so it happens at import time here.
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

ArrayLike = np.ndarray
Objective = Callable[[ArrayLike], float]
GradFn = Callable[[ArrayLike], ArrayLike]
HessFn = Callable[[ArrayLike], ArrayLike]


@dataclass
class Problem:
    """An optimization problem: minimize ``f(x)`` over ``x``.

    Constraints are handled by dedicated `Problem` subclasses / fields once Phase 2–4
    solvers (LP, QP, KKT-based) land; for now this covers the unconstrained problems
    that Chapters 1–2's gradient-based methods target.

    Parameters
    ----------
    f: the objective, ``x -> scalar``. Must accept a 1D numpy array (or, if you want
        JAX-traced autodiff, be written with ``jax.numpy`` ops so it also accepts a
        JAX array — plain ``numpy``/Python arithmetic works for both).
    x0: default starting point.
    grad, hess: closed-form gradient/Hessian, if you have (or want to derive) them.
        Left as ``None`` to fall back to JAX autodiff / finite differences.
    name: human-readable identifier used in reports and plot legends.
    minimum, f_min: known global minimizer / minimum value, for benchmark problems
        where the answer is known — lets tests and plots mark "how close did we get."
    domain: a ``(low, high)`` box used by plotting code to pick a default viewing window.
    reference: a short citation / URL for where this problem comes from.
    """

    f: Objective
    x0: ArrayLike
    grad: GradFn | None = None
    hess: HessFn | None = None
    name: str = "problem"
    minimum: ArrayLike | None = None
    f_min: float | None = None
    domain: tuple[float, float] | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.grad is None:
            self.grad = _autograd(self.f)
        if self.hess is None:
            self.hess = _autohess(self.f)

    @property
    def n_dim(self) -> int:
        return int(self.x0.size)


def _autograd(f: Objective) -> GradFn:
    try:
        import jax
        import jax.numpy as jnp

        jax_grad = jax.jit(jax.grad(lambda x: f(x)))

        def grad(x: ArrayLike) -> ArrayLike:
            return np.asarray(jax_grad(jnp.asarray(x, dtype=jnp.float64)))

        return grad
    except ImportError:
        return lambda x: _finite_diff_grad(f, x)


def _autohess(f: Objective) -> HessFn:
    try:
        import jax
        import jax.numpy as jnp

        jax_hess = jax.jit(jax.hessian(lambda x: f(x)))

        def hess(x: ArrayLike) -> ArrayLike:
            return np.asarray(jax_hess(jnp.asarray(x, dtype=jnp.float64)))

        return hess
    except ImportError:
        return lambda x: _finite_diff_hess(f, x)


def _finite_diff_grad(f: Objective, x: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        g[i] = (f(x + step) - f(x - step)) / (2 * eps)
    return g


def _finite_diff_hess(f: Objective, x: ArrayLike, eps: float = 1e-4) -> ArrayLike:
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros_like(x), np.zeros_like(x)
            ei[i], ej[j] = eps, eps
            val = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4 * eps**2)
            H[i, j] = H[j, i] = val
    return H


@dataclass
class OptimizeResult:
    """Standardized solver output — the common currency for comparing solvers/problems."""

    x: ArrayLike
    f: float
    n_iter: int
    converged: bool
    solver_name: str = ""
    message: str = ""
    wall_time: float = 0.0
    trajectory: list[ArrayLike] = field(default_factory=list)
    f_trajectory: list[float] = field(default_factory=list)
    grad_norm_trajectory: list[float] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "converged" if self.converged else "did NOT converge"
        return (
            f"OptimizeResult({self.solver_name}, {status} in {self.n_iter} iters, "
            f"f={self.f:.6g}, wall_time={self.wall_time * 1000:.2f}ms)"
        )


class Solver(Protocol):
    """The call signature every optimizer in optimlab (from-scratch or backend) implements."""

    def __call__(self, problem: Problem, **kwargs: Any) -> OptimizeResult: ...


def run(solver: Callable[..., OptimizeResult], problem: Problem, **kwargs: Any) -> OptimizeResult:
    """Run ``solver`` on ``problem``, stamping wall-clock time onto the result.

    Thin wrapper rather than a required entry point — every solver already returns a
    valid `OptimizeResult` on its own; this just standardizes timing so results from
    different solvers are fairly comparable in the solver-arena reports (Phase 8).
    """
    start = time.perf_counter()
    result = solver(problem, **kwargs)
    result.wall_time = time.perf_counter() - start
    return result


def track_iterations(
    x_hist: list[ArrayLike],
    f_hist: list[float],
    g_hist: list[float],
    *,
    converged: bool,
    solver_name: str,
    message: str = "",
) -> OptimizeResult:
    """Package per-iteration history lists (as every from-scratch solver accumulates them)
    into a final `OptimizeResult`. Small, but every optimizer needs it, so it lives once
    here instead of being re-written slightly differently in each `optimizers/*.py` file.
    """
    return OptimizeResult(
        x=x_hist[-1],
        f=f_hist[-1],
        n_iter=len(f_hist) - 1,
        converged=converged,
        solver_name=solver_name,
        message=message,
        trajectory=x_hist,
        f_trajectory=f_hist,
        grad_norm_trajectory=g_hist,
    )
