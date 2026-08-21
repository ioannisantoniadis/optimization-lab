"""scipy.optimize adapters — correctness oracles for optimlab's from-scratch solvers
(and, later, scale-up paths for problems too large for the from-scratch versions to
handle comfortably). scipy is already a core dependency, so these need no extra install.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares as _scipy_least_squares
from scipy.optimize import linprog as _scipy_linprog

from optimlab.core import OptimizeResult
from optimlab.optimizers.gauss_newton import NonlinearLeastSquaresProblem
from optimlab.optimizers.linear_programming import LinearProgram, LPResult

_STATUS_MAP = {0: "optimal", 1: "max_iter_reached", 2: "infeasible", 3: "unbounded", 4: "max_iter_reached"}


def scipy_linprog(lp: LinearProgram) -> LPResult:
    """Solve `lp` with scipy's HiGHS-backed `linprog` — the correctness oracle
    `optimlab.optimizers.linear_programming.simplex` is tested against (see
    `tests/test_linear_programming.py`). HiGHS doesn't expose its internal pivot path
    the way our from-scratch tableau simplex does, so `vertices` here is always a
    single-point list — this backend is for checking *answers*, not for
    `optimlab.viz.polytope`'s "walk across the polytope" visualization.
    """
    result = _scipy_linprog(
        lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, A_eq=lp.A_eq, b_eq=lp.b_eq,
        bounds=[(0, None)] * lp.n_vars, method="highs",
    )
    status = _STATUS_MAP.get(result.status, "max_iter_reached")
    x = np.asarray(result.x) if result.x is not None else np.full(lp.n_vars, np.nan)
    objective = float(result.fun) if result.fun is not None else float("nan")
    return LPResult(
        x=x, objective=objective, status=status,
        n_iter=int(getattr(result, "nit", 0) or 0), vertices=[x], solver_name="scipy_linprog",
    )


def scipy_nonlinear_least_squares(problem: NonlinearLeastSquaresProblem) -> OptimizeResult:
    """Solve `problem` with scipy's trust-region-reflective `least_squares` — the
    correctness oracle `optimlab.optimizers.gauss_newton.gauss_newton` is tested
    against. scipy doesn't expose a per-iteration trajectory, so `trajectory` /
    `f_trajectory` here hold only the start and end points, not one entry per step.
    """
    def residual_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(problem.residual(x))

    def jacobian_np(x: np.ndarray) -> np.ndarray:
        return np.asarray(problem.jacobian(x))

    start_time = time.perf_counter()
    result = _scipy_least_squares(residual_np, problem.x0, jac=jacobian_np)
    wall_time = time.perf_counter() - start_time

    f_start = problem.f(problem.x0)
    f_end = float(0.5 * result.fun @ result.fun)
    g_start = float(np.linalg.norm(jacobian_np(problem.x0).T @ residual_np(problem.x0)))
    g_end = float(np.linalg.norm(result.grad))

    return OptimizeResult(
        x=result.x, f=f_end, n_iter=int(result.nfev), converged=bool(result.success),
        solver_name="scipy_least_squares", message=result.message, wall_time=wall_time,
        trajectory=[problem.x0, result.x], f_trajectory=[f_start, f_end],
        grad_norm_trajectory=[g_start, g_end],
    )
