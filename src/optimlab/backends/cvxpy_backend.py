"""cvxpy adapters — a second, independent correctness oracle alongside
`optimlab.backends.scipy_backend` for the convex problems this repo solves from scratch
(LP, QP). Requires the `backends` extra (`uv sync --extra backends`): cvxpy pulls in its
own bundled solvers (Clarabel, OSQP, SCS) rather than reusing scipy's.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimlab.core import ArrayLike
from optimlab.optimizers.linear_programming import LinearProgram, LPResult

_STATUS_MAP = {
    cp.OPTIMAL: "optimal",
    cp.INFEASIBLE: "infeasible",
    cp.INFEASIBLE_INACCURATE: "infeasible",
    cp.UNBOUNDED: "unbounded",
    cp.UNBOUNDED_INACCURATE: "unbounded",
}


def cvxpy_linprog(lp: LinearProgram) -> LPResult:
    """Solve `lp` with cvxpy — a second oracle alongside `scipy_backend.scipy_linprog`,
    useful precisely because it goes through an entirely different solver stack
    (cvxpy's own problem reduction + Clarabel, rather than HiGHS), so the two backends
    agreeing is stronger evidence than either alone.
    """
    x = cp.Variable(lp.n_vars)
    constraints = [x >= 0]
    if lp.A_ub is not None:
        constraints.append(lp.A_ub @ x <= lp.b_ub)
    if lp.A_eq is not None:
        constraints.append(lp.A_eq @ x == lp.b_eq)

    problem = cp.Problem(cp.Minimize(lp.c @ x), constraints)
    problem.solve()

    status = _STATUS_MAP.get(problem.status, "max_iter_reached")
    x_val = np.asarray(x.value) if x.value is not None else np.full(lp.n_vars, np.nan)
    objective = float(problem.value) if problem.value is not None else float("nan")
    return LPResult(
        x=x_val, objective=objective, status=status,
        n_iter=0, vertices=[x_val], solver_name="cvxpy_linprog",
    )


def cvxpy_qp(
    P: ArrayLike,
    q: ArrayLike,
    *,
    A_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    A_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
) -> ArrayLike:
    """Solve `min_x 0.5 x^T P x + q^T x` subject to any mix of `A_eq x = b_eq` and
    `A_ub x <= b_ub` — the general quadratic program neither
    `optimlab.linalg.qp.equality_constrained_qp` (equalities only) nor
    `optimlab.optimizers.projected_gradient` (box inequalities only) covers on its own.
    Used as their correctness oracle on the cases they *do* cover, and as the
    reach-for-this option once a problem needs both constraint types together.
    """
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    n = P.shape[0]
    x = cp.Variable(n)

    constraints = []
    if A_eq is not None:
        constraints.append(np.asarray(A_eq, dtype=float) @ x == np.asarray(b_eq, dtype=float))
    if A_ub is not None:
        constraints.append(np.asarray(A_ub, dtype=float) @ x <= np.asarray(b_ub, dtype=float))

    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), constraints)
    problem.solve()
    return np.asarray(x.value)
