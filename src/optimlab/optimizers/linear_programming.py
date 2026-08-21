"""Linear programming (book §3): Dantzig's simplex method, implemented from scratch as a
two-phase full-tableau simplex with Bland's rule (guarantees no cycling, at the cost of
being slower than Dantzig's most-negative-reduced-cost rule — a fine trade for an
educational implementation over small problems).

Unlike `optimlab.core.Problem` (smooth `f`, gradients, local search), a linear program
has no gradient to follow — the optimum, if one exists, sits at a *vertex* of the
feasible polytope (a direct consequence of a linear objective over a convex polytope:
see `optimlab.viz.polytope` for that fact made visible). Simplex is a walk from vertex
to adjacent vertex, always improving the objective, until no adjacent vertex is better.
That's why `LinearProgram`/`LPResult` are their own small types here rather than reusing
`Problem`/`OptimizeResult` — the two problem classes are solved by fundamentally
different kinds of algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimlab.core import ArrayLike


@dataclass
class LinearProgram:
    """A linear program in general form:

        minimize    c^T x
        subject to  A_eq x = b_eq
                    A_ub x <= b_ub
                    x >= 0

    `A_ub`/`b_ub` and/or `A_eq`/`b_eq` may be left as `None` if not needed. Variables
    are assumed nonnegative — the standard-form convention simplex needs; shift a free
    variable to `x = x+ - x-` (two nonnegative variables) before constructing one of
    these if your problem has one.
    """

    c: ArrayLike
    A_ub: ArrayLike | None = None
    b_ub: ArrayLike | None = None
    A_eq: ArrayLike | None = None
    b_eq: ArrayLike | None = None
    name: str = "lp"

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=float)
        if self.A_ub is not None:
            self.A_ub = np.atleast_2d(np.asarray(self.A_ub, dtype=float))
            self.b_ub = np.asarray(self.b_ub, dtype=float)
        if self.A_eq is not None:
            self.A_eq = np.atleast_2d(np.asarray(self.A_eq, dtype=float))
            self.b_eq = np.asarray(self.b_eq, dtype=float)

    @property
    def n_vars(self) -> int:
        return int(self.c.size)


@dataclass
class LPResult:
    """Simplex's output. `vertices` is every basic feasible solution visited (the
    starting one and one per pivot) — the sequence `optimlab.viz.polytope` draws as the
    "walk across the polytope" path.
    """

    x: ArrayLike
    objective: float
    status: str  # "optimal", "infeasible", "unbounded", "max_iter_reached"
    n_iter: int
    vertices: list[ArrayLike] = field(default_factory=list)
    solver_name: str = "simplex"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LPResult({self.solver_name}, status={self.status!r}, "
            f"objective={self.objective:.6g}, n_iter={self.n_iter})"
        )


_TOL = 1e-9


def _price_out(tableau: np.ndarray, obj_row: np.ndarray, cost: np.ndarray, basis: list[int]) -> None:
    """Zero out `obj_row` at every basic column by subtracting `cost[basis[i]] * row i`
    — turns a plain `[cost | 0]` row into the reduced-cost row for the *current* basis,
    which every simplex step after the first needs re-derived from scratch here because
    `cost` changes between phase 1 (all-artificial) and phase 2 (the real objective).
    """
    obj_row[:] = np.concatenate([cost, [0.0]])
    for i, basic_var in enumerate(basis):
        coeff = obj_row[basic_var]
        if abs(coeff) > _TOL:
            obj_row -= coeff * tableau[i]


def _pivot(tableau: np.ndarray, obj_row: np.ndarray, basis: list[int], row: int, col: int) -> None:
    tableau[row] /= tableau[row, col]
    for i in range(tableau.shape[0]):
        if i != row and abs(tableau[i, col]) > _TOL:
            tableau[i] -= tableau[i, col] * tableau[row]
    if abs(obj_row[col]) > _TOL:
        obj_row -= obj_row[col] * tableau[row]
    basis[row] = col


def _run_simplex(
    tableau: np.ndarray,
    obj_row: np.ndarray,
    basis: list[int],
    max_iter: int,
    on_pivot=None,
) -> tuple[str, int]:
    """Bland's-rule simplex loop on an already-canonical tableau (basic columns are the
    identity, `obj_row` already priced out). Returns (status, n_iter); status is
    "optimal", "unbounded", or "max_iter_reached". `on_pivot`, if given, is called with
    no arguments after every successful pivot — `simplex()` uses it to record the vertex
    (basic feasible solution) each pivot moves to, for `optimlab.viz.polytope`.
    """
    n_iter = 0
    while n_iter < max_iter:
        # Bland's rule: enter the *smallest-indexed* column with a negative reduced
        # cost, not the most-negative one — slower, but provably never cycles.
        candidates = np.where(obj_row[:-1] < -_TOL)[0]
        if candidates.size == 0:
            return "optimal", n_iter
        entering = int(candidates[0])

        column = tableau[:, entering]
        eligible = np.where(column > _TOL)[0]
        if eligible.size == 0:
            return "unbounded", n_iter

        ratios = tableau[eligible, -1] / column[eligible]
        best = np.isclose(ratios, ratios.min(), atol=_TOL)
        # Tie-break (degenerate ratio) by smallest basic-variable index — Bland's rule
        # again, on the *leaving* side this time — same no-cycling guarantee.
        tied_rows = eligible[best]
        leaving = int(tied_rows[np.argmin([basis[r] for r in tied_rows])])

        _pivot(tableau, obj_row, basis, leaving, entering)
        n_iter += 1
        if on_pivot is not None:
            on_pivot()
    return "max_iter_reached", n_iter


def simplex(lp: LinearProgram, *, max_iter: int = 200) -> LPResult:
    """Solve `lp` with the two-phase simplex method.

    Phase 1 finds a starting basic feasible solution by minimizing the sum of
    artificial variables (added only to constraint rows that don't already have an
    obvious one, i.e. `<=` rows with nonnegative RHS contribute their own slack).
    Phase 2 then optimizes the real objective from that starting vertex. Both phases
    are the exact same tableau-pivoting loop (`_run_simplex`) with a different cost row.
    """
    n = lp.n_vars
    rows_A, rows_b, n_ub = [], [], 0
    if lp.A_ub is not None:
        rows_A.append(lp.A_ub)
        rows_b.append(lp.b_ub)
        n_ub = lp.A_ub.shape[0]
    if lp.A_eq is not None:
        rows_A.append(lp.A_eq)
        rows_b.append(lp.b_eq)

    A = np.vstack(rows_A) if rows_A else np.zeros((0, n))
    b = np.concatenate(rows_b) if rows_b else np.zeros(0)
    m = A.shape[0]

    if m == 0:
        # No constraints at all beyond x >= 0: minimize c^T x over the nonnegative
        # orthant — unbounded unless every cost is already nonnegative, in which case
        # x=0 is optimal. Simplex proper needs at least one row, so handle directly.
        status = "optimal" if np.all(lp.c >= -_TOL) else "unbounded"
        x0 = np.zeros(n)
        return LPResult(x=x0, objective=float(lp.c @ x0), status=status, n_iter=0, vertices=[x0])

    slack_block = np.zeros((m, n_ub))
    slack_block[:n_ub, :n_ub] = np.eye(n_ub)
    A_std = np.hstack([A, slack_block])
    b_std = b.copy()
    c_std = np.concatenate([lp.c, np.zeros(n_ub)])

    # Normalize every row to a nonnegative RHS — required for the "slack column is
    # already a valid basic column" trick below, and for phase 1's b >= 0 assumption.
    flipped = b_std < 0
    A_std[flipped] *= -1
    b_std[flipped] *= -1

    # A row already has a ready-made basic column iff it's an (unflipped) `<=` row: its
    # slack column is then a clean +1 in that row and 0 elsewhere. Every other row
    # (equality rows, and any `<=` row whose sign got flipped) needs an artificial
    # variable to seed phase 1.
    ready = np.zeros(m, dtype=bool)
    ready[:n_ub] = ~flipped[:n_ub]

    n_slack = n_ub
    artificial_cols = []
    basis: list[int] = [0] * m
    for i in range(m):
        if ready[i]:
            basis[i] = n + i  # that row's own slack column
        else:
            col = np.zeros(m)
            col[i] = 1.0
            artificial_cols.append(col)
            basis[i] = n + n_slack + len(artificial_cols) - 1

    total_vars = n + n_slack + len(artificial_cols)
    if artificial_cols:
        A_std = np.hstack([A_std, np.column_stack(artificial_cols)])
    tableau = np.hstack([A_std, b_std.reshape(-1, 1)])

    vertices: list[ArrayLike] = []

    def current_x() -> ArrayLike:
        x_full = np.zeros(total_vars)
        for i, basic_var in enumerate(basis):
            # A basic_var can briefly point past the current tableau width right after
            # artificial columns are dropped, for the rare row left with a redundant
            # (all-zero) constraint whose artificial never got pivoted out — it
            # contributes nothing to any real variable, so it's safe to skip.
            if basic_var < x_full.size:
                x_full[basic_var] = tableau[i, -1]
        return x_full[:n]

    # `vertices` records only genuine vertices of the *original* feasible polytope:
    # phase 1's intermediate pivots move through an auxiliary problem (with artificial
    # variables soaking up infeasibility) and are generally infeasible for the real
    # problem until phase 1's very last step, so they're deliberately not recorded —
    # only the first true feasible vertex phase 1 lands on, plus one per phase-2 pivot.
    if artificial_cols:
        art_start = n + n_slack
        cost_phase1 = np.zeros(total_vars)
        cost_phase1[art_start:] = 1.0
        obj_row = np.zeros(total_vars + 1)
        _price_out(tableau, obj_row, cost_phase1, basis)

        status, iters_p1 = _run_simplex(tableau, obj_row, basis, max_iter)
        if status != "optimal" or obj_row[-1] < -1e-6:
            # Phase 1's *minimum* achievable sum of artificials is > 0: no point in the
            # original feasible region reaches it, so the LP itself is infeasible.
            return LPResult(
                x=np.full(n, np.nan), objective=float("nan"), status="infeasible",
                n_iter=iters_p1, vertices=vertices, solver_name="simplex",
            )

        # Any artificial left in the basis is necessarily at value 0 (phase 1 optimum
        # is 0) — drive it out by pivoting on any non-artificial column with a nonzero
        # entry in its row; if none exists, that row is a redundant constraint and is
        # simply left alone (it no longer affects phase 2's answer).
        for i, basic_var in enumerate(basis):
            if basic_var >= art_start:
                candidates = np.where(np.abs(tableau[i, :art_start]) > _TOL)[0]
                if candidates.size > 0:
                    _pivot(tableau, obj_row, basis, i, int(candidates[0]))

        rhs_col = tableau[:, -1:]
        tableau = np.hstack([tableau[:, :art_start], rhs_col])
        total_vars = art_start
    else:
        iters_p1 = 0

    vertices.append(current_x())  # the starting vertex phase 2 improves from
    obj_row = np.zeros(total_vars + 1)
    _price_out(tableau, obj_row, c_std, basis)
    status, iters_p2 = _run_simplex(
        tableau, obj_row, basis, max_iter, on_pivot=lambda: vertices.append(current_x())
    )

    x = current_x()
    return LPResult(
        x=x, objective=float(lp.c @ x), status=status,
        n_iter=iters_p1 + iters_p2, vertices=vertices, solver_name="simplex",
    )
