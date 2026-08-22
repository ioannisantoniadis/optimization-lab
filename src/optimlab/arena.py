"""The solver arena: register a `Problem`, get a standardized report across every
applicable solver — the payoff every solver since Phase 1 sharing the identical
`Problem -> OptimizeResult` interface has been building toward, made literal rather
than left as an implicit property of the interface. "Port a new problem in, get
solvers for free" is this module's whole reason to exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from optimlab.core import OptimizeResult, Problem, Solver
from optimlab.optimizers import ALL_SOLVERS


@dataclass
class ArenaEntry:
    name: str
    result: OptimizeResult | None
    wall_time: float
    error: str | None = None


@dataclass
class ArenaReport:
    problem_name: str
    entries: list[ArenaEntry]

    def ranked_by_objective(self) -> list[ArenaEntry]:
        """Successful entries only, best (lowest) final objective first — a solver
        that raised an exception has nothing to rank.
        """
        successful = [e for e in self.entries if e.result is not None]
        return sorted(successful, key=lambda e: e.result.f)

    def summary_rows(self) -> list[dict]:
        """One dict per solver: `name`, `f` (`None` on failure), `n_iter`, `wall_time`,
        `converged`, `error` (`None` on success) — the flat, table-friendly shape both
        the arena's own printed summary and `optimlab.viz.arena_figure` build on.
        """
        rows = []
        for e in self.entries:
            if e.result is None:
                rows.append(
                    {"name": e.name, "f": None, "n_iter": None, "wall_time": e.wall_time,
                     "converged": False, "error": e.error}
                )
            else:
                rows.append(
                    {"name": e.name, "f": e.result.f, "n_iter": e.result.n_iter, "wall_time": e.wall_time,
                     "converged": e.result.converged, "error": None}
                )
        return rows


def run_arena(
    problem: Problem,
    *,
    solvers: dict[str, Solver] | None = None,
    solver_kwargs: dict[str, dict] | None = None,
) -> ArenaReport:
    """Run every solver in `solvers` (default: `optimlab.optimizers.ALL_SOLVERS`, the
    same registry `race_figure`/`convergence_figure` draw from) against `problem`,
    catching any exception a solver raises rather than letting one inapplicable solver
    (a gradient-free population method needing `bounds`/`problem.domain` the `Problem`
    doesn't set, say) stop the rest from running. `solver_kwargs` overrides the
    defaults for specific solvers by name, e.g. `{"adam": {"lr": 0.1}}`; every solver
    not mentioned runs with its own plain defaults — deliberately "out of the box"
    unless told otherwise, since that's the honest first thing to see for a new
    problem.
    """
    solvers = solvers if solvers is not None else ALL_SOLVERS
    solver_kwargs = solver_kwargs or {}

    entries = []
    for name, solver in solvers.items():
        kwargs = solver_kwargs.get(name, {})
        start = time.perf_counter()
        try:
            result = solver(problem, **kwargs)
            entries.append(ArenaEntry(name=name, result=result, wall_time=time.perf_counter() - start))
        except Exception as exc:  # noqa: BLE001 - a solver's own failure mode is exactly what's being reported here
            entries.append(
                ArenaEntry(
                    name=name, result=None, wall_time=time.perf_counter() - start,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return ArenaReport(problem_name=problem.name, entries=entries)
