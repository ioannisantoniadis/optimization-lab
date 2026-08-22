import numpy as np

from optimlab.arena import run_arena
from optimlab.core import Problem
from optimlab.landscapes import get
from optimlab.optimizers import gradient_descent


def test_every_applicable_solver_succeeds_on_a_well_behaved_2d_problem():
    bf = get("himmelblau")
    problem = bf.problem(x0=np.array([-4.0, 4.0]))
    report = run_arena(problem)

    assert report.problem_name == problem.name
    rows = report.summary_rows()
    assert len(rows) == 13  # every solver in ALL_SOLVERS
    assert all(row["error"] is None for row in rows)
    # every solver should land close to one of Himmelblau's four global minima
    assert all(row["f"] < 1e-3 for row in rows)


def test_population_solvers_fail_gracefully_without_a_domain():
    """genetic_algorithm/particle_swarm need bounds (problem.domain, if unset) --
    the arena should catch that failure and report it, not let it crash the whole run.
    """
    problem = Problem(f=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, x0=np.array([0.0, 0.0]))
    report = run_arena(problem)

    rows = {row["name"]: row for row in report.summary_rows()}
    assert rows["genetic_algorithm"]["error"] is not None
    assert rows["particle_swarm"]["error"] is not None
    assert rows["bfgs"]["error"] is None
    assert rows["bfgs"]["f"] < 1e-6


def test_ranked_by_objective_excludes_failures_and_sorts_ascending():
    problem = Problem(f=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, x0=np.array([0.0, 0.0]))
    report = run_arena(problem)

    ranked = report.ranked_by_objective()
    assert all(e.result is not None for e in ranked)
    assert len(ranked) == 11  # 13 minus the 2 that need bounds
    fs = [e.result.f for e in ranked]
    assert fs == sorted(fs)


def test_solver_kwargs_override_only_the_named_solver():
    problem = Problem(f=lambda x: x[0] ** 2 + x[1] ** 2, x0=np.array([5.0, 5.0]))
    report = run_arena(
        problem, solvers={"gradient_descent": gradient_descent}, solver_kwargs={"gradient_descent": {"max_iter": 3}}
    )
    row = report.summary_rows()[0]
    assert row["n_iter"] <= 3
