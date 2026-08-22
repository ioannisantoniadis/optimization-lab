"""Benchmark every solver in optimlab, with JAX JIT warmup accounted for, and write
the results to `benchmarks/BENCHMARKS.md`.

    uv run python benchmarks/run_benchmarks.py

Every timed problem/solver pair is built *once* and reused across a warmup call
(discarded) plus several timed repeats, reported as the median. This matters
specifically because of how `optimlab.core.Problem` derives `grad`/`hess`: each
`Problem` instance owns its own `jax.jit`-compiled closures (`_autograd`/`_autohess`,
and, where opted into, `_autof`/`_autoresidual`), compiled lazily on first call and
cached only on that instance's closures — not globally. Building a fresh `Problem` per
timed call would force retracing every time and mostly measure JIT compilation
overhead rather than steady-state solver cost; reusing one instance across repeats
measures what a real workload actually pays after warmup.

Three sections:
  A. The solver arena: every `ALL_SOLVERS` entry against every `optimlab.landscapes`
     benchmark function (2D, fixed default seed) -- a genuine apples-to-apples
     comparison, since every solver here shares the same `Problem` interface.
  B. Everything else with its own problem shape (`simplex`/`dual`, `conjugate_gradient`,
     `gauss_newton`, `projected_gradient`, `proximal_gradient`, `admm`,
     `barrier_method`) -- each timed on one representative, already-verified problem
     from this repo's own test suite or docs chapters. NOT cross-comparable to each
     other or to section A: different problem sizes, different problem shapes.
  C. The `jit_f` / `jit_residual` opt-in flags (added this session) measured directly:
     same problem, same solver, flag on vs off.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from optimlab.control.trajectory_optimization import simulate_pendulum
from optimlab.core import Problem
from optimlab.inverse.system_id import (
    oscillator_identification_problem,
    simulate_damped_oscillator,
)
from optimlab.landscapes import ALL_FUNCTIONS
from optimlab.optimizers import (
    ALL_SOLVERS,
    ADMMProblem,
    LinearProgram,
    NonlinearLeastSquaresProblem,
    admm,
    barrier_method,
    conjugate_gradient,
    dual,
    gauss_newton,
    projected_gradient,
    proximal_gradient,
    simplex,
    soft_threshold,
)
from optimlab.optimizers.proximal_gradient import CompositeProblem
from optimlab.problems.sociology import proportional_fairness_problem

WARMUP = 1
REPEATS = 7


@dataclass
class Timing:
    label: str
    median_ms: float
    min_ms: float
    max_ms: float
    f: float | str
    n_iter: int | str
    converged: bool | str
    error: str | None = None


def time_call(fn, *, warmup: int = WARMUP, repeats: int = REPEATS):
    for _ in range(warmup):
        fn()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, samples


def _extract(result) -> tuple[float, int, bool | str]:
    """`OptimizeResult` (`.f`/`.converged`) and `LPResult` (`.objective`/`.status`) name
    their fields differently -- `simplex`/`dual` are the only solvers in this file that
    return the latter.
    """
    if hasattr(result, "f"):
        return result.f, result.n_iter, result.converged
    return result.objective, result.n_iter, result.status


def summarize(label: str, fn) -> Timing:
    try:
        result, samples = time_call(fn)
    except Exception as exc:  # noqa: BLE001 -- a broken solver/problem pair is a result, not a crash
        return Timing(label, float("nan"), float("nan"), float("nan"), "-", "-", "-", f"{type(exc).__name__}: {exc}")
    f_value, n_iter, converged = _extract(result)
    return Timing(
        label=label,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        f=f_value,
        n_iter=n_iter,
        converged=converged,
    )


def fmt_f(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def render_table(rows: list[Timing]) -> str:
    lines = [
        "| solver | median ms | min ms | max ms | f | n_iter | converged |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.error:
            lines.append(f"| {r.label} | - | - | - | ERROR: {r.error} | - | - |")
        else:
            lines.append(
                f"| {r.label} | {r.median_ms:.3f} | {r.min_ms:.3f} | {r.max_ms:.3f} | "
                f"{fmt_f(r.f)} | {r.n_iter} | {r.converged} |"
            )
    return "\n".join(lines)


# --- Section A: the solver arena, every ALL_SOLVERS entry x every benchmark landscape ---


def benchmark_arena() -> dict[str, list[Timing]]:
    results = {}
    for name in sorted(ALL_FUNCTIONS):
        benchmark = ALL_FUNCTIONS[name]
        problem = benchmark.problem(n_dim=2)  # built once; reused across warmup + repeats
        rows = [
            summarize(solver_name, lambda solver=solver, problem=problem: solver(problem))
            for solver_name, solver in ALL_SOLVERS.items()
        ]
        rows.sort(key=lambda t: t.median_ms if not t.error else float("inf"))
        results[name] = rows
    return results


# --- Section B: solvers with their own problem shape, one representative problem each ---


def benchmark_other_solvers() -> list[Timing]:
    rows = []

    # LP: the textbook problem from tests/test_linear_programming.py, primal and dual.
    lp = LinearProgram(c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
    lp_dual = dual(lp)
    rows.append(summarize("simplex (primal LP, 2 vars / 3 rows)", lambda: simplex(lp)))
    rows.append(summarize("simplex (dual of the same LP)", lambda: simplex(lp_dual)))

    # Conjugate gradient: a 50x50 random SPD system (tests/test_conjugate_gradient.py's shape).
    rng_cg = np.random.default_rng(50)
    n_cg = 50
    M_cg = rng_cg.standard_normal((n_cg, n_cg))
    A_spd = M_cg @ M_cg.T + n_cg * np.eye(n_cg)
    x_true_cg = rng_cg.standard_normal(n_cg)
    b_cg = A_spd @ x_true_cg
    rows.append(
        summarize("conjugate_gradient (50x50 SPD system)", lambda: conjugate_gradient(A_spd, b_cg, tol=1e-10))
    )

    # Gauss-Newton: the shipped damped-oscillator system-ID problem (docs/chapters/08).
    true_omega, true_zeta = 2.5, 0.15
    x0_osc = np.array([1.0, 0.0])
    t_osc = np.linspace(0, 10, 40)
    true_trajectory = np.asarray(simulate_damped_oscillator(true_omega, true_zeta, x0_osc, t_osc))
    observed = true_trajectory + 0.02 * np.random.default_rng(0).standard_normal(true_trajectory.shape)
    sysid_problem = oscillator_identification_problem(t_osc, observed, x0_osc, params0=np.array([1.0, 0.5]))
    rows.append(summarize("gauss_newton (damped-oscillator system ID)", lambda: gauss_newton(sysid_problem, max_iter=30)))

    # Barrier method: the shipped 3-user/2-resource proportional-fairness problem
    # (docs/chapters/09).
    A_net = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    capacities = np.array([10.0, 10.0])
    fairness_problem = proportional_fairness_problem(A_net, capacities)
    rows.append(summarize("barrier_method (proportional fairness, 3 users)", lambda: barrier_method(fairness_problem)))

    # ADMM and proximal_gradient on the *identical* LASSO problem (rng seed 0, m=50,
    # n=20, alpha=1.0 -- tests/test_admm.py's and tests/test_proximal_gradient.py's
    # shared setup) so these two rows are directly comparable to each other, unlike
    # every other row in this section.
    rng_lasso = np.random.default_rng(0)
    m_lasso, n_lasso = 50, 20
    A_lasso = rng_lasso.standard_normal((m_lasso, n_lasso))
    x_true_lasso = np.zeros(n_lasso)
    x_true_lasso[[2, 5, 9]] = [3.0, -2.0, 1.5]
    b_lasso = A_lasso @ x_true_lasso + 0.01 * rng_lasso.standard_normal(m_lasso)
    alpha_lasso = 1.0

    AtA = A_lasso.T @ A_lasso
    Atb = A_lasso.T @ b_lasso

    def prox_f(v, t):
        r = 1.0 / t
        return np.linalg.solve(AtA + r * np.eye(n_lasso), Atb + r * v)

    def prox_g(v, t):
        return soft_threshold(v, alpha_lasso * t)

    admm_problem = ADMMProblem(
        prox_f=prox_f,
        prox_g=prox_g,
        x0=np.zeros(n_lasso),
        f_obj=lambda x: 0.5 * np.sum((A_lasso @ x - b_lasso) ** 2),
        g_obj=lambda x: alpha_lasso * np.sum(np.abs(x)),
    )
    rows.append(
        summarize(
            "admm (LASSO, m=50/n=20, shared setup with proximal_gradient below)",
            lambda: admm(admm_problem, rho=20.0, max_iter=2000, tol=1e-8),
        )
    )

    composite_problem = CompositeProblem(
        grad_smooth=lambda x: A_lasso.T @ (A_lasso @ x - b_lasso),
        prox_nonsmooth=lambda v, t: soft_threshold(v, alpha_lasso * t),
        x0=np.zeros(n_lasso),
        f_smooth=lambda x: 0.5 * np.sum((A_lasso @ x - b_lasso) ** 2),
        f_nonsmooth=lambda x: alpha_lasso * np.sum(np.abs(x)),
    )
    rows.append(
        summarize(
            "proximal_gradient (same LASSO problem as admm above)",
            lambda: proximal_gradient(composite_problem, max_iter=2000, tol=1e-8),
        )
    )

    # Projected gradient: the random 5D box-QP from tests/test_projected_gradient.py.
    rng_box = np.random.default_rng(0)
    n_box = 5
    M_box = rng_box.standard_normal((n_box, n_box))
    P_box = M_box @ M_box.T + n_box * np.eye(n_box)
    q_box = rng_box.uniform(-5, 5, size=n_box)

    def box_f(x):
        return 0.5 * x @ P_box @ x + q_box @ x

    def box_grad(x):
        return P_box @ x + q_box

    box_problem = Problem(f=box_f, x0=np.zeros(n_box), grad=box_grad)
    lr_box = 1.0 / np.linalg.eigvalsh(P_box).max()
    rows.append(
        summarize(
            "projected_gradient (5D random box QP)",
            lambda: projected_gradient(box_problem, lower=-1.0, upper=1.0, lr=lr_box, max_iter=5000),
        )
    )

    return rows


# --- Section C: jit_f / jit_residual, on vs off, same problem and solver otherwise ---


def benchmark_jit_flag() -> list[Timing]:
    rows = []

    # Pendulum swing-up (BFGS), at the docs chapter's actual budget: n_steps=20, dt=0.1.
    x0_pendulum = np.array([0.0, 0.0])
    x_target = np.array([np.pi, 0.0])
    n_steps, dt = 20, 0.1

    import jax.numpy as jnp

    from optimlab.optimizers.quasi_newton import bfgs

    x0_j = jnp.asarray(x0_pendulum, dtype=jnp.float64)
    x_target_j = jnp.asarray(x_target, dtype=jnp.float64)

    def swingup_cost(controls):
        trajectory = simulate_pendulum(x0_j, controls, dt)
        control_cost = 0.01 * jnp.sum(controls**2) * dt
        terminal_cost = 200.0 * jnp.sum((trajectory[-1] - x_target_j) ** 2)
        return control_cost + terminal_cost

    swing_problem_nojit = Problem(f=swingup_cost, x0=np.zeros(n_steps), jit_f=False, name="pendulum_swingup")
    swing_problem_jit = Problem(f=swingup_cost, x0=np.zeros(n_steps), jit_f=True, name="pendulum_swingup")
    rows.append(summarize("pendulum_swingup + bfgs, jit_f=False", lambda: bfgs(swing_problem_nojit, max_iter=50)))
    rows.append(summarize("pendulum_swingup + bfgs, jit_f=True", lambda: bfgs(swing_problem_jit, max_iter=50)))

    # Damped-oscillator system ID (Gauss-Newton), same problem, jit_residual on vs off.
    true_omega, true_zeta = 2.5, 0.15
    x0_osc = np.array([1.0, 0.0])
    t_osc = np.linspace(0, 10, 40)
    true_trajectory = np.asarray(simulate_damped_oscillator(true_omega, true_zeta, x0_osc, t_osc))
    observed = true_trajectory + 0.02 * np.random.default_rng(0).standard_normal(true_trajectory.shape)
    observed_j = jnp.asarray(observed)

    def osc_residual(params):
        omega, zeta = params[0], params[1]
        simulated = simulate_damped_oscillator(omega, zeta, x0_osc, t_osc)
        return simulated - observed_j

    sysid_nojit = NonlinearLeastSquaresProblem(residual=osc_residual, x0=np.array([1.0, 0.5]), jit_residual=False)
    sysid_jit = NonlinearLeastSquaresProblem(residual=osc_residual, x0=np.array([1.0, 0.5]), jit_residual=True)
    rows.append(summarize("oscillator system ID + gauss_newton, jit_residual=False", lambda: gauss_newton(sysid_nojit, max_iter=30)))
    rows.append(summarize("oscillator system ID + gauss_newton, jit_residual=True", lambda: gauss_newton(sysid_jit, max_iter=30)))

    return rows


def platform_info() -> str:
    devices = ", ".join(str(d) for d in jax.devices())
    return (
        f"- Date: {time.strftime('%Y-%m-%d')}\n"
        f"- Python: {sys.version.split()[0]}\n"
        f"- JAX: {jax.__version__} (devices: {devices})\n"
        f"- Platform: {platform.platform()}\n"
        f"- Methodology: each problem/solver pair built once, run {WARMUP} warmup call "
        f"(discarded) + {REPEATS} timed repeats; table shows the median. Warmup matters "
        f"because `Problem`'s `grad`/`hess` (and, where set, `jit_f`/`jit_residual`) are "
        f"`jax.jit`-compiled lazily on first call and cached per-instance -- an "
        f"un-warmed timing mostly measures JIT tracing, not steady-state solver cost."
    )


def main() -> None:
    arena = benchmark_arena()
    other = benchmark_other_solvers()
    jit_flag = benchmark_jit_flag()

    sections = [
        "# Solver benchmarks\n",
        (
            "Generated by `benchmarks/run_benchmarks.py` -- do not hand-edit; re-run the "
            "script to regenerate.\n"
        ),
        platform_info() + "\n",
        "## A. Solver arena: every `ALL_SOLVERS` entry x every benchmark landscape (2D)\n",
        (
            "Apples-to-apples: every solver here shares the same `Problem` interface and ran "
            "against the identical problem instance (fixed default `x0`, seed 0). Sorted "
            "fastest-to-slowest within each problem.\n"
        ),
    ]
    for name, rows in arena.items():
        sections.append(f"### {name}\n")
        sections.append(render_table(rows) + "\n")

    sections.append(
        "## B. Everything else, one representative problem each\n"
        "Each of these needs a differently-shaped problem than `Problem` (residuals, "
        "prox operators, inequality constraints, an `(A, b)` linear system, ...), so "
        "**these rows are not comparable to each other or to section A** -- different "
        "problem sizes and different problem shapes. `admm` and `proximal_gradient` "
        "*are* directly comparable to each other: both solve the identical LASSO "
        "problem.\n"
    )
    sections.append(render_table(other) + "\n")

    sections.append(
        "## C. `jit_f` / `jit_residual` opt-in flags, on vs off\n"
        "Same problem, same solver, only the flag changes -- isolates the effect of "
        "the JIT-compiled objective/residual added this session "
        "(`optimlab.core.Problem.jit_f`, `NonlinearLeastSquaresProblem.jit_residual`).\n"
    )
    sections.append(render_table(jit_flag) + "\n")

    out_path = Path(__file__).parent / "BENCHMARKS.md"
    out_path.write_text("\n".join(sections))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
