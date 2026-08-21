"""Optuna adapter: a third black-box hyperparameter-search backend (TPE — Tree-structured
Parzen Estimator, a Bayesian sequential model-based optimizer that fits a probabilistic
model of "which regions look promising" from every trial so far) alongside Phase 3's
from-scratch `genetic_algorithm` / `particle_swarm` / `simulated_annealing`. Requires the
`backends` extra (`uv sync --extra backends`).
"""

from __future__ import annotations

import time

import numpy as np
import optuna

from optimlab.core import OptimizeResult, Problem

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_minimize(
    problem: Problem, *, bounds: tuple[float, float] | None = None, n_trials: int = 200, seed: int = 0
) -> OptimizeResult:
    """Minimize `problem.f` over an n-dimensional box (`bounds`, defaulting to
    `problem.domain` — the same convention `genetic_algorithm`/`particle_swarm` use)
    with `n_trials` calls to `problem.f`. Unlike this repo's from-scratch population
    methods, TPE is *sequential* — each trial's search region is informed by every
    previous trial's result, not just a fixed population evolving together — which is
    the actual value a real Bayesian-optimization library adds over a from-scratch
    genetic algorithm for genuinely expensive-to-evaluate objectives (hyperparameter
    tuning, not the cheap benchmark functions used to test it here).
    """
    low, high = bounds if bounds is not None else problem.domain or (-5.0, 5.0)
    n = problem.n_dim

    def objective(trial: optuna.Trial) -> float:
        x = np.array([trial.suggest_float(f"x{i}", low, high) for i in range(n)])
        return float(problem.f(x))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    wall_time = time.perf_counter() - start

    x_best = np.array([study.best_params[f"x{i}"] for i in range(n)])
    f_trajectory = [t.value for t in study.trials if t.value is not None]
    return OptimizeResult(
        x=x_best, f=float(study.best_value), n_iter=n_trials, converged=True,
        solver_name="optuna_minimize", message="n_trials completed", wall_time=wall_time,
        trajectory=[x_best], f_trajectory=f_trajectory,
    )
