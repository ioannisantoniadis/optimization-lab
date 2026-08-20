"""Thin adapters to scipy / cvxpy / jax·optax / optuna / nevergrad / pymoo, each exposing
the same `Problem -> OptimizeResult` interface as the from-scratch solvers in
`optimlab.optimizers`, so they can be used as correctness oracles and scale-up backends
in the solver arena.

Not yet implemented — lands in Phase 2 (scipy/cvxpy, as correctness oracles for the
from-scratch simplex/CG implementations). See ROADMAP.md.
"""
