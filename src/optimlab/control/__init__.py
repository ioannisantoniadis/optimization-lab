"""Optimal control, three ways: LQR (closed form, via the discrete-time Riccati
equation), nonlinear trajectory optimization via direct shooting (an ordinary
`optimlab.core.Problem` over an entire control sequence, solved by this repo's
existing gradient-based solvers), and dynamic programming (value iteration — the one
genuinely different algorithm here, a discrete-state Bellman fixed-point iteration
rather than continuous optimization at all).
"""

import importlib

from optimlab.control.dynamic_programming import ACTIONS, GridWorld, value_iteration
from optimlab.control.lqr import LQRResult, simulate_lqr, solve_lqr

__all__ = [
    "ACTIONS",
    "GridWorld",
    "LQRResult",
    "pendulum_dynamics",
    "simulate_lqr",
    "simulate_pendulum",
    "solve_lqr",
    "swingup_problem",
    "value_iteration",
]

#: `optimlab.control.trajectory_optimization` hard-imports jax at module level (jax has
#: no WebAssembly/Pyodide build), so importing it eagerly here would break
#: `optimlab.control` for every caller — including LQR/dynamic-programming-only callers
#: like the WASM-exported marimo notebooks. Resolved lazily via PEP 562.
_LAZY = {
    "pendulum_dynamics": "optimlab.control.trajectory_optimization",
    "simulate_pendulum": "optimlab.control.trajectory_optimization",
    "swingup_problem": "optimlab.control.trajectory_optimization",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
