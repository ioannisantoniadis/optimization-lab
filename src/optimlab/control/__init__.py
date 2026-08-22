"""Optimal control, three ways: LQR (closed form, via the discrete-time Riccati
equation), nonlinear trajectory optimization via direct shooting (an ordinary
`optimlab.core.Problem` over an entire control sequence, solved by this repo's
existing gradient-based solvers), and dynamic programming (value iteration — the one
genuinely different algorithm here, a discrete-state Bellman fixed-point iteration
rather than continuous optimization at all).
"""

from optimlab.control.dynamic_programming import ACTIONS, GridWorld, value_iteration
from optimlab.control.lqr import LQRResult, simulate_lqr, solve_lqr
from optimlab.control.trajectory_optimization import (
    pendulum_dynamics,
    simulate_pendulum,
    swingup_problem,
)

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
