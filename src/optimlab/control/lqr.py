"""The Linear Quadratic Regulator: the one optimal-control problem with a closed-form
solution, via backward dynamic-programming recursion (the discrete-time Riccati
equation) rather than any of this repo's iterative solvers. For linear dynamics
`x_{k+1} = A x_k + B u_k` and a quadratic cost, the optimal control is always exactly
linear in the state, `u_k = -K_k x_k` — no search required. Included here as the
classical baseline `optimlab.control.trajectory_optimization`'s iterative, from-scratch
approach is cross-checked against on the identical problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimlab.core import ArrayLike


@dataclass
class LQRResult:
    gains: list[np.ndarray]  # K_0, ..., K_{N-1}: u_k = -K_k @ x_k
    cost_to_go: list[np.ndarray]  # P_0, ..., P_N


def solve_lqr(A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike, Q_f: ArrayLike, n_steps: int) -> LQRResult:
    """Backward Riccati recursion from the terminal cost `P_N = Q_f`: at each step,
    `K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A` is the optimal feedback gain, and
    `P_k = Q + A^T P_{k+1} A - A^T P_{k+1} B K_k` is the quadratic "cost still to come"
    from state `x_k` onward under that optimal policy — computed once, backward, before
    any control is ever applied forward.
    """
    A, B, Q, R, Q_f = (np.asarray(m, dtype=float) for m in (A, B, Q, R, Q_f))
    cost_to_go = [Q_f]
    gains: list[np.ndarray] = []
    for _ in range(n_steps):
        P_next = cost_to_go[0]
        S = R + B.T @ P_next @ B
        K_k = np.linalg.solve(S, B.T @ P_next @ A)
        P_k = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K_k
        gains.insert(0, K_k)
        cost_to_go.insert(0, P_k)
    return LQRResult(gains=gains, cost_to_go=cost_to_go)


def simulate_lqr(A: ArrayLike, B: ArrayLike, gains: list[np.ndarray], x0: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Roll the closed-loop system `x_{k+1} = (A - B K_k) x_k` forward from `x0` under
    `solve_lqr`'s feedback gains, returning the state trajectory (`n_steps+1` states)
    and the control sequence actually applied.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    x = np.asarray(x0, dtype=float)
    states = [x.copy()]
    controls = []
    for K_k in gains:
        u = -K_k @ x
        controls.append(u)
        x = A @ x + B @ u
        states.append(x.copy())
    return np.array(states), np.array(controls)
