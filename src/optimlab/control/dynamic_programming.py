"""Dynamic programming: solving a Markov Decision Process by exploiting Bellman's
optimality principle directly. Every other solver in this repo is continuous
optimization — take a step, evaluate a gradient, repeat. Value iteration is a
different kind of algorithm entirely: an exact fixed-point iteration over a *discrete*
state space, with no notion of a step size or a gradient at all. It's included in this
repo's control phase because it's the discrete-state, discrete-time relative of LQR and
trajectory optimization — all three are "find the optimal action given the future
consequences," just for genuinely different problem shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: (row delta, col delta) for up, down, left, right.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class GridWorld:
    """A deterministic grid-world MDP: an agent at `(row, col)` picks one of four
    moves each step, bumping into a wall or `obstacles` leaves it in place. Every step
    costs `step_reward` (negative, so shorter paths are preferred); reaching `goal`
    pays `goal_reward` once and the episode is absorbing (staying at the goal costs
    nothing further) — the standard "get there quickly" grid-world reward structure.
    """

    n_rows: int
    n_cols: int
    goal: tuple[int, int]
    obstacles: set[tuple[int, int]] = field(default_factory=set)
    step_reward: float = -1.0
    goal_reward: float = 10.0

    def is_in_bounds(self, state: tuple[int, int]) -> bool:
        r, c = state
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and state not in self.obstacles

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float]:
        if state == self.goal:
            return state, 0.0
        dr, dc = ACTIONS[action]
        next_state = (state[0] + dr, state[1] + dc)
        if not self.is_in_bounds(next_state):
            next_state = state
        reward = self.goal_reward if next_state == self.goal else self.step_reward
        return next_state, reward


def value_iteration(
    world: GridWorld, *, gamma: float = 0.95, tol: float = 1e-6, max_iter: int = 1000
) -> tuple[np.ndarray, np.ndarray, int]:
    """The Bellman optimality backup `V(s) <- max_a [R(s,a) + gamma * V(s')]` applied
    to every state simultaneously, repeated until `V` stops changing by more than
    `tol` — a genuine fixed-point iteration (this is a contraction mapping for
    `gamma < 1`, so it's mathematically guaranteed to converge, not just observed to in
    practice). Returns `(V, policy, n_iter)`: the converged value function, the greedy
    policy it implies (one action index per state), and how many sweeps it took.
    """
    V = np.zeros((world.n_rows, world.n_cols))
    n_iter = max_iter
    for it in range(max_iter):
        V_new = V.copy()
        delta = 0.0
        for r in range(world.n_rows):
            for c in range(world.n_cols):
                state = (r, c)
                if state in world.obstacles or state == world.goal:
                    continue
                action_values = [
                    reward + gamma * V[next_state] for next_state, reward in (world.step(state, a) for a in range(4))
                ]
                V_new[r, c] = max(action_values)
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < tol:
            n_iter = it + 1
            break

    policy = np.zeros((world.n_rows, world.n_cols), dtype=int)
    for r in range(world.n_rows):
        for c in range(world.n_cols):
            state = (r, c)
            if state in world.obstacles or state == world.goal:
                continue
            action_values = [
                reward + gamma * V[next_state] for next_state, reward in (world.step(state, a) for a in range(4))
            ]
            policy[r, c] = int(np.argmax(action_values))
    return V, policy, n_iter
