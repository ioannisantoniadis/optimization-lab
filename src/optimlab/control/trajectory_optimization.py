"""Nonlinear optimal control via direct shooting: parameterize the *entire control
sequence* as the optimization variable, simulate the (nonlinear) dynamics forward from
a fixed initial state, and minimize a cost combining control effort with how far the
final state lands from the target. Unlike LQR, there's no closed form once the dynamics
are nonlinear — but there doesn't need to be one: this is an ordinary `Problem`, no
different in kind from any other in this repo, just with "the state trajectory an
entire control sequence produces" as the thing being differentiated through.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike, Problem


def pendulum_dynamics(state: ArrayLike, u: ArrayLike, *, g: float = 9.81, length: float = 1.0, damping: float = 0.1) -> ArrayLike:
    """A torque-driven pendulum, `theta` measured from hanging straight down
    (`theta=0`, the stable equilibrium with no control) to upright
    (`theta=pi`): `theta'' = -(g/length) sin(theta) - damping * theta' + u`.
    """
    theta, theta_dot = state[0], state[1]
    theta_ddot = -(g / length) * jnp.sin(theta) - damping * theta_dot + u
    return jnp.array([theta_dot, theta_ddot])


def simulate_pendulum(x0: ArrayLike, controls: ArrayLike, dt: float, **dynamics_kwargs) -> ArrayLike:
    """RK4-integrate `pendulum_dynamics` forward from `x0`, applying `controls[k]` as a
    constant torque over each step of length `dt` — written in `jax.numpy` throughout
    so it stays differentiable with respect to `controls` (used by `swingup_problem`).
    """
    state = jnp.asarray(x0, dtype=jnp.float64)
    states = [state]
    for u in controls:
        k1 = pendulum_dynamics(state, u, **dynamics_kwargs)
        k2 = pendulum_dynamics(state + 0.5 * dt * k1, u, **dynamics_kwargs)
        k3 = pendulum_dynamics(state + 0.5 * dt * k2, u, **dynamics_kwargs)
        k4 = pendulum_dynamics(state + dt * k3, u, **dynamics_kwargs)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(state)
    return jnp.stack(states)


def swingup_problem(
    x0: ArrayLike,
    x_target: ArrayLike,
    n_steps: int,
    dt: float,
    *,
    control_penalty: float = 0.01,
    terminal_weight: float = 100.0,
    **dynamics_kwargs,
) -> Problem:
    """`x = the flat control sequence u_0, ..., u_{n_steps-1}`. The cost trades off
    total control effort (`control_penalty * sum(u_k^2) * dt`, a discretized `L2` norm
    of the torque signal) against a heavily-weighted terminal penalty for missing
    `x_target` — heavy enough that the optimizer's dominant concern is actually
    reaching the target, with minimizing effort only a secondary preference once it
    does.
    """
    x0_j = jnp.asarray(x0, dtype=jnp.float64)
    x_target_j = jnp.asarray(x_target, dtype=jnp.float64)

    def cost(controls: ArrayLike) -> float:
        trajectory = simulate_pendulum(x0_j, controls, dt, **dynamics_kwargs)
        control_cost = control_penalty * jnp.sum(controls**2) * dt
        terminal_cost = terminal_weight * jnp.sum((trajectory[-1] - x_target_j) ** 2)
        return control_cost + terminal_cost

    return Problem(f=cost, x0=np.zeros(n_steps), name="pendulum_swingup")
