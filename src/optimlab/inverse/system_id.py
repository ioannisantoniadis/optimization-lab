"""System identification: recovering a dynamical system's own physical parameters (not
its trajectory) from noisy observations of it — turning "here's what the system did"
into "here's what the system *is*." Framed as an ordinary nonlinear least squares
problem (`optimlab.optimizers.gauss_newton`): simulate the candidate system forward
from the same initial condition, and minimize the residual between that simulation and
what was actually observed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from optimlab.core import ArrayLike
from optimlab.optimizers.gauss_newton import NonlinearLeastSquaresProblem


def simulate_damped_oscillator(omega: ArrayLike, zeta: ArrayLike, x0: ArrayLike, t: ArrayLike) -> ArrayLike:
    """Position `x(t)` of a damped harmonic oscillator
    `x'' + 2*zeta*omega*x' + omega^2*x = 0`, starting from `x0 = (position, velocity)`,
    at every evenly-spaced time in `t`. Integrated with a plain RK4 stepper — written
    entirely in `jax.numpy` so it stays differentiable when `omega`/`zeta` are traced
    parameters, not just when called with concrete floats — rather than the closed
    form, so the identical code handles the under/critically/over-damped regimes a
    closed form would treat as separate cases.
    """
    t = np.asarray(t)
    dt = float(t[1] - t[0])

    def deriv(state: ArrayLike) -> ArrayLike:
        pos, vel = state[0], state[1]
        return jnp.array([vel, -2.0 * zeta * omega * vel - omega**2 * pos])

    state = jnp.asarray(x0, dtype=jnp.float64)
    positions = [state[0]]
    for _ in range(len(t) - 1):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        positions.append(state[0])
    return jnp.stack(positions)


def oscillator_identification_problem(
    t: ArrayLike, observed_positions: ArrayLike, x0: ArrayLike, *, params0: ArrayLike
) -> NonlinearLeastSquaresProblem:
    """`params = (omega, zeta)`; the residual at each time step is
    `simulated_position(params) - observed_position` — Gauss-Newton's own Jacobian
    (autodiff through the RK4 loop above) is exactly the sensitivity of the simulated
    trajectory to the unknown physical parameters, standard "adjoint sensitivity"
    territory obtained here for free from `jax.jacfwd` rather than derived by hand.
    """
    observed_j = jnp.asarray(observed_positions)

    def residual(params: ArrayLike) -> ArrayLike:
        omega, zeta = params[0], params[1]
        simulated = simulate_damped_oscillator(omega, zeta, x0, t)
        return simulated - observed_j

    return NonlinearLeastSquaresProblem(residual=residual, x0=params0, name="oscillator_identification")
