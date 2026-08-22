import numpy as np
import pytest

from optimlab.control.trajectory_optimization import simulate_pendulum, swingup_problem
from optimlab.optimizers import bfgs


def test_pendulum_dynamics_conserve_energy_with_zero_damping_and_control():
    """A frictionless, uncontrolled pendulum should keep swinging at (nearly) the same
    amplitude -- a basic sanity check on the RK4 integrator before trusting it inside
    an optimization loop.
    """
    x0 = np.array([1.0, 0.0])  # released from 1 radian, at rest
    controls = np.zeros(200)
    trajectory = np.asarray(simulate_pendulum(x0, controls, dt=0.02, damping=0.0))
    assert trajectory[:, 0].max() == pytest.approx(1.0, abs=0.02)


def test_swingup_reaches_upright_from_hanging_down():
    """The actual optimal-control claim: starting from the stable equilibrium (hanging
    straight down, no control needed to stay there), direct shooting finds a control
    sequence that swings the pendulum up to the unstable equilibrium (upright, at
    rest) -- something no small perturbation would ever do on its own.
    """
    x0 = np.array([0.0, 0.0])
    x_target = np.array([np.pi, 0.0])
    n_steps, dt = 20, 0.1

    problem = swingup_problem(x0, x_target, n_steps, dt, control_penalty=0.01, terminal_weight=200.0)
    result = bfgs(problem, max_iter=100)

    trajectory = np.asarray(simulate_pendulum(x0, result.x, dt))
    final_theta, final_theta_dot = trajectory[-1]

    assert abs(final_theta - np.pi) < 0.05
    assert abs(final_theta_dot) < 0.1
