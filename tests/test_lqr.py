import jax.numpy as jnp
import numpy as np
import pytest

from optimlab.control.lqr import simulate_lqr, solve_lqr
from optimlab.core import Problem
from optimlab.optimizers import bfgs


def _double_integrator():
    dt = 0.1
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.0], [dt]])
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])
    Q_f = np.diag([10.0, 10.0])
    return A, B, Q, R, Q_f


def test_lqr_drives_the_double_integrator_toward_the_origin():
    A, B, Q, R, Q_f = _double_integrator()
    result = solve_lqr(A, B, Q, R, Q_f, n_steps=30)
    states, _controls = simulate_lqr(A, B, result.gains, x0=np.array([1.0, 0.0]))
    assert np.linalg.norm(states[-1]) < 0.05
    assert np.linalg.norm(states[-1]) < np.linalg.norm(states[0])


def test_riccati_solution_matches_direct_optimization_over_the_control_sequence():
    """The actual correctness check: solve the identical finite-horizon LQR problem
    two completely different ways -- the closed-form backward Riccati recursion, and
    an ordinary Problem over the flattened control sequence solved by bfgs -- and
    confirm they land on the same cost and the same controls.
    """
    A, B, Q, R, Q_f = _double_integrator()
    n_steps = 30
    x0 = np.array([1.0, 0.0])

    lqr_result = solve_lqr(A, B, Q, R, Q_f, n_steps)
    states, controls = simulate_lqr(A, B, lqr_result.gains, x0)
    riccati_cost = sum(
        states[k] @ Q @ states[k] + controls[k] @ R @ controls[k] for k in range(n_steps)
    ) + states[-1] @ Q_f @ states[-1]

    n_u = B.shape[1]
    A_j, B_j, Q_j, R_j, Qf_j = (jnp.asarray(m) for m in (A, B, Q, R, Q_f))
    x0_j = jnp.asarray(x0)

    def rollout_cost(flat_u):
        u_seq = flat_u.reshape(n_steps, n_u)
        x = x0_j
        cost = 0.0
        for k in range(n_steps):
            u = u_seq[k]
            cost = cost + x @ Q_j @ x + u @ R_j @ u
            x = A_j @ x + B_j @ u
        return cost + x @ Qf_j @ x

    problem = Problem(f=rollout_cost, x0=np.zeros(n_steps * n_u), name="lqr_direct")
    direct_result = bfgs(problem, max_iter=500)

    assert direct_result.converged
    assert direct_result.f == pytest.approx(riccati_cost, rel=1e-5)
    np.testing.assert_allclose(direct_result.x, controls.ravel(), atol=1e-4)
