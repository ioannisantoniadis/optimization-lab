import numpy as np
import pytest

from optimlab.inverse.system_id import oscillator_identification_problem, simulate_damped_oscillator
from optimlab.optimizers import gauss_newton


def test_undamped_oscillator_conserves_amplitude():
    """zeta=0 should never lose energy -- position should keep returning to its
    starting amplitude every period rather than decaying.
    """
    t = np.linspace(0, 4 * np.pi, 400)  # a few full periods at omega=1
    trajectory = np.asarray(simulate_damped_oscillator(1.0, 0.0, x0=np.array([1.0, 0.0]), t=t))
    assert trajectory.max() == pytest.approx(1.0, abs=0.05)
    assert trajectory.min() == pytest.approx(-1.0, abs=0.05)


def test_damped_oscillator_decays():
    t = np.linspace(0, 20, 200)
    trajectory = np.asarray(simulate_damped_oscillator(2.0, 0.3, x0=np.array([1.0, 0.0]), t=t))
    # a damped oscillator's envelope shrinks -- late-time amplitude much smaller than early
    assert np.max(np.abs(trajectory[-40:])) < 0.3 * np.max(np.abs(trajectory[:40]))


def test_gauss_newton_recovers_true_parameters_from_noisy_observations():
    true_omega, true_zeta = 2.5, 0.15
    x0 = np.array([1.0, 0.0])
    t = np.linspace(0, 10, 40)

    true_trajectory = np.asarray(simulate_damped_oscillator(true_omega, true_zeta, x0, t))
    rng = np.random.default_rng(0)
    observed = true_trajectory + 0.02 * rng.standard_normal(true_trajectory.shape)

    problem = oscillator_identification_problem(t, observed, x0, params0=np.array([1.0, 0.5]))
    result = gauss_newton(problem, max_iter=30)

    assert result.converged
    assert result.x[0] == pytest.approx(true_omega, abs=0.05)
    assert result.x[1] == pytest.approx(true_zeta, abs=0.05)
