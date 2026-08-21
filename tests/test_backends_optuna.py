import numpy as np
import pytest

from optimlab.landscapes import get

# optuna lives behind the optional `backends` extra (`uv sync --extra backends`).
pytest.importorskip("optuna", reason="requires the 'backends' extra")
from optimlab.backends.optuna_backend import optuna_minimize


def test_optuna_minimize_finds_the_sphere_minimum():
    bf = get("sphere")
    problem = bf.problem(x0=np.array([3.0, -3.0]))
    result = optuna_minimize(problem, bounds=(-5.0, 5.0), n_trials=150, seed=0)
    assert result.converged
    np.testing.assert_allclose(result.x, [0.0, 0.0], atol=0.1)


def test_optuna_minimize_defaults_bounds_to_problem_domain():
    bf = get("rastrigin")
    problem = bf.problem(x0=np.array([4.0, -4.0]))
    result = optuna_minimize(problem, n_trials=200, seed=0)
    assert result.f < 5.0  # Rastrigin's global min is 0; a near-global basin is enough
