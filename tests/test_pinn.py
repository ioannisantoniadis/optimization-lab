import numpy as np
import pytest

from optimlab.highdim.nets import MLPShape
from optimlab.ml.pinn import ode_pinn_problem, predict
from optimlab.optimizers import bfgs


def test_pinn_matches_the_analytic_solution_it_was_never_shown():
    """The whole point of a PINN: the network is trained purely from the ODE's own
    residual and the initial condition -- the true solution y0*exp(-decay_rate*x)
    never appears anywhere in ode_pinn_problem's loss. Matching it this closely
    afterward is a genuine physics-informed result, not curve-fitting to labeled data.
    """
    decay_rate, y0 = 0.5, 2.0
    x_range = (0.0, 5.0)
    shape = MLPShape(layer_sizes=[1, 20, 20, 1])

    problem = ode_pinn_problem(decay_rate, y0, x_range, shape, n_collocation=50, seed=0)
    result = bfgs(problem, max_iter=1000)

    xs_test = np.linspace(*x_range, 20)
    predicted = np.asarray(predict(result.x, shape, xs_test))
    true = y0 * np.exp(-decay_rate * xs_test)

    assert np.max(np.abs(predicted - true)) < 0.01


def test_pinn_satisfies_the_initial_condition():
    decay_rate, y0 = 0.5, 2.0
    shape = MLPShape(layer_sizes=[1, 20, 20, 1])
    problem = ode_pinn_problem(decay_rate, y0, (0.0, 5.0), shape, n_collocation=50, seed=0)
    result = bfgs(problem, max_iter=1000)

    predicted_at_zero = float(np.asarray(predict(result.x, shape, np.array([0.0])))[0])
    assert predicted_at_zero == pytest.approx(y0, abs=0.01)
