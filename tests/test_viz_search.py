import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.landscapes import get
from optimlab.optimizers import genetic_algorithm, nelder_mead, particle_swarm
from optimlab.viz import population_snapshots_figure, simplex_steps_figure


def _rastrigin():
    return get("rastrigin").problem(x0=np.array([4.3, -3.7]))


def test_population_snapshots_draw_one_panel_per_requested_iteration():
    problem = _rastrigin()
    result = particle_swarm(problem, seed=0, max_iter=30, record_history=True)
    fig = population_snapshots_figure(problem, result, [0, 10, 30])
    assert isinstance(fig, go.Figure)
    scatters = [t for t in fig.data if t.name == "population"]
    assert len(scatters) == 3
    np.testing.assert_allclose(scatters[0].x, result.history["population"][0][:, 0])


def test_population_snapshots_clamp_iterations_past_the_end():
    problem = _rastrigin()
    result = genetic_algorithm(problem, seed=0, max_generations=5, record_history=True)
    fig = population_snapshots_figure(problem, result, [0, 999])
    assert fig.layout.annotations[-1].text == f"iteration {result.n_iter}"


def test_population_snapshots_need_recorded_history():
    problem = _rastrigin()
    with pytest.raises(ValueError, match="record_history"):
        population_snapshots_figure(problem, particle_swarm(problem, max_iter=5), [0])


def test_simplex_steps_draw_one_closed_triangle_per_step():
    problem = get("rosenbrock").problem(x0=np.array([-1.2, 1.0]))
    result = nelder_mead(problem, max_iter=50, record_history=True)
    fig = simplex_steps_figure(problem, result, n_steps=12)
    triangles = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(triangles) == 13  # the starting simplex plus 12 moves
    for tri in triangles:
        assert len(tri.x) == 4 and tri.x[0] == tri.x[-1]


def test_simplex_steps_need_recorded_history():
    problem = get("rosenbrock").problem(x0=np.array([-1.2, 1.0]))
    with pytest.raises(ValueError, match="record_history"):
        simplex_steps_figure(problem, nelder_mead(problem, max_iter=5))


def test_bayes_opt_step_marks_the_expected_improvement_maximizer():
    from optimlab.optimizers.bayesian_optimization import expected_improvement, gp_posterior
    from optimlab.viz import bayes_opt_step_figure

    def f(x):
        return np.sin(3 * x) + 0.3 * x**2

    X = np.array([-2.4, -0.9, 0.4, 2.2])
    fig = bayes_opt_step_figure(f, X, (-3.0, 3.0), length_scale=0.6, variance=1.5)
    xs = np.linspace(-3.0, 3.0, 300)
    post = gp_posterior(X[:, None], f(X), xs[:, None], length_scale=0.6, variance=1.5)
    ei = expected_improvement(post.mean, post.std, float(f(X).min()))
    assert fig.layout.shapes[0].x0 == pytest.approx(xs[np.argmax(ei)])
    evaluated = next(t for t in fig.data if t.name == "evaluated points")
    np.testing.assert_allclose(evaluated.y, f(X))
