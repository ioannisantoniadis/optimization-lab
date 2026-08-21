import numpy as np
import pytest

from optimlab.landscapes.testfunctions import ALL_FUNCTIONS, get


def test_registry_covers_expected_functions():
    expected = {
        "sphere", "rosenbrock", "rastrigin", "ackley",
        "himmelblau", "beale", "styblinski_tang", "matyas",
    }
    assert expected <= set(ALL_FUNCTIONS)


def test_known_minima_evaluate_to_f_min():
    for name in ("sphere", "himmelblau", "beale", "matyas"):
        bf = get(name)
        for minimum in bf.minima:
            n_dim = bf.n_dim or minimum.size
            x = np.resize(minimum, n_dim)
            assert float(bf.f(x)) == pytest.approx(bf.f_min, abs=1e-6)


def test_rosenbrock_minimum_is_zero_at_ones():
    bf = get("rosenbrock")
    for n_dim in (2, 5, 10):
        x = np.ones(n_dim)
        assert float(bf.f(x)) == pytest.approx(0.0, abs=1e-9)


def test_closed_form_gradients_match_autodiff():
    """Sphere and Rosenbrock carry a hand-derived closed-form gradient; check it agrees
    with the autodiff gradient a `Problem` would compute if `grad` were left as None.
    """
    from optimlab.core import _autograd

    for name in ("sphere", "rosenbrock"):
        bf = get(name)
        autograd_fn = _autograd(bf.f)
        rng = np.random.default_rng(42)
        x = rng.uniform(*bf.domain, size=6)
        np.testing.assert_allclose(bf.grad(x), autograd_fn(x), rtol=1e-4, atol=1e-6)


def test_problem_builds_with_random_start_in_domain():
    bf = get("rastrigin")
    problem = bf.problem(n_dim=8, seed=1)
    assert problem.n_dim == 8
    low, high = bf.domain
    assert np.all(problem.x0 >= low) and np.all(problem.x0 <= high)


@pytest.mark.parametrize("name,n_dim", [("rastrigin", 2), ("sphere", 4), ("rosenbrock", 3)])
def test_problem_minimum_is_broadcast_to_the_actual_dimension(name, n_dim):
    """These benchmarks store their minimum as a length-1 placeholder (any-dimension
    functions: the optimum is e.g. "every coordinate is 0"), which previously leaked
    straight into `Problem.minimum` unbroadcast -- fine for evaluating f_min, but a
    length-1 "point" silently breaks any 2D viz code indexing minimum[0], minimum[1]
    (caught via optimlab.viz.contour_figure raising IndexError on rastrigin).
    """
    bf = get(name)
    problem = bf.problem(n_dim=n_dim)
    assert problem.minimum.shape == (n_dim,)
    assert float(bf.f(problem.minimum)) == pytest.approx(bf.f_min, abs=1e-6)
