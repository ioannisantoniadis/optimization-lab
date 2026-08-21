import numpy as np
import pytest

from optimlab.core import Problem
from optimlab.landscapes.testfunctions import get
from optimlab.optimizers import (
    ALL_SOLVERS,
    adagrad,
    adam,
    bfgs,
    gradient_descent,
    heavy_ball,
    lbfgs,
    nesterov,
    newton_method,
    rmsprop,
)

SPHERE = get("sphere")


def _sphere_problem(n_dim: int = 5, seed: int = 0) -> Problem:
    return SPHERE.problem(n_dim=n_dim, seed=seed)


@pytest.mark.parametrize(
    "solver, kwargs",
    [
        (gradient_descent, {"lr": 0.3, "max_iter": 500}),
        (heavy_ball, {"lr": 0.1, "beta": 0.9, "max_iter": 500}),
        (nesterov, {"lr": 0.1, "beta": 0.9, "max_iter": 500}),
        (adagrad, {"lr": 0.5, "max_iter": 2000}),
        (adam, {"lr": 0.1, "max_iter": 2000}),
        (newton_method, {"max_iter": 20}),
        (bfgs, {"max_iter": 100}),
        (lbfgs, {"max_iter": 100}),
    ],
)
def test_solver_converges_on_convex_sphere(solver, kwargs):
    """Every solver in the repo should nail the textbook convex sanity check."""
    problem = _sphere_problem()
    result = solver(problem, **kwargs)
    assert result.converged, result.message
    assert result.f == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(result.x, np.zeros(problem.n_dim), atol=1e-2)


def test_rmsprop_converges_to_a_neighborhood_not_a_point():
    """RMSProp with a *constant* step size doesn't converge to the exact minimum on a
    deterministic (full-batch) convex problem — near x*, all coordinates' gradients
    shrink together, so the per-coordinate normalization `g / (sqrt(E[g^2]) + eps)`
    stays order-1 instead of shrinking, and the optimizer settles into a limit cycle
    whose radius scales with `lr`. This is the exact non-convergence issue Reddi et al.
    formalize in "On the Convergence of Adam and Beyond" — a real, documented property,
    not a bug in this implementation. (Adagrad avoids it because its accumulator only
    grows, so its effective step size decays to zero; Adam's momentum smooths it enough
    in practice that the ordinary sphere test above already passes.)
    """
    problem = _sphere_problem()
    result = rmsprop(problem, lr=0.01, max_iter=3000, tol=1e-6)
    assert not result.converged
    assert result.f < 1e-3


def test_all_solvers_registry_matches_importable_functions():
    assert set(ALL_SOLVERS) == {
        "gradient_descent", "heavy_ball", "nesterov", "adagrad",
        "rmsprop", "adam", "newton", "bfgs", "lbfgs",
        "nelder_mead", "simulated_annealing", "genetic_algorithm", "particle_swarm",
    }


def test_newton_converges_in_one_step_on_a_quadratic():
    """Newton's method solves a quadratic model exactly, so on an actual quadratic
    objective it should reach the minimum in a single Newton step (up to line-search
    bookkeeping), unlike every gradient-only method here.
    """
    def f(x):
        return 0.5 * (x[0] ** 2 + 100.0 * x[1] ** 2)

    def grad(x):
        return np.array([x[0], 100.0 * x[1]])

    def hess(x):
        return np.array([[1.0, 0.0], [0.0, 100.0]])

    problem = Problem(f=f, x0=np.array([10.0, 1.0]), grad=grad, hess=hess, name="anisotropic_quadratic")
    result = newton_method(problem, max_iter=20)
    assert result.converged
    assert result.n_iter <= 2
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-6)


def test_ill_conditioning_orders_solver_iteration_counts():
    """On an ill-conditioned quadratic (condition number 100), fixed-step gradient
    descent should zig-zag and take far more iterations than curvature-aware BFGS or
    Newton — the concrete demonstration behind ROADMAP Phase 1's "why momentum/Newton
    exist" narrative.
    """
    def f(x):
        return 0.5 * (x[0] ** 2 + 100.0 * x[1] ** 2)

    def grad(x):
        return np.array([x[0], 100.0 * x[1]])

    def hess(x):
        return np.array([[1.0, 0.0], [0.0, 100.0]])

    x0 = np.array([10.0, 1.0])
    gd_problem = Problem(f=f, x0=x0.copy(), grad=grad, hess=hess)
    bfgs_problem = Problem(f=f, x0=x0.copy(), grad=grad, hess=hess)
    newton_problem = Problem(f=f, x0=x0.copy(), grad=grad, hess=hess)

    gd_result = gradient_descent(gd_problem, lr=0.0099, max_iter=5000, tol=1e-6)
    bfgs_result = bfgs(bfgs_problem, max_iter=200, tol=1e-6)
    newton_result = newton_method(newton_problem, max_iter=20, tol=1e-6)

    assert gd_result.converged and bfgs_result.converged and newton_result.converged
    assert newton_result.n_iter < bfgs_result.n_iter < gd_result.n_iter


def test_bfgs_and_lbfgs_agree_on_rosenbrock():
    rosenbrock = get("rosenbrock")
    problem_bfgs = rosenbrock.problem(x0=np.array([-1.2, 1.0]))
    problem_lbfgs = rosenbrock.problem(x0=np.array([-1.2, 1.0]))

    result_bfgs = bfgs(problem_bfgs, max_iter=500, tol=1e-8)
    result_lbfgs = lbfgs(problem_lbfgs, max_iter=500, tol=1e-8, memory=10)

    assert result_bfgs.converged
    assert result_lbfgs.converged
    np.testing.assert_allclose(result_bfgs.x, np.ones(2), atol=1e-3)
    np.testing.assert_allclose(result_lbfgs.x, np.ones(2), atol=1e-3)
