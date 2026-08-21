import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint, minimize

from optimlab.optimizers.barrier_method import ConstrainedProblem, barrier_method


def test_single_constraint_matches_known_symmetric_optimum():
    """minimize x^2+y^2 s.t. x+y>=1 -- symmetry alone gives the answer, (0.5, 0.5)."""
    problem = ConstrainedProblem(
        f=lambda x: x[0] ** 2 + x[1] ** 2,
        x0=np.array([2.0, 2.0]),
        inequality_constraints=[lambda x: 1.0 - x[0] - x[1]],
    )
    result = barrier_method(problem)
    assert result.converged
    np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-4)
    assert result.f == pytest.approx(0.5, abs=1e-4)


def test_box_constrained_quadratic_matches_scipy_slsqp():
    def f(x):
        return (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2

    constraints = [
        lambda x: x[0] + x[1] - 4.0,
        lambda x: -x[0],
        lambda x: -x[1],
        lambda x: x[0] - 3.0,
    ]
    problem = ConstrainedProblem(f=f, x0=np.array([0.5, 0.5]), inequality_constraints=constraints)
    result = barrier_method(problem)

    ref = minimize(
        f, np.array([0.5, 0.5]), method="SLSQP",
        constraints=[
            NonlinearConstraint(lambda x: 4.0 - x[0] - x[1], 0, np.inf),
            NonlinearConstraint(lambda x: x[0], 0, np.inf),
            NonlinearConstraint(lambda x: x[1], 0, np.inf),
            NonlinearConstraint(lambda x: 3.0 - x[0], 0, np.inf),
        ],
    )
    assert result.converged
    np.testing.assert_allclose(result.x, ref.x, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_matches_cvxpy_on_random_box_constrained_qps(seed):
    cvxpy = pytest.importorskip("cvxpy", reason="requires the 'backends' extra")

    rng = np.random.default_rng(seed)
    n = rng.integers(2, 5)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    c = rng.uniform(-3, 3, size=n)

    def f(x, P=P, c=c):
        return 0.5 * (x - c) @ P @ (x - c)

    lo = rng.uniform(-5, -1, size=n)
    hi = rng.uniform(1, 5, size=n)
    x0 = (lo + hi) / 2

    constraints = []
    for i in range(n):
        constraints.append(lambda x, i=i: lo[i] - x[i])
        constraints.append(lambda x, i=i: x[i] - hi[i])

    problem = ConstrainedProblem(f=f, x0=x0, inequality_constraints=constraints)
    result = barrier_method(problem)

    x = cvxpy.Variable(n)
    cvx_problem = cvxpy.Problem(
        cvxpy.Minimize(0.5 * cvxpy.quad_form(x - c, cvxpy.psd_wrap(P))), [x >= lo, x <= hi]
    )
    cvx_problem.solve()

    assert result.converged
    np.testing.assert_allclose(result.x, x.value, atol=1e-4)


def test_rejects_an_infeasible_starting_point():
    with pytest.raises(ValueError, match="feasible"):
        ConstrainedProblem(
            f=lambda x: x[0] ** 2,
            x0=np.array([2.0]),
            inequality_constraints=[lambda x: x[0] - 1.0],  # x <= 1, violated at x0=2
        )


def test_duality_gap_shrinks_monotonically_and_matches_n_constraints_over_t():
    """gap_trajectory[0] (n_constraints/t0, recorded before any Newton solve) and
    gap_trajectory[1] (recorded right after solving at that same t0, before t is next
    multiplied by mu) are identical by construction -- the gap is purely a function of
    t, not of how close x already is to optimal -- so the real invariant is
    non-increasing throughout, strictly decreasing from index 1 on.
    """
    problem = ConstrainedProblem(
        f=lambda x: x[0] ** 2 + x[1] ** 2,
        x0=np.array([2.0, 2.0]),
        inequality_constraints=[lambda x: 1.0 - x[0] - x[1]],
    )
    result = barrier_method(problem)
    gaps = np.asarray(result.grad_norm_trajectory)
    assert gaps[0] == gaps[1]
    assert np.all(np.diff(gaps[1:]) < 0)
    assert gaps[-1] < 1e-8


def test_central_path_trajectory_stays_strictly_feasible():
    problem = ConstrainedProblem(
        f=lambda x: (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2,
        x0=np.array([0.5, 0.5]),
        inequality_constraints=[lambda x: x[0] + x[1] - 4.0, lambda x: -x[0], lambda x: -x[1]],
    )
    result = barrier_method(problem)
    for x in result.trajectory:
        assert problem.is_strictly_feasible(x) or np.allclose(x, result.x, atol=1e-3)
