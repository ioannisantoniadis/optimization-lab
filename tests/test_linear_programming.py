import numpy as np
import pytest
from scipy.optimize import linprog

from optimlab.optimizers.linear_programming import LinearProgram, simplex


def test_classic_ub_only_problem():
    """maximize 3x1+5x2 s.t. x1<=4, 2x2<=12, 3x1+2x2<=18, x>=0 -- textbook example with
    a well-known optimum at (2, 6).
    """
    lp = LinearProgram(c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
    result = simplex(lp)
    assert result.status == "optimal"
    np.testing.assert_allclose(result.x, [2.0, 6.0], atol=1e-8)
    assert result.objective == pytest.approx(-36.0, abs=1e-8)


def test_equality_constraints_need_phase_one():
    """No slack gives a free starting basic feasible solution here -- exercises the
    artificial-variable phase 1 path, not just phase 2.
    """
    lp = LinearProgram(c=[1.0, 1.0], A_eq=[[1, 2], [3, 1]], b_eq=[4, 6])
    result = simplex(lp)
    assert result.status == "optimal"
    np.testing.assert_allclose(result.x, [1.6, 1.2], atol=1e-8)
    assert result.objective == pytest.approx(2.8, abs=1e-8)


def test_mixed_ub_and_eq_constraints():
    lp = LinearProgram(
        c=[2.0, 3.0, 1.0],
        A_ub=[[1.0, 1.0, 1.0]], b_ub=[10.0],
        A_eq=[[1.0, -1.0, 0.0]], b_eq=[2.0],
    )
    result = simplex(lp)
    ref = linprog(
        lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, A_eq=lp.A_eq, b_eq=lp.b_eq,
        bounds=[(0, None)] * 3, method="highs",
    )
    assert result.status == "optimal"
    assert result.objective == pytest.approx(ref.fun, abs=1e-6)


def test_detects_infeasible_problem():
    """x1+x2 <= 1 and x1+x2 >= 3 can't both hold for x >= 0."""
    lp = LinearProgram(c=[1.0, 1.0], A_ub=[[1, 1], [-1, -1]], b_ub=[1, -3])
    result = simplex(lp)
    assert result.status == "infeasible"


def test_detects_unbounded_problem():
    """min -x1-x2 s.t. x1-x2<=1, x>=0 -- x2 can grow without bound."""
    lp = LinearProgram(c=[-1.0, -1.0], A_ub=[[1.0, -1.0]], b_ub=[1.0])
    result = simplex(lp)
    assert result.status == "unbounded"


def test_no_constraints_beyond_nonnegativity():
    lp_optimal = LinearProgram(c=[1.0, 2.0])
    result = simplex(lp_optimal)
    assert result.status == "optimal"
    np.testing.assert_allclose(result.x, [0.0, 0.0])

    lp_unbounded = LinearProgram(c=[1.0, -2.0])
    result = simplex(lp_unbounded)
    assert result.status == "unbounded"


def test_records_a_vertex_per_pivot():
    lp = LinearProgram(c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
    result = simplex(lp)
    assert len(result.vertices) == result.n_iter + 1
    # every recorded vertex must itself be feasible
    for v in result.vertices:
        assert np.all(np.asarray(lp.A_ub) @ v <= np.asarray(lp.b_ub) + 1e-6)
        assert np.all(v >= -1e-8)
    np.testing.assert_allclose(result.vertices[-1], result.x)


@pytest.mark.parametrize("seed", range(20))
def test_matches_scipy_on_random_bounded_feasible_lps(seed):
    """Randomized cross-check against scipy.optimize.linprog (HiGHS), restricted to a
    bounded feasible region (an explicit x <= 100 box) so every instance has an
    unambiguous 'optimal' status on both sides -- an earlier, unbounded-region version
    of this test hit one scipy/HiGHS instance that mislabeled a genuinely unbounded LP
    as infeasible (independently confirmed via an explicit unbounded recession
    direction), which is a HiGHS quirk on ambiguous instances, not a simplex bug here.
    """
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 6)
    m = rng.integers(1, 6)
    c = rng.uniform(-5, 5, size=n)
    A_ub = rng.uniform(-5, 5, size=(m, n))
    x_feas = rng.uniform(0, 5, size=n)
    b_ub = A_ub @ x_feas + rng.uniform(0, 5, size=m)
    A_ub = np.vstack([A_ub, np.eye(n)])
    b_ub = np.concatenate([b_ub, np.full(n, 100.0)])

    lp = LinearProgram(c=c, A_ub=A_ub, b_ub=b_ub)
    result = simplex(lp)
    ref = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method="highs")

    assert ref.status == 0
    assert result.status == "optimal"
    assert result.objective == pytest.approx(ref.fun, abs=1e-5, rel=1e-5)
