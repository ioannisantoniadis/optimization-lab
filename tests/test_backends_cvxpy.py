import numpy as np
import pytest

from optimlab.backends import scipy_linprog
from optimlab.linalg import equality_constrained_qp
from optimlab.optimizers import LinearProgram, simplex

# cvxpy lives behind the optional `backends` extra (`uv sync --extra backends`);
# scipy_backend's tests (test_backends_scipy.py) need nothing beyond this project's core
# dependencies and always run, but everything in this file is skippable.
pytest.importorskip("cvxpy", reason="requires the 'backends' extra")
from optimlab.backends.cvxpy_backend import cvxpy_linprog, cvxpy_qp

_CLASSIC_LP = LinearProgram(c=[-3.0, -5.0], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])


def test_cvxpy_linprog_matches_our_simplex():
    ours = simplex(_CLASSIC_LP)
    oracle = cvxpy_linprog(_CLASSIC_LP)
    assert oracle.status == "optimal"
    np.testing.assert_allclose(ours.x, oracle.x, atol=1e-5)
    assert ours.objective == pytest.approx(oracle.objective, abs=1e-5)


def test_scipy_and_cvxpy_agree_with_each_other_on_a_harder_lp():
    """Independent solver stacks (HiGHS vs. Clarabel) agreeing is stronger evidence
    than either alone -- and stronger still that our own simplex also lands there.
    """
    lp = LinearProgram(
        c=[2.0, -3.0, 1.0],
        A_ub=[[1.0, 1.0, 1.0], [2.0, -1.0, 0.0]], b_ub=[10.0, 5.0],
        A_eq=[[1.0, 0.0, 1.0]], b_eq=[4.0],
    )
    ours = simplex(lp)
    scipy_oracle = scipy_linprog(lp)
    cvxpy_oracle = cvxpy_linprog(lp)
    assert scipy_oracle.status == cvxpy_oracle.status == ours.status == "optimal"
    assert ours.objective == pytest.approx(scipy_oracle.objective, abs=1e-5)
    assert ours.objective == pytest.approx(cvxpy_oracle.objective, abs=1e-5)


def test_cvxpy_qp_matches_our_equality_constrained_qp():
    P = np.diag([2.0, 4.0, 6.0])
    q = np.array([-1.0, -2.0, -3.0])
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])

    ours = equality_constrained_qp(P, q, A_eq, b_eq)
    oracle = cvxpy_qp(P, q, A_eq=A_eq, b_eq=b_eq)
    np.testing.assert_allclose(ours, oracle, atol=1e-6)


def test_cvxpy_qp_handles_mixed_equality_and_inequality_constraints():
    """The one case neither from-scratch QP function covers alone (both constraint
    types at once) -- cvxpy is the reach-for-this option there, not just an oracle.
    """
    P = np.eye(2)
    q = np.zeros(2)
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([2.0])
    A_ub = np.array([[1.0, 0.0]])
    b_ub = np.array([0.5])  # forces x0 <= 0.5, so x1 must pick up the rest of the sum

    x = cvxpy_qp(P, q, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
    np.testing.assert_allclose(A_eq @ x, b_eq, atol=1e-6)
    assert x[0] <= 0.5 + 1e-6
