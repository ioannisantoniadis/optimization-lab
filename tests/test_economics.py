import numpy as np
import pytest

from optimlab.problems.economics import efficient_frontier, minimum_variance_portfolio


def _synthetic_market(seed=0, n_assets=5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + 0.01 * np.eye(n_assets)
    expected_returns = rng.uniform(0.02, 0.15, size=n_assets)
    return cov, expected_returns


def test_portfolio_weights_sum_to_one_and_hit_the_target_return():
    cov, expected_returns = _synthetic_market()
    target = 0.08
    w = minimum_variance_portfolio(cov, expected_returns, target)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert w @ expected_returns == pytest.approx(target, abs=1e-8)


def test_portfolio_matches_an_independent_qp_solver():
    """Cross-check against cvxpy's QP solver on the identical equality-constrained
    problem -- an entirely different solver stack landing on the same weights is
    stronger evidence than either alone.
    """
    pytest.importorskip("cvxpy", reason="requires the 'backends' extra")
    from optimlab.backends.cvxpy_backend import cvxpy_qp

    cov, expected_returns = _synthetic_market()
    target = 0.08
    n = cov.shape[0]
    A_eq = np.vstack([np.ones(n), expected_returns])
    b_eq = np.array([1.0, target])

    w = minimum_variance_portfolio(cov, expected_returns, target)
    w_cvxpy = cvxpy_qp(cov, np.zeros(n), A_eq=A_eq, b_eq=b_eq)
    np.testing.assert_allclose(w, w_cvxpy, atol=1e-6)


def test_efficient_frontier_is_a_hyperbola_with_a_single_minimum():
    """The classic Markowitz shape: risk decreases toward a single global-minimum-
    variance portfolio, then increases again on the other side -- not monotonic, and
    not flat.
    """
    cov, expected_returns = _synthetic_market()
    targets = np.linspace(0.03, 0.13, 21)
    frontier = efficient_frontier(cov, expected_returns, targets)

    min_idx = np.argmin(frontier.risks)
    assert 0 < min_idx < len(targets) - 1  # the minimum isn't at either sweep endpoint
    assert np.all(np.diff(frontier.risks[: min_idx + 1]) < 0)  # strictly decreasing before
    assert np.all(np.diff(frontier.risks[min_idx:]) > 0)  # strictly increasing after
