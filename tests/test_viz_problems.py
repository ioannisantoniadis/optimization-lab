import numpy as np
import plotly.graph_objects as go

from optimlab.problems.economics import efficient_frontier
from optimlab.problems.sociology import solve_fair_allocation
from optimlab.viz import efficient_frontier_figure, fair_allocation_figure


def test_efficient_frontier_figure_marks_the_minimum_variance_point():
    rng = np.random.default_rng(0)
    n = 4
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + 0.01 * np.eye(n)
    expected_returns = rng.uniform(0.02, 0.15, size=n)
    targets = np.linspace(0.03, 0.13, 11)
    frontier = efficient_frontier(cov, expected_returns, targets)

    fig = efficient_frontier_figure(frontier)
    assert isinstance(fig, go.Figure)
    star_trace = next(t for t in fig.data if t.name == "global minimum variance")
    min_idx = np.argmin(frontier.risks)
    assert star_trace.x[0] == frontier.risks[min_idx]
    assert star_trace.y[0] == frontier.target_returns[min_idx]


def test_fair_allocation_figure_has_allocation_and_usage_panels():
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    capacities = np.array([10.0, 10.0])
    result = solve_fair_allocation(A, capacities)

    fig = fair_allocation_figure(A, capacities, result.x)
    names = [t.name for t in fig.data]
    assert "usage" in names and "capacity" in names
    allocation_trace = fig.data[0]
    np.testing.assert_allclose(allocation_trace.y, result.x)
