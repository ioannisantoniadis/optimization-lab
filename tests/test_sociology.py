import numpy as np
import pytest

from optimlab.problems.sociology import solve_fair_allocation


def test_symmetric_users_get_an_exactly_equal_split():
    """The one case with a known closed form: n identical users sharing a single
    resource -- by symmetry, and provably from the KKT conditions, proportional
    fairness gives each of them exactly capacity/n.
    """
    n, capacity = 4, 10.0
    A = np.ones((1, n))
    result = solve_fair_allocation(A, np.array([capacity]))

    assert result.converged
    np.testing.assert_allclose(result.x, capacity / n, atol=1e-4)


def test_a_user_sharing_two_resources_gets_less_than_users_sharing_one():
    """The actual fairness claim: a user contending for two constrained resources at
    once ends up squeezed relative to users who only compete for one.
    """
    A = np.array(
        [
            [1.0, 1.0, 0.0],  # resource 0: users 0 and 1
            [0.0, 1.0, 1.0],  # resource 1: users 1 and 2
        ]
    )
    capacities = np.array([10.0, 10.0])
    result = solve_fair_allocation(A, capacities)

    assert result.converged
    assert result.x[1] < result.x[0]
    assert result.x[1] < result.x[2]
    # symmetric problem (users 0 and 2 are interchangeable) -> equal shares
    assert result.x[0] == pytest.approx(result.x[2], abs=1e-4)


def test_resources_are_used_at_or_under_capacity():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    capacities = np.array([8.0, 12.0])
    result = solve_fair_allocation(A, capacities)

    usage = A @ result.x
    assert np.all(usage <= capacities + 1e-4)
