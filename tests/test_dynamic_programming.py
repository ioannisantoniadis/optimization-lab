
from optimlab.control.dynamic_programming import GridWorld, value_iteration


def test_value_increases_toward_the_goal():
    world = GridWorld(n_rows=5, n_cols=5, goal=(4, 4))
    V, _policy, _n_iter = value_iteration(world)
    # value should be higher the closer a state is to the goal (Manhattan distance)
    assert V[3, 4] > V[0, 0]
    assert V[4, 3] > V[0, 0]
    assert V[3, 3] > V[1, 1]


def test_greedy_policy_navigates_around_obstacles_to_the_goal():
    world = GridWorld(n_rows=5, n_cols=5, goal=(4, 4), obstacles={(2, 2), (2, 3), (1, 3)})
    _V, policy, _n_iter = value_iteration(world)

    state = (0, 0)
    visited = [state]
    for _ in range(30):
        if state == world.goal:
            break
        state, _reward = world.step(state, policy[state])
        visited.append(state)

    assert state == world.goal
    assert all(s not in world.obstacles for s in visited)


def test_value_iteration_converges_before_max_iter_on_an_easy_grid():
    world = GridWorld(n_rows=3, n_cols=3, goal=(2, 2))
    _V, _policy, n_iter = value_iteration(world, max_iter=1000)
    assert n_iter < 1000
