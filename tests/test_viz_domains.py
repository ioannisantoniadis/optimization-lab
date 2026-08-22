import numpy as np
import plotly.graph_objects as go

from optimlab.control.dynamic_programming import GridWorld, value_iteration
from optimlab.viz import (
    deblurring_figure,
    gridworld_figure,
    pinn_solution_figure,
    system_id_figure,
    trajectory_and_control_figure,
)


def test_deblurring_figure_has_three_panels_on_a_shared_color_scale():
    n = 8
    true_image = np.zeros((n, n))
    fig = deblurring_figure(true_image, true_image, true_image)
    assert isinstance(fig, go.Figure)
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 3
    assert all(hm.zmin == heatmaps[0].zmin and hm.zmax == heatmaps[0].zmax for hm in heatmaps)


def test_system_id_figure_has_observed_and_fitted_series():
    t = np.linspace(0, 1, 10)
    fig = system_id_figure(t, observed=t, fitted=t)
    names = {trace.name for trace in fig.data}
    assert names == {"observed", "fitted model"}


def test_trajectory_and_control_figure_has_one_trace_per_dimension():
    t = np.linspace(0, 1, 6)
    states = np.zeros((6, 2))
    controls = np.zeros((5, 1))
    fig = trajectory_and_control_figure(t, states, controls, state_labels=["a", "b"], control_labels=["u"])
    names = {trace.name for trace in fig.data}
    assert names == {"a", "b", "u"}


def test_trajectory_and_control_figure_rejects_mismatched_control_length():
    t = np.linspace(0, 1, 6)
    states = np.zeros((6, 2))
    controls = np.zeros((6, 1))  # wrong: should be len(t) - 1
    try:
        trajectory_and_control_figure(t, states, controls)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError from the shape mismatch")


def test_gridworld_figure_marks_the_goal_and_leaves_obstacles_out_of_the_heatmap():
    world = GridWorld(n_rows=3, n_cols=3, goal=(2, 2), obstacles={(1, 1)})
    V, policy, _n_iter = value_iteration(world)
    fig = gridworld_figure(world, V, policy)
    z = np.asarray(fig.data[0].z)
    assert np.isnan(z[1, 1])
    star_annotations = [a for a in fig.layout.annotations if a.text == "★"]
    assert len(star_annotations) == 1
    assert star_annotations[0].x == 2 and star_annotations[0].y == 2


def test_pinn_solution_figure_has_two_series():
    xs = np.linspace(0, 1, 10)
    fig = pinn_solution_figure(xs, predicted=xs, true=xs)
    names = {trace.name for trace in fig.data}
    assert names == {"analytic solution", "PINN prediction"}
