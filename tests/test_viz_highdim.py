import numpy as np
import plotly.graph_objects as go

from optimlab.highdim.loss_landscape import LossSlice2D
from optimlab.highdim.ntk import NTKConcentrationResult
from optimlab.highdim.random_landscapes import critical_point_index_stats
from optimlab.viz import (
    cosine_similarity_figure,
    curve_comparison_figure,
    hessian_spectrum_figure,
    loss_landscape_figure,
    ntk_concentration_figure,
    saddle_point_figure,
)


def test_saddle_point_figure_has_two_series():
    stats = critical_point_index_stats(dims=[1, 2, 4], n_samples=500, seed=0)
    fig = saddle_point_figure(stats)
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert names == {"P(local minimum)", "P(saddle)"}
    assert fig.layout.yaxis.type == "log"


def test_cosine_similarity_figure_has_one_histogram_per_dim():
    sims = {3: np.random.default_rng(0).uniform(-1, 1, 100), 50: np.random.default_rng(1).uniform(-0.1, 0.1, 100)}
    fig = cosine_similarity_figure(sims)
    assert len(fig.data) == 2
    assert all(isinstance(t, go.Histogram) for t in fig.data)


def test_hessian_spectrum_figure_sorts_eigenvalues_descending():
    eigenvalues = np.array([0.1, 5.0, -0.2, 3.0])
    fig = hessian_spectrum_figure(eigenvalues)
    y = np.asarray(fig.data[0].y)
    assert np.all(np.diff(y) <= 0)
    assert y[0] == 5.0


def test_loss_landscape_figure_marks_the_base_point_at_the_origin():
    A, B = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    Z = A**2 + B**2
    fig = loss_landscape_figure(LossSlice2D(A=A, B=B, Z=Z))
    star_trace = next(t for t in fig.data if t.name == "trained minimum")
    assert star_trace.x[0] == 0 and star_trace.y[0] == 0


def test_curve_comparison_figure_has_one_trace_per_curve():
    xs = np.linspace(0, 1, 10)
    fig = curve_comparison_figure(
        {"straight": (xs, xs), "curved": (xs, xs**2)}, title="test", xaxis_title="t", yaxis_title="loss"
    )
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"straight", "curved"}


def test_ntk_concentration_figure_has_a_shaded_band_and_a_line():
    result = NTKConcentrationResult(
        widths=np.array([4, 64, 1024]),
        mean_relative_diff=np.array([0.5, 0.2, 0.05]),
        std_relative_diff=np.array([0.1, 0.05, 0.01]),
    )
    fig = ntk_concentration_figure(result)
    assert len(fig.data) == 2
    line_trace = next(t for t in fig.data if t.name == "mean relative NTK difference")
    np.testing.assert_allclose(line_trace.y, result.mean_relative_diff)
