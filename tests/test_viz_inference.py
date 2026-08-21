import numpy as np
import plotly.graph_objects as go
import pytest

from optimlab.inference.em import em_gmm
from optimlab.inference.laplace import GaussianApprox
from optimlab.viz import gmm_figure, mcmc_trace_figure, posterior_figure


def test_posterior_figure_with_no_series_still_builds_an_empty_figure():
    fig = posterior_figure(x_range=(0.0, 1.0))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_posterior_figure_includes_one_trace_per_series_given():
    fig = posterior_figure(
        x_range=(0.0, 1.0),
        true_pdf=lambda xs: np.ones_like(xs),
        laplace=GaussianApprox(mean=np.array([0.5]), cov=np.array([[0.01]])),
        mcmc_samples=np.random.default_rng(0).normal(0.5, 0.1, size=500),
    )
    names = {t.name for t in fig.data}
    assert names == {"true posterior", "Laplace approximation", "MCMC samples"}


def test_mcmc_trace_figure_builds_two_panels():
    samples = np.random.default_rng(0).normal(size=1000)
    fig = mcmc_trace_figure(samples)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # trace line + marginal histogram


def test_gmm_figure_rejects_non_2d_data():
    X = np.random.default_rng(0).standard_normal((30, 3))
    result = em_gmm(X, n_components=2, seed=0)
    with pytest.raises(ValueError, match="2D"):
        gmm_figure(X, result)


def test_gmm_figure_has_one_scatter_and_one_ellipse_per_component():
    rng = np.random.default_rng(0)
    true_means = np.array([[0.0, 0.0], [8.0, 8.0]])
    X = np.concatenate([rng.multivariate_normal(m, np.eye(2), size=50) for m in true_means])
    result = em_gmm(X, n_components=2, seed=0)

    fig = gmm_figure(X, result)
    scatter_traces = [t for t in fig.data if t.mode == "markers" and t.name and t.name.startswith("component")]
    assert len(scatter_traces) == 2
    # each component also draws an ellipse (lines) + a mean marker (markers, no name)
    assert len(fig.data) == 2 * 3
