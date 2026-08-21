from optimlab.viz.compare import (
    add_trajectory,
    add_trajectory_3d,
    convergence_figure,
    race_figure,
    solver_color_map,
    surface_race_figure,
)
from optimlab.viz.constrained import central_path_figure, duality_gap_figure, kkt_geometry_figure
from optimlab.viz.highdim import (
    cosine_similarity_figure,
    curve_comparison_figure,
    hessian_spectrum_figure,
    loss_landscape_figure,
    ntk_concentration_figure,
    saddle_point_figure,
)
from optimlab.viz.inference import gmm_figure, mcmc_trace_figure, posterior_figure
from optimlab.viz.landscape import contour_figure, surface_figure, transform_values
from optimlab.viz.polytope import polytope_figure
from optimlab.viz.regression import (
    lasso_path_figure,
    regression_fit_figure,
    residuals_figure,
    ridge_path_figure,
    svd_conditioning_figure,
)

__all__ = [
    "add_trajectory",
    "add_trajectory_3d",
    "central_path_figure",
    "contour_figure",
    "convergence_figure",
    "cosine_similarity_figure",
    "curve_comparison_figure",
    "duality_gap_figure",
    "gmm_figure",
    "hessian_spectrum_figure",
    "kkt_geometry_figure",
    "lasso_path_figure",
    "loss_landscape_figure",
    "mcmc_trace_figure",
    "ntk_concentration_figure",
    "polytope_figure",
    "posterior_figure",
    "race_figure",
    "regression_fit_figure",
    "residuals_figure",
    "ridge_path_figure",
    "saddle_point_figure",
    "solver_color_map",
    "surface_figure",
    "surface_race_figure",
    "svd_conditioning_figure",
    "transform_values",
]
