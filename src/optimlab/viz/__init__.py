from optimlab.viz.compare import (
    add_trajectory,
    add_trajectory_3d,
    convergence_figure,
    race_figure,
    solver_color_map,
    surface_race_figure,
)
from optimlab.viz.constrained import central_path_figure, duality_gap_figure, kkt_geometry_figure
from optimlab.viz.control import gridworld_figure, trajectory_and_control_figure
from optimlab.viz.highdim import (
    cosine_similarity_figure,
    curve_comparison_figure,
    hessian_spectrum_figure,
    loss_landscape_figure,
    ntk_concentration_figure,
    saddle_point_figure,
)
from optimlab.viz.inference import gmm_figure, mcmc_trace_figure, posterior_figure
from optimlab.viz.inverse import deblurring_figure, system_id_figure
from optimlab.viz.landscape import contour_figure, surface_figure, transform_values
from optimlab.viz.ml import pinn_solution_figure
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
    "deblurring_figure",
    "duality_gap_figure",
    "gmm_figure",
    "gridworld_figure",
    "hessian_spectrum_figure",
    "kkt_geometry_figure",
    "lasso_path_figure",
    "loss_landscape_figure",
    "mcmc_trace_figure",
    "ntk_concentration_figure",
    "pinn_solution_figure",
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
    "system_id_figure",
    "trajectory_and_control_figure",
    "transform_values",
]
