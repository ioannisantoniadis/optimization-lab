from optimlab.viz.compare import (
    add_trajectory,
    add_trajectory_3d,
    convergence_figure,
    race_figure,
    solver_color_map,
    surface_race_figure,
)
from optimlab.viz.constrained import central_path_figure, duality_gap_figure, kkt_geometry_figure
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
    "duality_gap_figure",
    "kkt_geometry_figure",
    "lasso_path_figure",
    "polytope_figure",
    "race_figure",
    "regression_fit_figure",
    "residuals_figure",
    "ridge_path_figure",
    "solver_color_map",
    "surface_figure",
    "surface_race_figure",
    "svd_conditioning_figure",
    "transform_values",
]
