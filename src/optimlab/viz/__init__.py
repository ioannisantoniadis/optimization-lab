import importlib

from optimlab.viz.arena import arena_figure
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
from optimlab.viz.economics import efficient_frontier_figure
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
from optimlab.viz.sociology import fair_allocation_figure

#: `optimlab.viz.highdim`/`inverse`/`ml` hard-import jax at module level (jax has no
#: WebAssembly/Pyodide build), so importing them eagerly here would break `optimlab.viz`
#: for every caller — including marimo notebooks exported to run client-side that never
#: touch a jax-only figure. Resolved lazily via PEP 562 so `from optimlab.viz import X`
#: still works for these names; only touching one actually imports its jax-dependent
#: module.
_LAZY = {
    "cosine_similarity_figure": "optimlab.viz.highdim",
    "curve_comparison_figure": "optimlab.viz.highdim",
    "hessian_spectrum_figure": "optimlab.viz.highdim",
    "loss_landscape_figure": "optimlab.viz.highdim",
    "ntk_concentration_figure": "optimlab.viz.highdim",
    "saddle_point_figure": "optimlab.viz.highdim",
    "deblurring_figure": "optimlab.viz.inverse",
    "system_id_figure": "optimlab.viz.inverse",
    "pinn_solution_figure": "optimlab.viz.ml",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "add_trajectory",
    "add_trajectory_3d",
    "arena_figure",
    "central_path_figure",
    "contour_figure",
    "convergence_figure",
    "cosine_similarity_figure",
    "curve_comparison_figure",
    "deblurring_figure",
    "duality_gap_figure",
    "efficient_frontier_figure",
    "fair_allocation_figure",
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
