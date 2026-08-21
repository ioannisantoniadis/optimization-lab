"""The flagship module (ROADMAP Phase 6): building real intuition for what a
high-dimensional non-convex loss surface looks like, and why gradient descent still
works on it anyway. Every earlier chapter stayed in a handful of dimensions —
`optimlab.highdim.nets` is what changes that, wrapping a small neural network's weights
into an ordinary `optimlab.core.Problem` so the from-scratch solvers already built in
Chapters 1-4 can train a real model with hundreds of parameters, no new optimizer
required.
"""

from optimlab.highdim.geometry import ball_shell_volume_fraction, pairwise_cosine_similarities
from optimlab.highdim.hessian_spectrum import (
    LanczosResult,
    hessian_vector_product,
    lanczos_eigenvalues,
)
from optimlab.highdim.loss_landscape import (
    LossSlice2D,
    filter_normalized_direction,
    linear_interpolation_loss,
    loss_landscape_slice,
)
from optimlab.highdim.mode_connectivity import (
    bezier_curve_problem,
    bezier_point,
    evaluate_curve_loss,
)
from optimlab.highdim.nets import MLPShape, forward, init_params, mlp_training_problem, unflatten
from optimlab.highdim.ntk import (
    NTKConcentrationResult,
    ntk_concentration_experiment,
    ntk_matrix,
    relative_frobenius_difference,
)
from optimlab.highdim.random_landscapes import (
    CriticalPointStats,
    critical_point_index_stats,
    sample_goe_eigenvalues,
)

__all__ = [
    "CriticalPointStats",
    "LanczosResult",
    "LossSlice2D",
    "MLPShape",
    "NTKConcentrationResult",
    "ball_shell_volume_fraction",
    "bezier_curve_problem",
    "bezier_point",
    "critical_point_index_stats",
    "evaluate_curve_loss",
    "filter_normalized_direction",
    "forward",
    "hessian_vector_product",
    "init_params",
    "lanczos_eigenvalues",
    "linear_interpolation_loss",
    "loss_landscape_slice",
    "mlp_training_problem",
    "ntk_concentration_experiment",
    "ntk_matrix",
    "pairwise_cosine_similarities",
    "relative_frobenius_difference",
    "sample_goe_eigenvalues",
    "unflatten",
]
