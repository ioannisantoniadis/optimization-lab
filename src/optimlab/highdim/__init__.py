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
from optimlab.highdim.nets import MLPShape, forward, init_params, mlp_training_problem, unflatten
from optimlab.highdim.random_landscapes import (
    CriticalPointStats,
    critical_point_index_stats,
    sample_goe_eigenvalues,
)

__all__ = [
    "CriticalPointStats",
    "LanczosResult",
    "MLPShape",
    "ball_shell_volume_fraction",
    "critical_point_index_stats",
    "forward",
    "hessian_vector_product",
    "init_params",
    "lanczos_eigenvalues",
    "mlp_training_problem",
    "pairwise_cosine_similarities",
    "sample_goe_eigenvalues",
    "unflatten",
]
