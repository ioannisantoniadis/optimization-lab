"""Inverse problems: recovering an unknown cause (a sharp image, a system's physical
parameters) from indirect, noisy observations of its effect. Both problems here are
posed as ordinary optimization — `optimlab.core.Problem` for image deblurring,
`optimlab.optimizers.gauss_newton.NonlinearLeastSquaresProblem` for system
identification — reusing existing solvers rather than introducing inverse-problem-
specific machinery.
"""

from optimlab.inverse.deblurring import blur_image, deblurring_problem, gaussian_blur_matrix
from optimlab.inverse.system_id import oscillator_identification_problem, simulate_damped_oscillator

__all__ = [
    "blur_image",
    "deblurring_problem",
    "gaussian_blur_matrix",
    "oscillator_identification_problem",
    "simulate_damped_oscillator",
]
