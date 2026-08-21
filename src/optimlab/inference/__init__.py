"""Statistical estimation built on the same solvers as the rest of the repo: MLE/MAP
are ordinary `optimlab.core.Problem`s in disguise (Chapter 3's least squares is one
specific instance — the Gaussian-noise MLE), the Laplace approximation reuses the same
autodiff Hessian `optimlab.optimizers.newton` does, and EM is the one genuinely new
algorithm here (an exact-maximization-of-a-surrogate iteration, not a gradient method
at all).
"""

from optimlab.inference.em import GMMResult, em_gmm
from optimlab.inference.laplace import GaussianApprox, laplace_approximation
from optimlab.inference.mcmc import MCMCResult, metropolis_hastings
from optimlab.inference.mle import map_fit, mle_fit

__all__ = [
    "GMMResult",
    "GaussianApprox",
    "MCMCResult",
    "em_gmm",
    "laplace_approximation",
    "map_fit",
    "metropolis_hastings",
    "mle_fit",
]
