"""Machine learning, closing the loop this repo has been building toward: manual
backpropagation (cross-checked against `optimlab.core`'s JAX autodiff on the identical
network) and physics-informed neural networks (training a network purely from a
differential equation's own residual, reusing `optimlab.highdim.nets`' MLP-as-`Problem`
machinery with one addition — derivatives of the network's output with respect to its
*input*, not just its parameters).
"""

from optimlab.ml.backprop import manual_mlp_gradient
from optimlab.ml.pinn import ode_pinn_problem, predict

__all__ = ["manual_mlp_gradient", "ode_pinn_problem", "predict"]
