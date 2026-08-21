from optimlab.optimizers.adaptive import adagrad, adam, rmsprop
from optimlab.optimizers.conjugate_gradient import conjugate_gradient
from optimlab.optimizers.gauss_newton import NonlinearLeastSquaresProblem, gauss_newton
from optimlab.optimizers.gradient_descent import gradient_descent
from optimlab.optimizers.line_search import backtracking_armijo, strong_wolfe_line_search
from optimlab.optimizers.linear_programming import LinearProgram, LPResult, simplex
from optimlab.optimizers.momentum import heavy_ball, nesterov
from optimlab.optimizers.newton import newton_method
from optimlab.optimizers.projected_gradient import projected_gradient
from optimlab.optimizers.quasi_newton import bfgs, lbfgs

#: Every from-scratch solver that speaks the `Problem -> OptimizeResult` interface,
#: name -> callable(problem, **kwargs) -> OptimizeResult. Used by the solver-arena
#: benchmarking harness (ROADMAP Phase 8) to run "all solvers against this problem"
#: without listing them out by hand at each call site. Not here, each for its own
#: reason tied to a differently-shaped problem (see each module's docstring):
#: `simplex` (LinearProgram), `conjugate_gradient` (a plain (A, b) linear system),
#: `gauss_newton` (NonlinearLeastSquaresProblem, residuals + Jacobian), and
#: `projected_gradient` (a Problem plus box bounds it needs as extra arguments).
ALL_SOLVERS = {
    "gradient_descent": gradient_descent,
    "heavy_ball": heavy_ball,
    "nesterov": nesterov,
    "adagrad": adagrad,
    "rmsprop": rmsprop,
    "adam": adam,
    "newton": newton_method,
    "bfgs": bfgs,
    "lbfgs": lbfgs,
}

__all__ = [
    "ALL_SOLVERS",
    "LPResult",
    "LinearProgram",
    "NonlinearLeastSquaresProblem",
    "adagrad",
    "adam",
    "backtracking_armijo",
    "bfgs",
    "conjugate_gradient",
    "gauss_newton",
    "gradient_descent",
    "heavy_ball",
    "lbfgs",
    "nesterov",
    "newton_method",
    "projected_gradient",
    "rmsprop",
    "simplex",
    "strong_wolfe_line_search",
]
